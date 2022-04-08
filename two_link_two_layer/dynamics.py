import numpy as np
import torch
import torch.nn as nn
from math import sin
from math import cos
from torch import randn as randn

devicecuda = torch.device("cuda:0")
devicecpu = torch.device("cpu")
torch.set_default_dtype(torch.float)

torch.manual_seed(0)

# class for the two link dynamics
class Dynamics():
    # constructor to initialize a Dynamics object
    def __init__(self,alpha,betar,betadel,gammath,gammaw,gammav,tauN=0.1,phiN=0.01,phiDN=0.05,phiDDN=0.05,l=5,useNN=True,useYth=True,useBias=True,useCuda=False):
        """
        Initialize the dynamics \n
        Inputs:
        -------
        \t alpha:  error gain \n
        \t betar:  filtered error gain \n
        \t betadel:  filtered error robust gain \n
        \t gammath: theta parameter update gain \n
        \t gammaw: W parameter update gain \n
        \t kCL: CL parameter update gain \n
        \t tauN: input noise is a disturbance \n
        \t phiN: angle measurement noise \n
        \t phiDN: velocity measurement noise \n
        \t phiDDN: acceleration measurement noise \n
        
        Returns:
        -------
        """
        self.useCuda = useCuda
        # gains
        self.ll = l
        self.L = l+1
        
        if not useBias:
            self.L -= 1
        self.alpha = self.usecuda(alpha)
        self.betar = self.usecuda(betar)
        self.betadel = self.usecuda(betadel)
        
        self.Gammath = self.usecuda(gammath*torch.eye(5,dtype=torch.float))
        self.Gammaw = self.usecuda(gammaw*torch.eye(self.L,dtype=torch.float))
        self.Gammav = self.usecuda(gammav*torch.eye(5,dtype=torch.float))
        self.useNN = useNN
        self.useYth = useYth

        # desired trajectory parameters
        self.phidMag = self.usecuda(torch.tensor([torch.pi/8,torch.pi/4],dtype=torch.float)) # amplitude of oscillations in rad
        self.fphid = 0.2 # frequency in Hz
        self.aphid = torch.pi/2 # phase shift in rad
        self.bphid = self.usecuda(torch.tensor([torch.pi/2,torch.pi/4],dtype=torch.float)) # bias in rad


        # noise
        self.tauNM=tauN/3.0
        self.phiNM=phiN/3.0
        self.phiDNM=phiDN/3.0
        self.phiDDNM=phiDDN/3.0

        # rigid body parameters
        self.m = self.usecuda(torch.tensor([2.0,2.0],dtype=torch.float)) # mass in kg
        self.l = self.usecuda(torch.tensor([0.5,0.5],dtype=torch.float)) # length in m
        self.mBnds = self.usecuda(torch.tensor([1.0,3.0],dtype=torch.float)) # mass bounds in kg
        self.lBnds = self.usecuda(torch.tensor([0.25,0.75],dtype=torch.float)) # length bounds in m
        self.g = 9.8 # gravity in m/s^2

        # unknown structured dynamics
        self.theta = self.usecuda(self.getTheta(self.m,self.l)) # initialize theta
        self.thetaH = self.usecuda(self.getTheta(self.mBnds[0]*self.usecuda(torch.ones(2,dtype=torch.float)),self.lBnds[0]*self.usecuda(torch.ones(2,dtype=torch.float)))) # initialize theta estimate to the lowerbounds

        # unknown unstructured dynamics
        #gaussian spaced throughout desired state space
        self.WH = self.usecuda((0.01/self.L)*randn(self.L,2))
        self.VH = self.usecuda((0.01/self.ll)*randn(5,self.ll))
        self.maxInner = 10.0

        # desired trajectory parameters
        self.phidMag = self.usecuda(torch.tensor([torch.pi/8,torch.pi/4],dtype=torch.float)) # amplitude of oscillations in rad
        self.fphid = 0.2 # frequency in Hz
        self.aphid = torch.pi/2 # phase shift in rad
        self.bphid = self.usecuda(torch.tensor([torch.pi/2,torch.pi/4],dtype=torch.float)) # bias in rad

        # initialize state
        self.phi,_,_ = self.getDesiredState(0.0) # set the initial angle to the initial desired angle
        self.phi = self.usecuda(self.phi)
        self.phiD = self.usecuda(torch.zeros(2,dtype=torch.float))
        self.phiDD = self.usecuda(torch.zeros(2,dtype=torch.float))
        self.tau = self.usecuda(torch.zeros(2,dtype=torch.float))
        self.phiN = self.phiNM*self.usecuda(randn(1))
        self.phiDN = self.phiDNM*self.usecuda(randn(1))
        self.phiDDN = self.phiDDNM*self.usecuda(randn(1))
        self.tauN = self.tauNM*self.usecuda(randn(1))
    
    def usecuda(self,tensor: torch.Tensor):
        if self.useCuda and torch.cuda.is_available():
            tensor = tensor.to(devicecuda)
        return tensor

    def getTheta(self,m,l):
        """
        Inputs:
        -------
        \t m: link masses \n
        \t l: link lengths \n
        
        Returns:
        -------
        \t theta: parameters
        """
        theta = torch.tensor([(m[0]+m[1])*l[0]**2+m[1]*l[1]**2,
                          m[1]*l[0]*l[1],
                          m[1]*l[1]**2,
                          (m[0]+m[1])*l[0],
                          m[1]*l[1]],dtype=torch.float)
        return theta

    def getDesiredState(self,t):
        """
        Determines the desired state of the system \n
        Inputs:
        -------
        \t t: time \n
        
        Returns:
        -------
        \t phid:   desired angles \n
        \t phiDd:  desired angular velocity \n
        \t phiDDd: desired angular acceleration
        """
        # desired angles
        phid = torch.tensor([self.phidMag[0]*sin(2*torch.pi*self.fphid*t-self.aphid)-self.bphid[0],
                         self.phidMag[1]*sin(2*torch.pi*self.fphid*t-self.aphid)+self.bphid[1]],dtype=torch.float)

        #desired angular velocity
        phiDd = torch.tensor([2*torch.pi*self.fphid*self.phidMag[0]*cos(2*torch.pi*self.fphid*t-self.aphid),
                          2*torch.pi*self.fphid*self.phidMag[1]*cos(2*torch.pi*self.fphid*t-self.aphid)],dtype=torch.float)

        #desired angular acceleration
        phiDDd = torch.tensor([-((2*torch.pi*self.fphid)**2)*self.phidMag[0]*sin(2*torch.pi*self.fphid*t-self.aphid),
                           -((2*torch.pi*self.fphid)**2)*self.phidMag[1]*sin(2*torch.pi*self.fphid*t-self.aphid)],dtype=torch.float)
        
        return phid,phiDd,phiDDd

    # returns the inertia matrix
    def getM(self,m,l,phi):
        """
        Determines the inertia matrix \n
        Inputs:
        -------
        \t m:   link masses \n
        \t l:   link lengths \n
        \t phi: angles \n
        
        Returns:
        -------
        \t M: inertia matrix
        """
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]
        c2 = cos(phi[1])
        M = torch.tensor([[m1*l1**2+m2*(l1**2+2*l1*l2*c2+l2**2),m2*(l1*l2*c2+l2**2)],
                      [m2*(l1*l2*c2+l2**2),m2*l2**2]],dtype=torch.float)
        return self.usecuda(M)

    # returns the centripetal coriolis matrix
    def getC(self,m,l,phi,phiD):
        """
        Determines the centripetal coriolis matrix \n
        Inputs:
        -------
        \t m:    link masses \n
        \t l:    link lengths \n
        \t phi:  angles \n
        \t phiD: angular velocities \n
        
        Returns:
        -------
        \t C: cetripetal coriolis matrix
        """
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]
        s2 = sin(phi[1])
        phi1D = phiD[0]
        phi2D = phiD[1]
        C = torch.tensor([-2*m2*l1*l2*s2*phi1D*phi2D-m2*l1*l2*s2*phi2D**2,
                      m2*l1*l2*s2*phi1D**2],dtype=torch.float)
        return self.usecuda(C)

    # returns the gravity matrix
    def getG(self,m,l,phi):
        """
        Determines the gravity matrix \n
        Inputs:
        -------
        \t m:   link masses \n
        \t l:   link lengths \n
        \t phi: angles \n
        
        Returns:
        -------
        \t G: gravity matrix
        """
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]
        c1 = cos(phi[0])
        c12 = cos(phi[0]+phi[1])
        G = torch.tensor([(m1+m2)*self.g*l1*c1+m2*self.g*l2*c12,
                      m2*self.g*l2*c12],dtype=torch.float)
        return self.usecuda(G)

    # returns the inertia matrix regressor
    def getYM(self,vphi,phi):
        """
        Determines the inertia matrix regressor \n
        Inputs:
        -------
        \t vphi: phiDDd+alpha*eD or phiDD \n
        \t phi:  angles \n
        
        Returns:
        -------
        \t YM: inertia matrix regressor
        """
        vphi1 = vphi[0]
        vphi2 = vphi[1]
        c2 = cos(phi[1])
        YM = torch.tensor([[vphi1,2*c2*vphi1+c2*vphi2,vphi2,0.0,0.0],
                       [0.0,c2*vphi1,vphi1+vphi2,0.0,0.0]],dtype=torch.float)
        return self.usecuda(YM)

    # returns the centripetal coriolis matrix regressor
    def getYC(self,phi,phiD):
        """
        Determines the centripetal coriolis matrix regressor \n
        Inputs:
        -------
        \t phi:  angles \n
        \t phiD: angular velocity \n
        
        Returns:
        -------
        \t YC: centripetal coriolis matrix regressor
        """
        s2 = sin(phi[1])
        phi1D = phiD[0]
        phi2D = phiD[1]
        YC = torch.tensor([[0.0,-2*s2*phi1D*phi2D-s2*phi2D**2,0.0,0.0,0.0],
                       [0.0,s2*phi1D**2,0.0,0.0,0.0]],dtype=torch.float)
        return self.usecuda(YC)

    # returns the gravity matrix regressor
    def getYG(self,phi):
        """
        Determines the gravity matrix regressor \n
        Inputs:
        -------
        \t phi: angles \n
        
        Returns:
        -------
        \t YG: gravity matrix regressor
        """
        c1 = cos(phi[0])
        c12 = cos(phi[0]+phi[1])
        YG = torch.tensor([[0.0,0.0,0.0,self.g*c1,self.g*c12],
                     [0.0,0.0,0.0,0.0,self.g*c12]],dtype=torch.float)
        return self.usecuda(YG)

    # returns the inertia matrix derivative regressor
    def getYMD(self,phi,phiD,r):
        """
        Determines the inertia derivative regressor \n
        Inputs:
        -------
        \t phi:  angles \n
        \t phiD: angular velocoty \n
        \t r:    filtered tracking error \n
        
        Returns:
        -------
        \t YMD: inertia matrix derivative regressor
        """

        s2 = sin(phi[1])
        phi2D = phiD[1]
        r1 = r[0]
        r2 = r[1]
        YMD = torch.tensor([[0.0,-2*s2*phi2D*r1-s2*phi2D*r2,0.0,0.0,0.0],
                        [0.0,-s2*phi2D*r1,0.0,0.0,0.0]],dtype=torch.float)
        return self.usecuda(YMD)

    def getsigma(self,phi,phiD,V):
        """
        Determines the basis for the system \n
        Inputs:
        -------
        \t phi: position \n
        \t phiD: velocity \n
        \t V: inner weights \n
        
        Returns:
        -------
        \t sigma: basis \n
        """
        sigma = self.usecuda(torch.ones(self.L,dtype=torch.float))
        xi = self.usecuda(torch.ones(5,dtype=torch.float))
        xi[0] = phi[0]
        xi[1] = phi[1]
        xi[2] = phiD[0]
        xi[3] = phiD[1]
        Phi = V.T@xi
        for ii in range(self.ll):
            sigma[ii] = torch.exp(-Phi[ii]**2)
        return sigma

    def getsigmaPrime(self,phi,phiD,V):
        """
        Determines the basis for the system \n
        Inputs:
        -------
        \t phi: position \n
        \t phiD: velocity \n
        \t V: inner weights \n
        
        Returns:
        -------
        \t sigma: basis \n
        """
        sigmaPrimeInner = self.usecuda(torch.zeros(self.ll,dtype=torch.float))
        sigmaPrime = self.usecuda(torch.zeros((self.L,self.ll),dtype=torch.float))
        xi = self.usecuda(torch.ones(5,dtype=torch.float))
        xi[0] = phi[0]
        xi[1] = phi[1]
        xi[2] = phiD[0]
        xi[3] = phiD[1]
        Phi = V.T@xi
        for ii in range(self.ll):
            sigmaPrimeInner[ii] = -2.0*Phi[ii]*torch.exp(-Phi[ii]**2)
        sigmaPrime[:self.ll,:] = torch.diag(sigmaPrimeInner)
        return sigmaPrime

    def getState(self,t):
        """
        Returns the state of the system and parameter estimates \n
        Inputs:
        -------
        \t t: time \n
        
        Returns:
        -------
        \t phi:    angles \n
        \t phiD:   angular velocity \n
        \t phiDD:  angular acceleration \n
        \t thetaH: structured parameter estimate \n
        \t WH: unstructured parameter estiate \n
        \t VH: unstructured parameter estiate \n
        """
        return self.phi,self.phiD,self.phiDD,self.thetaH,self.WH,self.VH

    #returns the error state
    def getErrorState(self,t):
        """
        Returns the errors \n
        Inputs:
        -------
        \t t:  time \n
        
        Returns:
        -------
        \t em:     measured tracking error \n
        \t eDm:    measured tracking error derivative \n
        \t rm:    measured filtered tracking error \n
        \t phim:     measured angle \n
        \t phiDm:    measured velocity \n
        \t phiDDm:    measured acceleration \n
        \t thetaH: structured estimate \n
        \t WH: unstructured estimate \n
        """
        
        # get the desired state
        phid,phiDd,phiDDd = self.getDesiredState(t)

        phid = self.usecuda(phid)
        phiDd = self.usecuda(phiDd)
        phiDDd = self.usecuda(phiDDd)

        # get the tracking error
        em = phid - self.phi
        eDm = phiDd - self.phiD
        rm = eDm+self.alpha@em

        return em,eDm,rm

    def getTau(self,t,phi,phiD,thetaH,WH,VH):
        """
        Returns tau \n
        Inputs:
        -------
        \t t: time \n
        \t phi: angles \n
        \t phiD: velocity \n
        \t thetaH: structured estimate \n
        \t WH: unstructured estimate outer \n
        \t VH: unstructured estimate inner \n
        
        Returns:
        -------
        \t tau: input \n
        \t tauff: feedforward component \n
        \t taufb: feedback component \n
        \t Y: regressor \n
        \t sigma: basis \n
        \t r: filtered error \n

        """

        #get desired state
        phid,phiDd,phiDDd = self.getDesiredState(t)
        phid = self.usecuda(phid)
        phiDd = self.usecuda(phiDd)
        phiDDd = self.usecuda(phiDDd)

        #calculate error
        e = phid - phi
        eD = phiDd - phiD
        r = eD + self.alpha@e
        vphi = phiDDd + self.alpha@eD

        # get the regressors
        vphi = phiDDd + self.alpha@eD
        YM = self.getYM(vphi,phi)
        YC = self.getYC(phi,phiD)
        YG = self.getYG(phi)
        YMD = self.getYMD(phi,phiD,r)
        Y = YM+YC+YG+0.5*YMD

        #get sigma
        sigma = self.getsigma(phi,phiD,VH)

        #calculate the controller and update law
        tauff = self.usecuda(torch.zeros(2))
        if self.useYth:
            tauff+=Y@thetaH
        if self.useNN:
            tauff+=WH.T@sigma
        taufb = e+self.betar@r+self.betadel@torch.sign(r)
        tau = tauff + taufb

        return tau,tauff,taufb,Y,sigma,r

    def getTaud(self,phi,phiD):
        # return np.zeros(2)
        taudFriction = self.usecuda(torch.tensor([5.0*torch.tanh(5.0*phiD[0])*phiD[0]**2+5.0*phiD[0],5.0*torch.tanh(5.0*phiD[1])*phiD[1]**2+5.0*phiD[1]]))
        taudSpring = self.usecuda(torch.tensor([5.0*phi[0]**3+5.0*phi[0],5.0*phi[1]**3+5.0*phi[1]]))
        taudForce = self.usecuda(torch.tensor([2.0*self.g*cos(phi[0])+self.g*cos(phi[0]+phi[1]),self.g*cos(phi[0]+phi[1])]))
        return taudFriction+taudSpring+taudForce

    def getfunc(self,phi,phiD,tau):
        M = self.getM(self.m,self.l,phi)
        C = self.getC(self.m,self.l,phi,phiD)
        G = self.getG(self.m,self.l,phi)
        taud = self.getTaud(phi,phiD)
        phiDD = torch.linalg.inv(M)@(-C-G-taud+self.tauN+tau)
        return phiDD

    def getfuncComp(self,phi,phiD,phiDD,tau,thetaH,WH,VH):
        """
        Dynamics callback for function approx compare \n
        Inputs:
        -------
        \t x: position \n
        \t WH: estimates \n
        
        Returns:
        -------
        \t f: value of dynamics \n
        \t fH: approximate of dynamics \n
        """
        
        #calculate the actual
        M = self.getM(self.m,self.l,phi)
        C = self.getC(self.m,self.l,phi,phiD)
        G = self.getG(self.m,self.l,phi)
        taud = self.getTaud(phi,phiD)
        
        f = M@phiDD+C+G+taud

        # calculate the approximate
        # get regressors
        YM = self.getYM(phiDD,phi)
        YC = self.getYC(phi,phiD)
        YG = self.getYG(phi)
        Y = YM+YC+YG

        #get sigma
        sigmam = self.getsigma(phi,phiD,VH)

        # get the function approximate
        fH = self.usecuda(torch.zeros(2))
        if self.useYth:
            fH+=Y@thetaH
        if self.useNN:
            fH += WH.T@sigmam

        return f,fH

    def getf(self,t,X):
        """
        Dynamics callback \n
        Inputs:
        -------
        \t t:  time \n
        \t X:  stacked phi,phiD,thetaH \n
        
        Returns:
        -------
        \t XD: derivative approximate at time \n
        \t tau: control input at time \n
        """

        phi = X[0:2]
        phiD = X[2:4]
        thetaH = X[4:9]
        WH = torch.reshape(X[9:9+2*self.L],(2,self.L)).T
        VH = torch.reshape(X[9+2*self.L:],(5,self.ll))

        # get the noisy measurements for the control design
        phim = phi+self.phiN
        phiDm = phiD+self.phiDN
        xim = self.usecuda(torch.ones(5,dtype=torch.float))
        xim[0] = phim[0]
        xim[1] = phim[1]
        xim[2] = phiDm[0]
        xim[3] = phiDm[1]

        # get the input and regressors
        taum,_,_,Ym,sigmam,rm = self.getTau(t,phim,phiDm,thetaH,WH,VH)

        sigmaPm = self.getsigmaPrime(phim,phiDm,VH)

        #parameter updates
        thetaHD = self.Gammath@Ym.T@rm
        WHD = self.Gammaw@(torch.outer(sigmam,rm))
        VHD = self.Gammav@torch.outer(xim,rm)@(WH.T@sigmaPm)

        # get the dynamics using the unnoised position and velocity but designed input
        phiDD = self.getfunc(phi,phiD,taum)

        #calculate and return the derivative
        XD = torch.zeros_like(X)
        XD[0:2] = phiD
        XD[2:4] = phiDD
        XD[4:9] = thetaHD
        XD[9:9+2*self.L] = torch.reshape(WHD.T,(2*self.L,))
        XD[9+2*self.L:] = torch.reshape(VHD,(5*self.ll,))

        return XD,taum

    #classic rk4 method
    def rk4(self,dt,t,X):
        """
        Classic rk4 method \n
        Inputs:
        -------
        \t dt:  total time step for interval \n
        \t t:  time \n
        \t X:  stacked x,WH \n
        
        Returns:
        -------
        \t XD: derivative approximate over total interval \n
        \t tau: control input approximate over total interval \n
        \t Xh: integrated value \n
        """

        k1,tau1 = self.getf(t,X)
        k2,tau2 = self.getf(t+0.5*dt,X+0.5*dt*k1)
        k3,tau3 = self.getf(t+0.5*dt,X+0.5*dt*k2)
        k4,tau4 = self.getf(t+dt,X+dt*k3)
        XD = (1.0/6.0)*(k1+2.0*k2+2.0*k3+k4)
        taum = (1.0/6.0)*(tau1+2.0*tau2+2.0*tau3+tau4)

        return XD,taum

    # take a step of the dynamics
    def step(self,dt,t):
        """
        Steps the internal state using the dynamics \n
        Inputs:
        -------
        \t dt: time step \n
        \t t:  time \n
        
        Returns:
        -------
        """
        
        # update the internal state
        X = self.usecuda(torch.zeros(2+2+5+2*self.L+5*self.ll,dtype=torch.float))
        X[0:2] = self.phi
        X[2:4] = self.phiD
        X[4:9] = self.thetaH
        X[9:9+2*self.L] = torch.reshape(self.WH.T,(2*self.L,))
        X[9+2*self.L:] = torch.reshape(self.VH,(5*self.ll,))

        #get the derivative and input from rk
        XD,taum = self.rk4(dt,t,X)
        phiD = XD[0:2]
        phiDD = XD[2:4]
        thetaHD = XD[4:9]
        # WHD = np.reshape(XD[9:9+2*self.L],(2,self.L)).T
        # VHD = np.reshape(XD[9+2*self.L:],(5,self.ll))

        #do the projection
        XNN = X[9:]
        XDNN = XD[9:]
        for ii in range(len(XNN)):
            if abs(XNN[ii]+dt*XDNN[ii]) > self.maxInner:
                XDNN[ii] = 0.0

        WHD = torch.reshape(XDNN[0:2*self.L],(2,self.L)).T
        VHD = torch.reshape(XDNN[2*self.L:],(5,self.ll))

        # update the internal state
        # X(ii+1) = X(ii) + dt*f(X)
        self.phi += dt*phiD
        self.phiD += dt*phiDD
        self.thetaH += dt*thetaHD
        self.WH += dt*WHD
        self.VH += dt*VHD

        self.phiN = self.phiNM*self.usecuda(randn(1))
        self.phiDN = self.phiDNM*self.usecuda(randn(1))
        self.phiDDN = self.phiDDNM*self.usecuda(randn(1))
        self.tauN = self.tauNM*self.usecuda(randn(1))