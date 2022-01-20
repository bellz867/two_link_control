import numpy as np
from math import sin
from math import cos

# class for the two link dynamics
class Dynamics():
    # constructor to initialize a Dynamics object
    def __init__(self,alpha=0.25*np.ones(2,dtype=np.float32),beta=0.1*np.ones(2,dtype=np.float32),gamma=0.01*np.ones(5,dtype=np.float32)):
        """
        Initialize the dynamics \n
        Inputs:
        -------
        \t alpha: error gain \n
        \t beta:  filtered error gain \n
        \t gamma: parameter update gain \n
        
        Returns:
        -------
        """
        # gains
        self.alpha = np.diag(alpha)
        self.beta = np.diag(beta)
        self.Gamma = np.diag(gamma)

        # rigid body parameters
        self.m = np.array([2.0,2.0],dtype=np.float32) # mass in kg
        self.l = np.array([0.5,0.5],dtype=np.float32) # length in m
        self.mBnds = np.array([1.0,3.0],dtype=np.float32) # mass bounds in kg
        self.lBnds = np.array([0.25,0.75],dtype=np.float32) # length bounds in m
        self.g = 9.8 # gravity in m/s^2

        # unknown parameters
        self.theta = self.getTheta(self.m,self.l) # initialize theta
        self.thetaH = self.getTheta(self.mBnds[0]*np.ones(2,dtype=np.float32),self.lBnds[0]*np.ones(2,dtype=np.float32)) # initialize theta estimate to the lowerbounds

        # desired trajectory parameters
        self.phidMag = np.array([np.pi/8,np.pi/4],dtype=np.float32) # amplitude of oscillations in rad
        self.fphid = 0.2 # frequency in Hz
        self.aphid = np.pi/2 # phase shift in rad
        self.bphid = np.array([np.pi/2,np.pi/4],dtype=np.float32) # bias in rad

        # initialize state
        self.phi,_,_ = self.getDesiredState(0.0) # set the initial angle to the initial desired angle
        self.phiD = np.zeros(2,dtype=np.float32) # initial angular velocity
        self.phiDD = np.zeros(2,dtype=np.float32) # initial angular acceleration

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
        theta = np.array([(m[0]+m[1])*l[0]**2+m[1]*l[1]**2,
                          m[1]*l[0]*l[1],
                          m[1]*l[1]**2,
                          (m[0]+m[1])*l[0],
                          m[1]*l[1]],dtype=np.float32)
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
        phid = np.array([self.phidMag[0]*sin(2*np.pi*self.fphid*t-self.aphid)-self.bphid[0],
                         self.phidMag[1]*sin(2*np.pi*self.fphid*t-self.aphid)+self.bphid[1]],dtype=np.float32)

        #desired angular velocity
        phiDd = np.array([2*np.pi*self.fphid*self.phidMag[0]*cos(2*np.pi*self.fphid*t-self.aphid),
                          2*np.pi*self.fphid*self.phidMag[1]*cos(2*np.pi*self.fphid*t-self.aphid)],dtype=np.float32)

        #desired angular acceleration
        phiDDd = np.array([-((2*np.pi*self.fphid)**2)*self.phidMag[0]*sin(2*np.pi*self.fphid*t-self.aphid),
                           -((2*np.pi*self.fphid)**2)*self.phidMag[1]*sin(2*np.pi*self.fphid*t-self.aphid)],dtype=np.float32)
        
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
        M = np.array([[m1*l1**2+m2*(l1**2+2*l1*l2*c2+l2**2),m2*(l1*l2*c2+l2**2)],
                      [m2*(l1*l2*c2+l2**2),m2*l2**2]],dtype=np.float32)
        return M

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
        C = np.array([-2*m2*l1*l2*s2*phi1D*phi2D-m2*l1*l2*s2*phi2D**2,
                      m2*l1*l2*s2*phi1D**2],dtype=np.float32)
        return C

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
        G = np.array([(m1+m2)*self.g*l1*c1+m2*self.g*l2*c12,
                      m2*self.g*l2*c12],dtype=np.float32)
        return G

    # returns the inertia matrix regressor
    def getYM(self,vphi,phi):
        """
        Determines the inertia matrix regressor \n
        Inputs:
        -------
        \t vphi: phiDDd+alpha*eD \n
        \t phi:  angles \n
        
        Returns:
        -------
        \t YM: inertia matrix regressor
        """
        vphi1 = vphi[0]
        vphi2 = vphi[1]
        c2 = cos(phi[1])
        YM = np.array([[vphi1,2*c2*vphi1+c2*vphi2,vphi2,0.0,0.0],
                       [0.0,c2*vphi1,vphi1+vphi2,0.0,0.0]],dtype=np.float32)
        return YM

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
        YC = np.array([[0.0,-2*s2*phi1D*phi2D-s2*phi2D**2,0.0,0.0,0.0],
                       [0.0,s2*phi1D**2,0.0,0.0,0.0]],dtype=np.float32)
        return YC

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
        YG = np.array([[0.0,0.0,0.0,self.g*c1,self.g*c12],
                     [0.0,0.0,0.0,0.0,self.g*c12]],dtype=np.float32)
        return YG

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
        YMD = np.array([[0.0,-2*s2*phi2D*r1-s2*phi2D*r2,0.0,0.0,0.0],
                       [0.0,-s2*phi2D*r1,0.0,0.0,0.0]],dtype=np.float32)
        return YMD

    #returns the state
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
        \t thetaH: parameter estimate
        """
        return self.phi,self.phiD,self.phiDD,self.thetaH

    #returns the error state
    def getErrorState(self,t):
        """
        Returns the errors \n
        Inputs:
        -------
        \t t:  time \n
        
        Returns:
        -------
        \t e:          tracking error \n
        \t eD:         tracking error derivative \n
        \t r:          filtered tracking error \n
        \t thetaTilde: parameter estimate error
        """
        # get the desired state
        phid,phiDd,_ = self.getDesiredState(t)

        # get the tracking error
        e = phid - self.phi
        eD = phiDd - self.phiD
        r = eD + self.alpha@e

        # calculate the parameter error
        thetaTilde = self.theta-self.thetaH
        return e,eD,r,thetaTilde

    # returns the input and update law
    def getTauThetaHD(self,t):
        """
        Calculates the input and adaptive update law \n
        Inputs:
        -------
        \t t:  time \n
        
        Returns:
        -------
        \t tau:     control input \n
        \t thetaHD: parameter estimate adaptive update law \n
        \t tauff:   input from the feedforward portion of control \n
        \t taufb:   input from the feedback portion of control
        """
        # get the desired state
        _,_,phiDDd = self.getDesiredState(t)

        # get the error
        e,eD,r,_ = self.getErrorState(t)

        # get the regressors
        vphi = phiDDd + self.alpha@eD
        YM = self.getYM(vphi,self.phi)
        YC = self.getYC(self.phi,self.phiD)
        YG = self.getYG(self.phi)
        YMD = self.getYMD(self.phi,self.phiD,r)
        Y = YM+YC+YG+YMD

        #calculate the controller and update law
        tauff = Y@self.thetaH
        taufb = e+self.beta@r
        tau = tauff + taufb

        thetaHD = self.Gamma@Y.T@r
        return tau,thetaHD,tauff,taufb

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
        # get the dynamics
        M = self.getM(self.m,self.l,self.phi)
        C = self.getC(self.m,self.l,self.phi,self.phiD)
        G = self.getG(self.m,self.l,self.phi)

        # get the input and update law
        tau,thetaHD,_,_ = self.getTauThetaHD(t)

        # calculate the dynamics using the input
        self.phiDD = np.linalg.inv(M)@(-C-G+tau)

        # update the internal state
        # X(ii+1) = X(ii) + dt*f(X)
        self.phi += dt*self.phiD
        self.phiD += dt*self.phiDD
        self.thetaH += dt*thetaHD
        