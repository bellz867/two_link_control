import numpy as np
from math import sin
from math import cos
from integral_concurrent_learning import ConcurrentLearning

np.random.seed(0)

# class for the two link dynamics
class Dynamics():
    # constructor to initialize a Dynamics object
    def __init__(self,alpha=0.25*np.ones(2,dtype=np.float32),beta=0.1*np.ones(2,dtype=np.float32),gamma=0.01*np.ones(5,dtype=np.float32),lambdaCL=0.1,YYminDiff=0.1,kCL=0.9):
        """
        Initialize the dynamics \n
        Inputs:
        -------
        \t alpha: error gain \n
        \t beta:  filtered error gain \n
        \t gamma: parameter update gain \n
        \t kCL: CL parameter update gain \n
        
        Returns:
        -------
        """
        # gains
        self.alpha = np.diag(alpha)
        self.beta = np.diag(beta)
        self.Gamma = np.diag(gamma)
        self.kCL = kCL

        # rigid body parameters
        self.m = np.array([2.0,2.0],dtype=np.float32) # mass in kg
        self.l = np.array([0.5,0.5],dtype=np.float32) # length in m
        self.mBnds = np.array([1.0,3.0],dtype=np.float32) # mass bounds in kg
        self.lBnds = np.array([0.25,0.75],dtype=np.float32) # length bounds in m
        self.g = 9.8 # gravity in m/s^2

        # unknown parameters
        self.theta = self.getTheta(self.m,self.l) # initialize theta
        self.thetaH = self.getTheta(self.mBnds[0]*np.ones(2,dtype=np.float32),self.lBnds[0]*np.ones(2,dtype=np.float32)) # initialize theta estimate to the lowerbounds
        
        # concurrent learning
        self.concurrentLearning = ConcurrentLearning(lambdaCL,YYminDiff)
        self.tau = np.zeros(2,np.float32)

        # desired trajectory parameters
        self.phidMag = np.array([np.pi/8,np.pi/4],dtype=np.float32) # amplitude of oscillations in rad
        self.fphid = 0.2 # frequency in Hz
        self.aphid = np.pi/2 # phase shift in rad
        self.bphid = np.array([np.pi/2,np.pi/4],dtype=np.float32) # bias in rad

        # initialize state
        self.maxDiff = 10**(-4)
        self.phi,_,_ = self.getDesiredState(0.0) # set the initial angle to the initial desired angle
        self.phiD = np.zeros(2,dtype=np.float32) # initial angular velocity
        self.phiDD = np.zeros(2,dtype=np.float32) # initial angular acceleration

        #butcher table for ode45 from https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
        #implement from https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge.E2.80.93Kutta_methods
        #c is the time weights, b is the out weights, balt is the alternative out weights, and a is the table weights
        self.BTc = np.array([0,1/5,3/10,4/5,8/9,1,1])
        self.BTb = np.array([35/384,0,500/1113,125/192,-2187/6784,11/84,0])
        self.BTbalt = np.array([5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40])
        self.BTa = np.zeros((7,6),dtype=np.float32)
        self.BTa[1,0] = 1/5
        self.BTa[2,0:2] = [3/40,9/40]
        self.BTa[3,0:3] = [44/45,-56/15,32/9]
        self.BTa[4,0:4] = [19372/6561,-25360/2187,64448/6561,-212/729]
        self.BTa[5,0:5] = [9017/3168,-355/33,46732/5247,49/176,-5103/18656]
        self.BTa[6,0:6] = [35/384,0,500/1113,125/192,-2187/6784,11/84]


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
        \t vphi: phiDDd+alpha*eD or phiDD \n
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
        \t thetaH: parameter estimate \n
        \t thetaH: parameter
        """
        return self.phi,self.phiD,self.phiDD,self.thetaH,self.theta

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

    def getCLstate(self):
        """
        Returns select parameters CL \n
        Inputs:
        -------
        
        Returns:
        -------
        \t YYsumMinEig: current minimum eigenvalue of sum of the Y^T*Y terms \n
        \t TCL: time of the minimum eigenvalue found \n
        \t YYsum: Y^T*Y sum \n
        \t YtauSum: Y^T*tau sum \n

        """
        return self.concurrentLearning.getState()

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
        \t taufb:   input from the feedback portion of control \n
        \t thetaCL: approximate of theta from CL \n
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
        Y = YM+YC+YG+0.5*YMD

        #calculate the controller and update law
        tauff = Y@self.thetaH
        taufb = e+self.beta@r
        tau = tauff + taufb
        
        #update the CL stack and the update law
        YYsumMinEig,_,YYsum,YtauSum = self.concurrentLearning.getState()
        thetaCL = np.zeros_like(self.theta,np.float32)
        if YYsumMinEig > 0.001:
            thetaCL = np.linalg.inv(YYsum)@YtauSum
        thetaHD = self.Gamma@Y.T@r + self.kCL*self.Gamma@(YtauSum - YYsum@self.thetaH)
        return tau,thetaHD,tauff,taufb,thetaCL

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

        # get the desired state
        phid,phiDd,phiDDd = self.getDesiredState(t)

        # get the errors
        e = phid - phi
        eD = phiDd - phiD
        r = eD + self.alpha@e

        # get the regressors
        vphi = phiDDd + self.alpha@eD
        YM = self.getYM(vphi,phi)
        YC = self.getYC(phi,phiD)
        YG = self.getYG(phi)
        YMD = self.getYMD(phi,phiD,r)
        Y = YM+YC+YG+0.5*YMD

        #calculate the controller and update law
        tauff = Y@thetaH
        taufb = e+self.beta@r
        tau = tauff + taufb

        # get the dynamics
        M = self.getM(self.m,self.l,phi)
        C = self.getC(self.m,self.l,phi,phiD)
        G = self.getG(self.m,self.l,phi)
        phiDD = np.linalg.inv(M)@(-C-G+tau)

        # get the update law
        _,_,YYsum,YtauSum = self.concurrentLearning.getState()
        thetaHD = self.Gamma@Y.T@r + self.kCL*self.Gamma@(YtauSum - YYsum@thetaH)

        #calculate and return the derivative
        XD = np.zeros_like(X)
        XD[0:2] = phiD
        XD[2:4] = phiDD
        XD[4:9] = thetaHD

        return XD,tau

    #classic rk1 method aka Euler
    def rk1(self,dt,t,X):
        """
        Classic rk1 method aka Euler \n
        Inputs:
        -------
        \t dt:  total time step for interval \n
        \t t:  time \n
        \t X:  stacked phi,phiD,thetaH \n
        
        Returns:
        -------
        \t XD: derivative approximate over total interval \n
        \t tau: control input approximate over total interval \n
        \t Xh: integrated value \n
        """
        XD,tau = self.getf(t,X)
        Xh = X + dt*XD

        M = self.getM(self.m,self.l,Xh[0:2])
        C = self.getC(self.m,self.l,Xh[0:2],Xh[2:4])
        G = self.getG(self.m,self.l,Xh[0:2])
        tau = M@XD[2:4]+C+G

        return XD,tau,Xh

    #classic rk4 method
    def rk4(self,dt,t,X):
        """
        Classic rk4 method \n
        Inputs:
        -------
        \t dt:  total time step for interval \n
        \t t:  time \n
        \t X:  stacked phi,phiD,thetaH \n
        
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
        # tau = (1.0/6.0)*(tau1+2.0*tau2+2.0*tau3+tau4)
        Xh = X+dt*XD

        M = self.getM(self.m,self.l,Xh[0:2])
        C = self.getC(self.m,self.l,Xh[0:2],Xh[2:4])
        G = self.getG(self.m,self.l,Xh[0:2])
        tau = M@XD[2:4]+C+G

        return XD,tau,Xh

    #adaptive step using classic Dormand Prince method aka rk45 or ode45 method
    def rk45(self,dt,t,X):
        """
        Adaptive step using classic Dormand Prince method aka ode45 method \n
        Inputs:
        -------
        \t dt:  total time step for interval \n
        \t t:  time \n
        \t X:  stacked phi,phiD,thetaH \n
        
        Returns:
        -------
        \t XD: derivative approximate over total interval \n
        \t tau: control input approximate over total interval \n
        \t Xh: integrated value \n
        """

        #initially time step is equal to full dt
        steps = 1
        XDdiff = 100.0
        XD = np.zeros(9,dtype=np.float32)
        Xh = X.copy()
        while XDdiff >= self.maxDiff:
            Xh = X.copy()
            th = t
            h = dt/steps
            for ii in range(steps):
                #calculate the ks and taus
                ks = np.zeros((7,9),np.float32)
                ks[0,:],_ = self.getf(th,Xh)
                ks[1,:],_ = self.getf(th+self.BTc[1]*h,Xh+h*(self.BTa[1,0]*ks[0,:]))
                ks[2,:],_ = self.getf(th+self.BTc[2]*h,Xh+h*(self.BTa[2,0]*ks[0,:]+self.BTa[2,1]*ks[1,:]))
                ks[3,:],_ = self.getf(th+self.BTc[3]*h,Xh+h*(self.BTa[3,0]*ks[0,:]+self.BTa[3,1]*ks[1,:]+self.BTa[3,2]*ks[2,:]))
                ks[4,:],_ = self.getf(th+self.BTc[4]*h,Xh+h*(self.BTa[4,0]*ks[0,:]+self.BTa[4,1]*ks[1,:]+self.BTa[4,2]*ks[2,:]+self.BTa[4,3]*ks[3,:]))
                ks[5,:],_ = self.getf(th+self.BTc[5]*h,Xh+h*(self.BTa[5,0]*ks[0,:]+self.BTa[5,1]*ks[1,:]+self.BTa[5,2]*ks[2,:]+self.BTa[5,3]*ks[3,:]+self.BTa[5,4]*ks[4,:]))
                ks[6,:],_ = self.getf(th+self.BTc[6]*h,Xh+h*(self.BTa[6,0]*ks[0,:]+self.BTa[6,1]*ks[1,:]+self.BTa[6,2]*ks[2,:]+self.BTa[6,3]*ks[3,:]+self.BTa[6,4]*ks[4,:]+self.BTa[6,5]*ks[5,:]))
                
                #calculate the complete derivate, alternative derivative, and input
                XDh = np.zeros(9,dtype=np.float32)
                XDalth = np.zeros(9,dtype=np.float32)
                for ii in range(7):
                    XDh += self.BTb[ii]*ks[ii,:]
                    XDalth += self.BTbalt[ii]*ks[ii,:]
                            
                th += h
                Xh += h*XDh

                # update the difference 
                XDdiff = np.linalg.norm(XDh-XDalth)
                if XDdiff >= self.maxDiff:
                    print("h ",str(h))
                    print("th ",str(th))
                    print("XD diff ",str(XDdiff))
                    phiDdiff = np.linalg.norm(XDh[0:2]-Xh[2:4])
                    print("phiD diff ",str(phiDdiff))
                    steps += 1
                    break
            if XDdiff < self.maxDiff:
                XD = (1.0/dt)*(Xh-X)

        M = self.getM(self.m,self.l,Xh[0:2])
        C = self.getC(self.m,self.l,Xh[0:2],Xh[2:4])
        G = self.getG(self.m,self.l,Xh[0:2])
        tau = M@XD[2:4]+C+G

        return XD,tau,Xh

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
        X = np.zeros(9,dtype=np.float32)
        X[0:2] = self.phi
        X[2:4] = self.phiD
        X[4:9] = self.thetaH

        #get the derivative and input from rk
        XD,tau,Xh = self.rk4(dt,t,X)

        self.phi = Xh[0:2]
        self.phiD = Xh[2:4]
        self.thetaH = Xh[4:9]
        self.phiDD = XD[2:4]

        # update the concurrent learning
        # get the inertia regressor for CL
        YMCL = self.getYM(self.phiDD,self.phi)
        YC = self.getYC(self.phi,self.phiD)
        YG = self.getYG(self.phi)
        YCL = YMCL+YC+YG
        self.concurrentLearning.append(YCL,tau,t+dt)
        