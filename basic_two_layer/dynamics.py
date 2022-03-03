import numpy as np
from math import sin
from math import cos
from integral_concurrent_learning import ConcurrentLearning
from numpy.random import rand
from numpy.random import randn

np.random.seed(0)

# class for the two link dynamics
class Dynamics():
    # constructor to initialize a Dynamics object
    def __init__(self,alphae=0.1,alphadel=0.1,gammaw=0.01,gammav=0.01,lambdaCL=0.1,YYminDiff=0.1,kCL=0.9,uN=0.1,xN=0.01,xDN=0.05,l=100,deltaT=1.0,useCL=True,useBias=True):
        """
        Initialize the dynamics \n
        Inputs:
        -------
        \t betae:  error gain \n
        \t betaeps:  error robust gain \n
        \t gamma: parameter update gain \n
        \t kCL: CL parameter update gain \n
        \t uN: CL input noise is a disturbance \n
        \t xN: CL position measurement noise \n
        \t xDN: CL velocity measurement noise \n
        
        Returns:
        -------
        """
        # gains
        self.l = l
        self.L = self.l+1
        if not useBias:
            self.L -= 1
        self.alphae = alphae
        self.alphadel = alphadel
        self.Gammaw = gammaw*np.identity(self.L,dtype=np.float64)
        self.Gammav = gammav*np.identity(2,dtype=np.float64)
        self.kCL = kCL
        self.useCL = useCL

        # noise
        self.uNM=uN
        self.xNM=xN
        self.xDNM=xDN

        # parameters
        self.a = 1.0
        self.b = 0.5
        self.c = 0.25

        # unknown parameters
        self.WH = 0.1*randn(self.L) # initialize theta estimate to the lowerbounds
        self.VH = 0.1*randn(2,self.l) # initialize theta estimate to the lowerbounds
        
        # concurrent learning
        self.concurrentLearning = ConcurrentLearning(lambdaCL=lambdaCL,YYminDiff=YYminDiff,deltaT=deltaT,L=self.L)
        self.u = 0.0

        # desired trajectory parameters
        self.xdMag = 5.0 # amplitude of oscillations
        self.fxd = 0.1 # frequency in Hz
        self.axd = 0.0 # phase shift in rad
        self.bxd = 0.0 # bias in rad

        # initialize state
        self.x,_ = self.getDesiredState(0.0) # set the initial angle to the initial desired angle
        self.xD = 0.0 # initial angular velocity
        self.xN = self.xNM*randn()
        self.xDN = self.xDNM*randn()
        self.uN = self.uNM*randn()

    def getDesiredState(self,t):
        """
        Determines the desired state of the system \n
        Inputs:
        -------
        \t t: time \n
        
        Returns:
        -------
        \t xd:   desired position \n
        \t xDd:  desired velocity \n
        """
        # desired angles
        xd = self.xdMag*sin(2*np.pi*self.fxd*t-self.axd)-self.bxd

        #desired angular velocity
        xDd = 2*np.pi*self.fxd*self.xdMag*cos(2*np.pi*self.fxd*t-self.axd)
        
        return xd,xDd

    def getsigma(self,x,V):
        """
        Determines the basis for the system \n
        Inputs:
        -------
        \t x: position \n
        \t V: inner weights \n
        
        Returns:
        -------
        \t sigma: basis \n
        """
        sigma = np.ones(self.L,dtype=np.float64)
        xi = np.ones(2,dtype=np.float64)
        xi[0] = x
        Phi = V.T@xi
        for ii in range(self.l):
            sigma[ii] = sin(Phi[ii])
        return sigma
    
    def getsigmaPrime(self,x,V):
        """
        Determines the basis for the system \n
        Inputs:
        -------
        \t x: position \n
        \t V: inner weights \n
        
        Returns:
        -------
        \t sigma: basis \n
        """
        sigmaPrimeInner = np.zeros(self.l,dtype=np.float64)
        sigmaPrime = np.zeros((self.L,self.l),dtype=np.float64)
        xi = np.ones(2,dtype=np.float64)
        xi[0] = x
        Phi = V.T@xi
        for ii in range(self.l):
            sigmaPrimeInner[ii] = cos(Phi[ii])
        sigmaPrime[:self.l,:] = np.diag(sigmaPrimeInner)
        return sigmaPrime

    # returns the state
    def getState(self,t):
        """
        Returns the state of the system and parameter estimates \n
        Inputs:
        -------
        \t t: time \n
        
        Returns:
        -------
        \t xm:   measured position \n
        \t xDm:  measured angular velocity \n
        \t thetaH: parameter estimate \n
        """
        
        xm = self.x+self.xN
        xDm = self.xD+self.xDN
        
        return xm,xDm,self.WH,self.VH

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
        \t xm:     measured position \n
        \t xDm:    measured angular velocity \n
        \t thetaH: parameter estimate \n
        """
        
        # get the desired state
        xd,xDd = self.getDesiredState(t)

        xm,xDm,WH,VH = self.getState(t)

        # get the tracking error
        em = xm - xd
        eDm = xDm - xDd

        return em,eDm,xm,xDm,WH,VH

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
        YYsumMinEig,TCL,YYsum,YtauSum = self.concurrentLearning.getState()
        return YYsumMinEig,TCL,YYsum,YtauSum

    # returns the input and update law
    def getinputWHD(self,t):
        """
        Calculates the input and adaptive update law \n
        Inputs:
        -------
        \t t:  time \n
        
        Returns:
        -------
        \t u:     control input \n
        \t WHD: parameter estimate adaptive update law \n
        \t uff:   input from the feedforward portion of control \n
        \t ufb:   input from the feedback portion of control \n
        """
        # get the desired state
        xd,xDd = self.getDesiredState(t)

        # get the error
        em,eDm,xm,xDm,WH,VH = self.getErrorState(t)

        #get sigma
        sigmam = self.getsigma(xm,VH)
        # sigmaPm = self.getsigmaPrime(xm,VH)
        # xim = np.ones(2,dtype=np.float64)
        # xim[0] = xm

        #calculate the controller and update law
        uff = xDd-WH@sigmam
        ufb = -self.alphae*em-self.alphadel*np.sign(em)
        u = uff + ufb

        #update the CL stack and the update law
        # YYsumMinEig,_,YYsum,YuSum = self.concurrentLearning.getState()
        # WHD = em*self.Gammaw@sigmam
        # VHD = em*self.Gammav@xim@WH.T@sigmaPm

        # if self.useCL:
        #     WHD += self.kCL*self.Gamma@(YuSum - YYsum@WH)
            # if YYsumMinEig > 0.000001:
            #     WCL = np.linalg.inv(YYsum)@YuSum
            #     print("WCL \n"+str(WCL))
            #     print("WCL agree \n"+str(xDm - u - WCL@sigmam))
        return u,uff,ufb

    def getfunc(self,x):
        return self.a*sin(self.b*x+self.c)*cos(self.c*x)

    def getfuncComp(self,x,WH,VH):
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

        #get sigma
        sigmam = self.getsigma(x,VH)

        #calculate the actual
        f = self.getfunc(x)

        #calculate the approximate
        fH = WH@sigmam

        return f,fH

    def getf(self,t,X):
        """
        Dynamics callback \n
        Inputs:
        -------
        \t t:  time \n
        \t X:  stacked x,WH,VH \n
        
        Returns:
        -------
        \t XD: derivative approximate at time \n
        \t u: control input at time \n
        """

        x = X[0]
        WH = X[1:self.L+1]
        VH = np.reshape(X[self.L+1:],(2,self.l))

        # get the desired state
        xd,xDd = self.getDesiredState(t)

        # get the error
        xm = x + self.xN
        em = xm - xd

        #get sigma
        sigmam = self.getsigma(xm,VH)
        sigmaPm = self.getsigmaPrime(xm,VH)
        xim = np.ones(2,dtype=np.float64)
        xim[0] = xm

        #calculate the controller and update law
        uff = xDd-WH@sigmam
        ufb = -self.alphae*em-self.alphadel*np.sign(em)
        u = uff + ufb

        # calculate the dynamics using the input
        xD = self.getfunc(self.x) + u + self.uN

        #update the CL stack and the update law
        # _,_,YYsum,YuSum = self.concurrentLearning.getState()

        WHD = em*self.Gammaw@sigmam
        VHD = em*self.Gammav@np.outer(xim,WH)@sigmaPm

        # if self.useCL:
        #     WHD += self.kCL*self.Gamma@(YuSum - YYsum@WH)

        #calculate and return the derivative
        XD = np.zeros_like(X)
        XD[0] = xD
        XD[1:self.L+1] = WHD
        XD[self.L+1:] = np.reshape(VHD,(2*self.l))

        return XD,u

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

        k1,u1 = self.getf(t,X)
        k2,u2 = self.getf(t+0.5*dt,X+0.5*dt*k1)
        k3,u3 = self.getf(t+0.5*dt,X+0.5*dt*k2)
        k4,u4 = self.getf(t+dt,X+dt*k3)
        XD = (1.0/6.0)*(k1+2.0*k2+2.0*k3+k4)
        um = (1.0/6.0)*(u1+2.0*u2+2.0*u3+u4)

        return XD,um

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
        X = np.zeros(1+self.L+2*self.l,dtype=np.float64)
        X[0] = self.x
        X[1:self.L+1] = self.WH
        X[self.L+1:] = np.reshape(self.VH,(2*self.l))

        #get the derivative and input from rk
        XD,um = self.rk4(dt,t,X)
        self.xD = XD[0]
        WHD = XD[1:self.L+1]
        VHD = np.reshape(XD[self.L+1:],(2,self.l))

        #get the new state for learning
        # xm = self.x + self.xN
        # xDm = self.xD + self.xDN
        # sigmam = self.getsigma(xm)

        # print("model agree " + str(self.xD - um - self.WH@sigmam))

        # update the internal state
        # X(ii+1) = X(ii) + dt*f(X)
        self.x += dt*self.xD
        self.WH += dt*WHD
        self.VH += dt*VHD

        self.xN = self.xNM*randn()
        self.xDN = self.xDNM*randn()
        self.uN = self.uNM*randn()

        # update the concurrent learning
        # get the inertia regressor for CL
        # self.concurrentLearning.append(sigmam,xDm-um,t+dt)