import numpy as np
from math import sin
from math import cos
from integral_concurrent_learning import ConcurrentLearning

np.random.seed(0)

# class for the two link dynamics
class Dynamics():
    # constructor to initialize a Dynamics object
    def __init__(self,alpha=0.25,beta=0.1,gamma=0.01*np.ones(3,dtype=np.float32),lambdaCL=0.1,YYminDiff=0.1,kCL=0.9,addNoise=False,tauN=0.1,phiN=0.01,phiDN=0.05,phiDDN=0.1):
        """
        Initialize the dynamics \n
        Inputs:
        -------
        \t alpha: error gain \n
        \t beta:  filtered error gain \n
        \t gamma: parameter update gain \n
        \t kCL: CL parameter update gain \n
        \t addNoise: indicator to add noise to the signals \n
        \t tauN: CL input noise is a disturbance \n
        \t phiN: CL angle measurement noise \n
        \t phiDN: CL velocity measurement noise \n
        \t phiDDN: CL acceleration measurement noise \n
        
        Returns:
        -------
        """
        # gains
        self.alpha = alpha
        self.beta = beta
        self.Gamma = np.diag(gamma)
        self.kCL = kCL

        # noise
        self.addNoise=addNoise
        self.tauN=tauN
        self.phiN=phiN
        self.phiDN=phiDN
        self.phiDDN=phiDDN

        # rigid body parameters
        self.m = 1.0 # moment of inertia in kg m^2
        self.c = 0.5 # damper coefficient in kg m^2/s
        self.k = 0.25 # spring in kg m^2/s^2

        # unknown parameters
        self.theta = self.getTheta(self.m,self.c,self.k) # initialize theta
        self.thetaH = self.getTheta(0.5*self.m,0.5*self.c,0.5*self.k) # initialize theta estimate to the lowerbounds
        self.thetaHm = self.getTheta(0.5*self.m,0.5*self.c,0.5*self.k) # initialize theta estimate to the lowerbounds
        
        # concurrent learning
        self.concurrentLearning = ConcurrentLearning(lambdaCL,YYminDiff)
        self.concurrentLearningm = ConcurrentLearning(lambdaCL,YYminDiff)
        self.tau = 0.0

        # desired trajectory parameters
        self.phidMag = np.pi/2 # amplitude of oscillations in radians
        self.fphid = 0.2 # frequency in Hz
        self.aphid = 0.0 # phase shift in rad
        self.bphid = 0.0 # bias in rad

        # initialize state
        self.phi,_,_ = self.getDesiredState(0.0) # set the initial angle to the initial desired angle
        self.phim = self.phi
        self.phiD = 0.0 # initial angular velocity
        self.phiDm = 0.0
        self.phiDD = 0.0 # initial angular acceleration
        self.phiDDm = 0.0



    def getTheta(self,m,c,k):
        """
        Inputs:
        -------
        \t m: mass \n
        \t c: damper \n
        \t c: spring \n
        
        Returns:
        -------
        \t theta: parameters
        """
        theta = np.array([m,c,k],dtype=np.float32)
        return theta

    def getDesiredState(self,t):
        """
        Determines the desired state of the system \n
        Inputs:
        -------
        \t t: time \n
        
        Returns:
        -------
        \t phid:   desired angle \n
        \t phiDd:  desired angular velocity \n
        \t phiDDd: desired angular acceleration
        """
        # desired angles
        phid = self.phidMag*sin(2*np.pi*self.fphid*t-self.aphid)-self.bphid

        #desired angular velocity
        phiDd = 2*np.pi*self.fphid*self.phidMag*cos(2*np.pi*self.fphid*t-self.aphid)

        #desired angular acceleration
        phiDDd = -((2*np.pi*self.fphid)**2)*self.phidMag*sin(2*np.pi*self.fphid*t-self.aphid)
        
        return phid,phiDd,phiDDd

    # returns the state
    def getState(self,t):
        """
        Returns the state of the system and parameter estimates \n
        Inputs:
        -------
        \t t: time \n
        
        Returns:
        -------
        \t phi:    angles \n
        \t phim:   measured angles \n
        \t phiD:   angular velocity \n
        \t phiDm:  measured angular velocity \n
        \t phiDD:  angular acceleration \n
        \t phiDDm:  measured angular acceleration \n
        \t thetaH: parameter estimate \n
        \t thetaHm: measured parameter estimate \n
        \t theta: parameter
        """

        phim = self.phim+self.phiN*np.random.randn()
        phiDm = self.phiDm+self.phiDN*np.random.randn()
        phiDDm = self.phiDDm+self.phiDDN*np.random.randn()
        
        return self.phi,phim,self.phiD,phiDm,self.phiDD,phiDDm,self.thetaH,self.thetaHm,self.theta

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
        \t em:          measured tracking error \n
        \t eD:         tracking error derivative \n
        \t eDm:          measured tracking error derivative \n
        \t r:          filtered tracking error \n
        \t rm:          measured filtered tracking error \n
        \t thetaTilde: parameter estimate error \n
        \t thetaTildem: measured parameter estimate error \n
        """
        
        # get the desired state
        phid,phiDd,_ = self.getDesiredState(t)

        phi,phim,phiD,phiDm,_,_,thetaH,thetaHm,theta = self.getState(t)

        # get the tracking error
        e = phid - phi
        em = phid - phim
        eD = phiDd - phiD
        eDm = phiDd - phiDm
        r = eD + self.alpha*e
        rm = eDm + self.alpha*em

        # calculate the parameter error
        thetaTilde = theta-thetaH
        thetaTildem = theta-thetaHm
        return e,em,eD,eDm,r,rm,thetaTilde,thetaTildem,phim,phiDm

    def getCLstate(self):
        """
        Returns select parameters CL \n
        Inputs:
        -------
        
        Returns:
        -------
        \t YYsumMinEig: current minimum eigenvalue of sum of the Y^T*Y terms \n
        \t YYsumMinEigm: measrured current minimum eigenvalue of sum of the Y^T*Y terms \n
        \t TCL: time of the minimum eigenvalue found \n
        \t TCLm: measured time of the minimum eigenvalue found \n
        \t YYsum: Y^T*Y sum \n
        \t YYsumm: Ym^T*Ym sum \n
        \t YtauSum: Y^T*tau sum \n
        \t YtauSumm: Ym^T*taum sum \n

        """
        YYsumMinEig,TCL,YYsum,YtauSum = self.concurrentLearning.getState()
        YYsumMinEigm,TCLm,YYsumm,YtauSumm = self.concurrentLearningm.getState()
        return YYsumMinEig,YYsumMinEigm,TCL,TCLm,YYsum,YYsumm,YtauSum,YtauSumm

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
        \t taum:     measured control input \n
        \t thetaHD: parameter estimate adaptive update law \n
        \t thetaHDm: measrued parameter estimate adaptive update law \n
        \t tauff:   input from the feedforward portion of control \n
        \t tauffm:  measured input from the feedforward portion of control \n
        \t taufb:   input from the feedback portion of control \n
        \t taufbm:  measured input from the feedback portion of control \n
        \t thetaCL: approximate of theta from CL \n
        \t thetaCLm: measured approximate of theta from CL \n
        """
        # get the desired state
        _,_,phiDDd = self.getDesiredState(t)

        # get the state
        phi,_,phiD,_,_,_,thetaH,thetaHm,_ = self.getState(t)

        # get the error
        e,em,eD,eDm,r,rm,_,_,phim,phiDm = self.getErrorState(t)

        # get the regressors
        Y = np.array([phiDDd+self.alpha*eD,phiD,phi],dtype=np.float32)
        Ym = np.array([phiDDd+self.alpha*eDm,phiDm,phim],dtype=np.float32)

        #calculate the controller and update law
        tauff = Y@self.thetaH
        taufb = e+self.beta*r
        tau = tauff + taufb

        tauffm = Ym@self.thetaHm
        taufbm = em+self.beta*rm
        taum = tauffm + taufbm

        #update the CL stack and the update law
        YYsumMinEig,_,YYsum,YtauSum = self.concurrentLearning.getState()
        YYsumMinEigm,_,YYsumm,YtauSumm = self.concurrentLearningm.getState()
        thetaCL = np.zeros_like(self.theta,np.float32)
        thetaCLm = np.zeros_like(self.theta,np.float32)
        if YYsumMinEig > 0.001:
            thetaCL = np.linalg.inv(YYsum)@YtauSum
            # print("thetaCL "+str(thetaCL))

        if YYsumMinEigm > 0.001:
            thetaCLm = np.linalg.inv(YYsumm)@YtauSumm
            # print("thetaCL "+str(thetaCL))

        thetaHD = r*self.Gamma@Y.T + self.kCL*self.Gamma@(YtauSum - YYsum@thetaH)
        thetaHDm = rm*self.Gamma@Ym.T + self.kCL*self.Gamma@(YtauSumm - YYsumm@thetaHm)
        return tau,taum,thetaHD,thetaHDm,tauff,tauffm,taufb,taufbm,thetaCL,thetaCLm

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

        # get the input and update law
        tau,taum,thetaHD,thetaHDm,_,_,_,_,_,_ = self.getTauThetaHD(t)

        # calculate the dynamics using the input
        self.phiDD = (1.0/self.m)*(-self.c*self.phiD-self.k*self.phi+tau)
        self.phiDDm = (1.0/self.m)*(-self.c*self.phiDm-self.k*self.phim+taum+self.tauN*np.random.randn())

        # update the internal state
        # X(ii+1) = X(ii) + dt*f(X)
        self.phi += dt*self.phiD
        self.phim += dt*self.phiDm
        self.phiD += dt*self.phiDD
        self.phiDm += dt*self.phiDDm
        self.thetaH += dt*thetaHD
        self.thetaHm += dt*thetaHDm

        #get the new state fpr learning
        phi,phim,phiD,phiDm,phiDD,phiDDm,_,_,_ = self.getState(t)
        taum = self.m*phiDDm+self.c*phiDm+self.k*phim-self.tauN*np.random.randn()

        # update the concurrent learning
        # get the inertia regressor for CL
        YCL = np.array([phiDD,phiD,phi],dtype=np.float32)
        YCLm = np.array([phiDDm,phiDm,phim],dtype=np.float32)
        self.concurrentLearning.append(YCL,tau,t+dt)
        self.concurrentLearningm.append(YCLm,taum,t+dt)