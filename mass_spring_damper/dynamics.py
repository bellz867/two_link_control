import numpy as np
from math import sin
from math import cos
from concurrent_learning import ConcurrentLearning

# class for the two link dynamics
class Dynamics():
    # constructor to initialize a Dynamics object
    def __init__(self,alpha=0.25,beta=0.1,gamma=0.01*np.ones(3,dtype=np.float32),lambdaCL=0.1,YYminDiff=0.1,kCL=0.9):
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
        self.alpha = alpha
        self.beta = beta
        self.Gamma = np.diag(gamma)
        self.kCL = kCL

        # rigid body parameters
        self.m = 1.0 # moment of inertia in kg m^2
        self.c = 0.5 # damper coefficient in kg m^2/s
        self.k = 0.25 # spring in kg m^2/s^2

        # unknown parameters
        self.theta = self.getTheta(self.m,self.c,self.k) # initialize theta
        self.thetaH = self.getTheta(0.5*self.m,0.5*self.c,0.5*self.k) # initialize theta estimate to the lowerbounds
        
        # concurrent learning
        self.concurrentLearning = ConcurrentLearning(lambdaCL,YYminDiff)
        self.tau = 0.0

        # desired trajectory parameters
        self.phidMag = np.pi/2 # amplitude of oscillations in radians
        self.fphid = 0.2 # frequency in Hz
        self.aphid = 0.0 # phase shift in rad
        self.bphid = 0.0 # bias in rad

        # initialize state
        self.phi,_,_ = self.getDesiredState(0.0) # set the initial angle to the initial desired angle
        self.phiD = 0.0 # initial angular velocity
        self.phiDD = 0.0 # initial angular acceleration


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
        r = eD + self.alpha*e

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
        Y = np.array([phiDDd+self.alpha*eD,self.phiD,self.phi],dtype=np.float32)

        #calculate the controller and update law
        print("Y \n"+str(Y))
        print("thetaH \n"+str(self.thetaH))
        tauff = Y@self.thetaH
        taufb = e+self.beta*r
        tau = tauff + taufb
        #update the CL stack and the update law
        YYsumMinEig,_,YYsum,YtauSum = self.concurrentLearning.getState()
        thetaCL = np.zeros_like(self.theta,np.float32)
        if YYsumMinEig > 0.001:
            thetaCL = np.linalg.inv(YYsum)@YtauSum
            # print("thetaCL "+str(thetaCL))
        thetaHD = r*self.Gamma@Y.T + self.kCL*self.Gamma@(YtauSum - YYsum@self.thetaH)
        return tau,thetaHD,tauff,taufb,thetaCL

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
        tau,thetaHD,_,_,_ = self.getTauThetaHD(t)

        # calculate the dynamics using the input
        self.phiDD = (1.0/self.m)*(-self.c*self.phiD-self.k*self.phi+tau)

        # update the internal state
        # X(ii+1) = X(ii) + dt*f(X)
        self.phi += dt*self.phiD
        self.phiD += dt*self.phiDD
        self.thetaH += dt*thetaHD

        # update the concurrent learning
        # get the inertia regressor for CL
        YCL = np.array([self.phiDD,self.phiD,self.phi],dtype=np.float32)
        self.concurrentLearning.append(YCL,tau,t+dt)
        