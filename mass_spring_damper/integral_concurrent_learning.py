import numpy as np
from numpy.linalg import eigvals

from collections import namedtuple

TrainData = namedtuple('TrainData',('Y','tau','t'))

# class for the two link dynamics
class ConcurrentLearning():
    # constructor to initialize a Dynamics object
    def __init__(self,lambdaCL=0.1,YYminDiff=0.1,deltaT=1.0):
        """
        Initialize the learning \n
        Inputs:
        -------
        \t lambdaCL: minimum eigenvalue for the sum \n
        \t YYminDiff: minimum difference between data to save it to the buffer \n
        
        Returns:
        -------
        """
        self.deltaT = deltaT # integration window size
        self.intBuff = [] # buffer for the terms to integrate over
        self.Ybuff = [] # buffer for the Y terms to check min eig
        self.YYsum = np.zeros((3,3),dtype=np.float32) # sum of the Y^T*Y terms
        self.YtauSum = np.zeros(3,dtype=np.float32) # sum of the Y^T*tau terms
        self.YYsumMinEig = 0.0 # current minimum eigenvalue for the sum
        self.lambdaCL = lambdaCL  # desired minimum eigenvalue for the sum
        self.TCL = 0.0 # time the minimum eigenvalue is satisfied
        self.lambdaCLMet = False # indicates if the eigenvalue condition is satisfied
        self.YYminDiff = YYminDiff # minimum difference between new and existing data to add it

    def append(self,Y,tau,t):
        """
        Adds the new data to the buffer if it is different enough and the minimum eigenvalue is not satisfied
        Inputs:
        -------
        \t Y: regressor \n
        \t tau: torque \n
        \t t: time \n
        
        Returns:
        -------
        """

        # dont add the data if the minimum eigenvalue is good or the new data has a good minimum singular value
        if not self.lambdaCLMet:
            # add the point to the integral
            self.intBuff.append(TrainData(Y,tau,t))
            Yint = np.zeros(3)
            tauInt = 0.0

            # if enough data to integrate then integrate
            if len(self.intBuff) > 2:

                # check if more data than need for the integral then remove oldest if there is
                deltatCurr = self.intBuff[-1].t-self.intBuff[1].t
                if deltatCurr > self.deltaT:
                    self.intBuff.pop(0)
                
                # calculate the integral using trapezoidal rule
                for ii in range(len(self.intBuff)-1):
                    dti = self.intBuff[ii+1].t-self.intBuff[ii].t
                    YAvgi = 0.5*(self.intBuff[ii+1].Y+self.intBuff[ii].Y)
                    tauAvgi =  0.5*(self.intBuff[ii+1].tau+self.intBuff[ii].tau)
                    Yint += dti*YAvgi
                    tauInt += dti*tauAvgi
            else:
                return


            YSV = np.linalg.norm(Yint)
            if (YSV > self.YYminDiff) and (np.linalg.norm(tauInt) > self.YYminDiff):
                # check to make sure the data is different enough from the other data
                # find the minimum difference
                minDiff = 100.0
                for Yi in self.Ybuff:
                    YYdiffi = np.linalg.norm(Yi-Yint)
                    if YYdiffi < minDiff:
                        minDiff = YYdiffi
                
                # if the minimum difference is large enough add the data
                if minDiff > self.YYminDiff:
                    self.Ybuff.append(Yint)
                    # print("Y \n"+str(Y))
                    YY = np.outer(Yint,Yint)
                    # print("YY \n"+str(YY))
                    Ytau = Yint.T*tauInt
                    # print("Ytau \n"+str(Ytau))
                    self.YYsum += YY
                    self.YtauSum += Ytau
                    self.YYsumMinEig = np.min(eigvals(self.YYsum))

                    # check if the new data makes the eigenvalue large enough
                    if self.YYsumMinEig > self.lambdaCL:
                        self.TCL = t
                        self.lambdaCLMet = True

    def getState(self):
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
        return self.YYsumMinEig,self.TCL,self.YYsum,self.YtauSum