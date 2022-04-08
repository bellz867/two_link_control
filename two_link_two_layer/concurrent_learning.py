import numpy as np
from numpy.linalg import eigvals

# class for the two link dynamics
class ConcurrentLearning():
    # constructor to initialize a Dynamics object
    def __init__(self,lambdaCL=0.1,YYminDiff=0.1,deltaT=1.0,L=100):
        """
        Initialize the learning \n
        Inputs:
        -------
        \t lambdaCL: minimum eigenvalue for the sum \n
        \t YYminDiff: minimum difference between data to save it to the buffer \n
        
        Returns:
        -------
        """
        self.L = L
        self.Ybuff = [] # buffer for the Y terms to check min eig
        self.YYsum = np.zeros((self.L,self.L),dtype=np.float64) # sum of the Y^T*Y terms
        self.YtauSum = np.zeros(self.L,dtype=np.float64) # sum of the Y^T*tau terms
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
        
        Returns:
        -------
        """

        # dont add the data if the minimum eigenvalue is good or the new data has a good minimum singular value
        if not self.lambdaCLMet:
            YSV = np.linalg.norm(Y)
            if (YSV > self.YYminDiff) and (np.linalg.norm(tau) > self.YYminDiff):
                # check to make sure the data is different enough from the other data
                # find the minimum difference
                minDiff = 100.0
                for Yi in self.Ybuff:
                    YYdiffi = np.linalg.norm(Yi-Y)
                    if YYdiffi < minDiff:
                        minDiff = YYdiffi
                
                # if the minimum difference is large enough add the data
                if minDiff > self.YYminDiff:
                    self.Ybuff.append(Y)
                    # print("Y \n"+str(Y))
                    YY = np.outer(Y,Y)
                    # print("YY \n"+str(YY))
                    Ytau = Y.T*tau
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