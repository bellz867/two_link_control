import numpy as np
from numpy.linalg import eigvals

from collections import namedtuple

TrainData = namedtuple('TrainData',('Y','tau','t'))
StackData = namedtuple('StackData',('Y','U'))

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
        self.L = 2*L
        self.deltaT = deltaT # integration window size
        self.intBuff = [] # buffer for the terms to integrate over
        self.stackBuff = [] # buffer for the stack terms
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
        \t t: time \n
        
        Returns:
        -------
        """

        # dont add the data if the minimum eigenvalue is good or the new data has a good minimum singular value
        if not self.lambdaCLMet:
            # add the point to the integral
            self.intBuff.append(TrainData(Y,tau,t))
            Yint = np.zeros(self.L)
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
            if (YSV > self.YYminDiff) and (np.linalg.norm(tauInt) > self.YYminDiff) and (t > self.deltaT):
                addData = False
                YY = np.outer(Yint,Yint)

                #save data until there is more data than size
                # print(len(self.stackBuff))
                if (len(self.stackBuff) > self.L):
                    minDiff = 100.0
                    for Yi,Ui in self.stackBuff:
                        YYdiffi = np.linalg.norm(Yi-Yint)
                        if YYdiffi < minDiff:
                            minDiff = YYdiffi

                    # if enough data begin only adding data if it satisfies a condition
                    # go through and replace each value on the stack to find the worst switch
                    # minEig = 10000.0
                    # minIdx = 0
                    # for ii in range(len(self.stackBuff)):
                    #     YYi = np.outer(self.stackBuff[ii].Y,self.stackBuff[ii].Y)
                    #     YYsumTest = self.YYsum + YY - YYi
                    #     YYsumTestMinEig = np.min(eigvals(YYsumTest))

                    #     if YYsumTestMinEig < minEig:
                    #         minIdx = ii
                    #         minEig = YYsumTestMinEig
                    # print(minEig)
                    # print(minIdx)
                    YYsumTest = self.YYsum + YY
                    YYsumTestMinEig = np.min(eigvals(YYsumTest))

                    Ystack = np.zeros((len(self.stackBuff),self.L),dtype=np.float64)
                    YstackSum = np.zeros((self.L,self.L),dtype=np.float64)
                    for ii in range(len(self.stackBuff)):
                        # print(self.stackBuff[ii].Y)
                        Ystack[ii,:] = self.stackBuff[ii].Y
                        YYi = np.outer(self.stackBuff[ii].Y,self.stackBuff[ii].Y)
                        YstackSum += YYi

                    _,YSVstack,_ = np.linalg.svd(Ystack)
                    YYstack = Ystack.T@Ystack
                    print("svd(Ystack) \n"+str(np.min(YSVstack)))
                    # print("eig(YYstack) \n"+str(eigvals(YYstack)))
                    # print("eig(YstackSum) \n"+str(eigvals(YstackSum)))
                    print("eig(YYsum) \n"+str(np.min(eigvals(self.YYsum))))

                    # switch the new data with the old if it increased the min eigenvalue in the switch
                    if (YYsumTestMinEig > self.YYsumMinEig) and (minDiff > self.YYminDiff):
                        # print("was bigger")
                        # YYmin = np.outer(self.stackBuff[minIdx].Y,self.stackBuff[minIdx].Y)
                        # YtauMin = self.stackBuff[minIdx].Y*self.stackBuff[minIdx].U
                        # self.YYsum -= YYmin
                        # self.YtauSum -= YtauMin
                        # self.stackBuff.pop(minIdx)
                        self.stackBuff.append(StackData(Yint,tauInt))
                        addData = True
                else:
                    # if not enough data just save data if its different enough
                    # self.stackBuff.append(StackData(Yint,tauInt))
                    # addData = True
                    minDiff = 100.0
                    for Yi,Ui in self.stackBuff:
                        YYdiffi = np.linalg.norm(Yi-Yint)
                        if YYdiffi < minDiff:
                            minDiff = YYdiffi
                    if minDiff > self.YYminDiff:
                        self.stackBuff.append(StackData(Yint,tauInt))
                        addData = True
                
                # if add data then add the data
                if addData:
                    # print("YY \n"+str(YY))
                    Ytau = Yint*tauInt
                    # print("Ytau \n"+str(Ytau))
                    self.YYsum += YY
                    # print("YYsum "+str(self.YYsum))
                    self.YtauSum += Ytau
                    self.YYsumMinEig = np.min(eigvals(self.YYsum))
                    # print("eigs "+str(eigvals(self.YYsum)))

                    # check if the new data makes the eigenvalue large enough
                    if self.YYsumMinEig > self.lambdaCL:
                        self.TCL = t
                        self.lambdaCLMet = True

            # print("Nj "+str(len(self.Ybuff)))
            

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