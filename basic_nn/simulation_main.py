import numpy as np
import dynamics
import csv
import os
import datetime
import matplotlib
import matplotlib.pyplot as plot
from matplotlib import rc
# rc('text',usetex=True)
# matplotlib.rcParams['text.usetex'] = True


if __name__ == '__main__':
    uNoise = 0.1
    xNoise = 0.1
    xDNoise = 0.1
    useCL = True
    dt = 0.01 # time step
    tf = 120.0 # final time
    t = np.linspace(0.0,tf,int(tf/dt),dtype=np.float64) # times
    betae = 1.0
    betaeps = 0.1
    gamma = 0.05
    lambdaCL = 0.1
    YYminDiff = 0.2
    deltaT = 1.0
    kCL = 0.1
    L = 4
    dyn = dynamics.Dynamics(betae=betae,betaeps=betaeps,gamma=gamma,lambdaCL=lambdaCL,YYminDiff=YYminDiff,kCL=kCL,uN=uNoise,xN=xNoise,xDN=xDNoise,L=L,deltaT=deltaT,useCL=useCL)
    xHist = np.zeros(len(t),dtype=np.float64)
    xdHist = np.zeros(len(t),dtype=np.float64)
    eHist = np.zeros(len(t),dtype=np.float64)
    eNormHist = np.zeros(len(t),dtype=np.float64)
    WHHist = np.zeros((L+1,len(t)),dtype=np.float64)
    lambdaCLMinHist = np.zeros(len(t),dtype=np.float64)
    uHist = np.zeros(len(t),dtype=np.float64)
    uffHist = np.zeros(len(t),dtype=np.float64)
    ufbHist = np.zeros(len(t),dtype=np.float64)
    fHist = np.zeros(len(t),dtype=np.float64)
    fHHist = np.zeros(len(t),dtype=np.float64)
    fDiffNormHist = np.zeros(len(t),dtype=np.float64)
    TCL = 0
    TCLindex = 0
    TCLfound = False
    TCLm = 0
    TCLmindex = 0
    TCLmfound = False

    #start save file
    savePath = "C:/Users/bell_/OneDrive/Documents/_teaching/AdaptiveControlSpring2022/projects/project3/basic_nn"
    now = datetime.datetime.now()
    nownew = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = savePath+"/sim-"+nownew
    os.mkdir(path)
    
    # loop through
    for jj in range(0,len(t)):
        # get the state and input data
        xdj,xDdj = dyn.getDesiredState(t[jj])
        emj,eDmj,xmj,xDmj,WHj = dyn.getErrorState(t[jj])
        uj,WHDj,uffj,ufbj = dyn.getinputWHD(t[jj])
        lamdaCLMinj,TCLj,_,_ = dyn.getCLstate()
        fj,fHj = dyn.getfuncComp(xmj,WHj)
        fDiffNormj = np.linalg.norm(fj-fHj)

        if not TCLfound:
            if TCLj > 0:
                TCL = TCLj
                TCLindex = jj
                TCLfound = True
        
        # save the data to the buffers
        xHist[jj] = xmj
        xdHist[jj] = xdj
        eHist[jj] = emj
        eNormHist[jj] = np.linalg.norm(emj)
        WHHist[:,jj] = WHj
        lambdaCLMinHist[jj] = lamdaCLMinj
        uHist[jj] = uj
        uffHist[jj] = uffj
        ufbHist[jj] = ufbj
        fHist[jj] = fj
        fHHist[jj] = fHj
        fDiffNormHist[jj] = fDiffNormj
        

        # step the internal state of the dyanmics
        dyn.step(dt,t[jj])

    # plot the data
    #plot the angles
    xplot,xax = plot.subplots()
    xax.plot(t,xdHist,color='orange',linewidth=2,linestyle='-')
    xax.plot(t,xHist,color='blue',linewidth=2,linestyle='-')
    xax.set_xlabel("$t$ $(sec)$")
    xax.set_ylabel("$x$")
    xax.set_title("Position")
    xax.legend(["$x_{d}$","$x$"],loc='upper right')
    xax.grid()
    xplot.savefig(path+"/position.pdf")

    #plot the error norm
    eNplot,eNax = plot.subplots()
    eNax.plot(t,eNormHist,color='orange',linewidth=2,linestyle='-')
    eNax.set_xlabel("$t$ $(sec)$")
    eNax.set_ylabel("$\Vert e \Vert$")
    eNax.set_title("Error Norm")
    eNax.grid()
    eNplot.savefig(path+"/errorNorm.pdf")

    #plot the inputs
    uplot,uax = plot.subplots()
    uax.plot(t,uHist,color='red',linewidth=2,linestyle='-')
    uax.plot(t,uffHist,color='green',linewidth=2,linestyle='--')
    uax.plot(t,ufbHist,color='blue',linewidth=2,linestyle='-.')
    uax.set_xlabel("$t$ $(sec)$")
    uax.set_ylabel("$input$")
    uax.set_title("Control Input")
    uax.legend(["$\\tau$","$\\tau_{ff}$","$\\tau_{fb}$"],loc='upper right')
    uax.grid()
    uplot.savefig(path+"/input.pdf")

    #plot the parameter estiamtes
    WHplot,WHax = plot.subplots()
    for ii in range(L+1):
        WHax.plot(t,WHHist[ii,:],color=np.random.rand(3),linewidth=2,linestyle='-')
    WHax.set_xlabel("$t$ $(sec)$")
    WHax.set_ylabel("$W_i$")
    WHax.set_title("Parameter Estimates")
    WHax.grid()
    WHplot.savefig(path+"/WHat.pdf")

    #plot the approx
    fplot,fax = plot.subplots()
    fax.plot(t,fHist,color='orange',linewidth=2,linestyle='-')
    fax.plot(t,fHHist,color='blue',linewidth=2,linestyle='--')
    fax.set_xlabel("$t$ $(sec)$")
    fax.set_ylabel("$function$")
    fax.set_title("Function Approximate")
    fax.legend(["$f$","$\hat{f}$"],loc='upper right')
    fax.grid()
    fplot.savefig(path+"/fapprox.pdf")

    #plot the approx norm
    fdplot,fdax = plot.subplots()
    fdax.plot(t,fDiffNormHist,color='orange',linewidth=2,linestyle='-')
    fdax.set_xlabel("$t$ $(sec)$")
    fdax.set_ylabel("$function$")
    fdax.set_title("Function Difference Norm")
    fdax.grid()
    fdplot.savefig(path+"/fdiffnorm.pdf")

    #plot the approx norm
    fdplot,fdax = plot.subplots()
    fdax.plot(t[TCLindex:],fDiffNormHist[TCLindex:],color='orange',linewidth=2,linestyle='-')
    fdax.set_xlabel("$t$ $(sec)$")
    fdax.set_ylabel("$function$")
    fdax.set_title("Function Difference Norm After Learn")
    fdax.grid()
    fdplot.savefig(path+"/fdiffnormafter.pdf")

    

    #plot the minimum eigenvalue
    eigplot,eigax = plot.subplots()
    eigax.plot(t,lambdaCLMinHist,color='orange',linewidth=2,linestyle='-')
    eigax.plot([TCL,TCL],[0.0,lambdaCLMinHist[TCLindex]],color='black',linewidth=1,linestyle='-')
    eigax.set_xlabel("$t$ $(sec)$")
    eigax.set_ylabel("$\lambda_{min}$")
    eigax.set_title("Minimum Eigenvalue $T_{CL}$="+str(round(TCL,2)))
    eigax.grid()
    eigplot.savefig(path+"/minEig.pdf")
    

    
                