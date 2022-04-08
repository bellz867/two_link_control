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
    tauNoise = 1.0
    phiNoise = 0.04
    phiDNoise = 0.05
    phiDDNoise = 0.06
    # tauNoise = 0.001
    # phiNoise = 0.001
    # phiDNoise = 0.001
    # phiDDNoise = 0.001
    useCL = False
    useNN = True
    useYth = True
    useBias = True
    dt = 0.01 # time step
    tf = 5.0 # final time
    t = np.linspace(0.0,tf,int(tf/dt),dtype=np.float64) # times
    alpha = 3.0*np.identity(2)
    betar = 1.5*np.identity(2)
    betadel = 0.001*np.identity(2)
    gammath = 0.5
    gammaw = 0.5
    gammav = 10.0
    lambdaCL = 0.0001
    YYminDiff = 0.1
    deltaT = 1.5
    kCL = 0.2
    l = 20
    L = l+1
    dyn = dynamics.Dynamics(alpha=alpha,betar=betar,betadel=betadel,gammath=gammath,gammaw=gammaw,gammav=gammav,lambdaCL=lambdaCL,YYminDiff=YYminDiff,kCL=kCL,tauN=tauNoise,phiN=phiNoise,phiDN=phiDNoise,phiDDN=phiDDNoise,l=l,deltaT=deltaT,useCL=useCL,useNN=useNN,useYth=useYth,useBias=useBias)
    phiHist = np.zeros((2,len(t)),dtype=np.float64)
    phidHist = np.zeros((2,len(t)),dtype=np.float64)
    eHist = np.zeros((2,len(t)),dtype=np.float64)
    eNormHist = np.zeros(len(t),dtype=np.float64)
    rHist = np.zeros((2,len(t)),dtype=np.float64)
    rNormHist = np.zeros(len(t),dtype=np.float64)
    thetaHHist = np.zeros((5,len(t)),dtype=np.float64)
    WHHist = np.zeros((2*L,len(t)),dtype=np.float64)
    VHHist = np.zeros((5*l,len(t)),dtype=np.float64)
    lambdaCLMinHist = np.zeros(len(t),dtype=np.float64)
    tauHist = np.zeros((2,len(t)),dtype=np.float64)
    tauffHist = np.zeros((2,len(t)),dtype=np.float64)
    taufbHist = np.zeros((2,len(t)),dtype=np.float64)
    fHist = np.zeros((2,len(t)),dtype=np.float64)
    fHHist = np.zeros((2,len(t)),dtype=np.float64)
    fDiffNormHist = np.zeros(len(t),dtype=np.float64)
    TCL = 0
    TCLindex = 0
    TCLfound = False
    TCLm = 0
    TCLmindex = 0
    TCLmfound = False

    #start save file
    savePath = "C:/Users/bell_/OneDrive/Documents/_teaching/AdaptiveControlSpring2022/projects/project3/two_link_two_layer"
    now = datetime.datetime.now()
    nownew = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = savePath+"/sim-"+nownew
    os.mkdir(path)
    
    # loop through
    for jj in range(0,len(t)):
        # get the state and input data
        phidj,phiDdj,phiDDdj = dyn.getDesiredState(t[jj])
        phimj,phiDmj,phiDDmj,thetaHj,WHj,VHj = dyn.getState(t[jj])
        emj,eDmj,rmj = dyn.getErrorState(t[jj])
        tauj,tauffj,taufbj,_,_,_ = dyn.getTau(t[jj],phi=phimj,phiD=phiDmj,thetaH=thetaHj,WH=WHj,VH=VHj)
        lamdaCLMinj,TCLj,_,_ = dyn.getCLstate()
        fj,fHj = dyn.getfuncComp(phi=phimj,phiD=phiDmj,phiDD=phiDDmj,tau=tauj,thetaH=thetaHj,WH=WHj,VH=VHj)
        fDiffNormj = np.linalg.norm(fj-fHj)

        if not TCLfound:
            if TCLj > 0:
                TCL = TCLj
                TCLindex = jj
                TCLfound = True
        
        # save the data to the buffers
        phiHist[:,jj] = phimj
        phidHist[:,jj] = phidj
        eHist[:,jj] = emj
        rHist[:,jj] = rmj
        eNormHist[jj] = np.linalg.norm(emj)
        eNormHist[jj] = np.linalg.norm(rmj)
        thetaHHist[:,jj] = thetaHj
        WHHist[:,jj] = np.reshape(WHj.T,(2*L))
        VHHist[:,jj] = np.reshape(VHj,(5*l))
        lambdaCLMinHist[jj] = lamdaCLMinj
        tauHist[:,jj] = tauj
        tauffHist[:,jj] = tauffj
        taufbHist[:,jj] = taufbj
        fHist[:,jj] = fj
        fHHist[:,jj] = fHj
        fDiffNormHist[jj] = fDiffNormj

        if np.linalg.norm(phimj) > 5.0*np.linalg.norm(phidj) or np.linalg.norm(tauj) > 1000:
            print("GOING UNSTABLE")
            break
        
        # step the internal state of the dyanmics
        dyn.step(dt,t[jj])

    # plot the data
    #plot the angles
    phiplot,phiax = plot.subplots()
    phiax.plot(t,phidHist[0,:],color='orange',linewidth=2,linestyle='-')
    phiax.plot(t,phiHist[0,:],color='orange',linewidth=2,linestyle='--')
    phiax.plot(t,phidHist[1,:],color='blue',linewidth=2,linestyle='-')
    phiax.plot(t,phiHist[1,:],color='blue',linewidth=2,linestyle='--')
    phiax.set_xlabel("$t$ $(sec)$")
    phiax.set_ylabel("$\phi$")
    phiax.set_title("Angles")
    phiax.legend(["$\phi_{d1}$","$\phi_1$","$\phi_{d2}$","$\phi_2$"],loc='upper right')
    phiax.grid()
    phiplot.savefig(path+"/angle.pdf")

    #plot the error norm
    eNplot,eNax = plot.subplots()
    eNax.plot(t,eNormHist,color='orange',linewidth=2,linestyle='-')
    eNax.set_xlabel("$t$ $(sec)$")
    eNax.set_ylabel("$\Vert e \Vert$")
    eNax.set_title("Error Norm RMS = "+str(np.around(np.sqrt(np.mean(eNormHist**2)),decimals=2)))
    eNax.grid()
    eNplot.savefig(path+"/errorNorm.pdf")

    #plot the inputs
    uplot,uax = plot.subplots()
    uax.plot(t,tauHist[0,:],color='red',linewidth=2,linestyle='-')
    uax.plot(t,tauffHist[0,:],color='green',linewidth=2,linestyle='-')
    uax.plot(t,taufbHist[0,:],color='blue',linewidth=2,linestyle='-')
    uax.plot(t,tauHist[1,:],color='red',linewidth=2,linestyle='--')
    uax.plot(t,tauffHist[1,:],color='green',linewidth=2,linestyle='--')
    uax.plot(t,taufbHist[1,:],color='blue',linewidth=2,linestyle='--')
    uax.set_xlabel("$t$ $(sec)$")
    uax.set_ylabel("$input$")
    uax.set_title("Control Input")
    uax.legend(["$\\tau_1$","$\\tau_{ff1}$","$\\tau_{fb1}$","$\\tau_2$","$\\tau_{ff2}$","$\\tau_{fb2}$"],loc='upper right')
    uax.grid()
    uplot.savefig(path+"/input.pdf")

    #plot the parameter estiamtes
    thHplot,thHax = plot.subplots()
    for ii in range(5):
        thHax.plot(t,thetaHHist[ii,:],color=np.random.rand(3),linewidth=2,linestyle='--')
    thHax.set_xlabel("$t$ $(sec)$")
    thHax.set_ylabel("$\theta_"+str(ii)+"$")
    thHax.set_title("Structured Parameter Estimates")
    thHax.grid()
    thHplot.savefig(path+"/thetaHat.pdf")

    #plot the parameter estiamtes
    WHplot,WHax = plot.subplots()
    for ii in range(2*L):
        WHax.plot(t,WHHist[ii,:],color=np.random.rand(3),linewidth=2,linestyle='--')
    WHax.set_xlabel("$t$ $(sec)$")
    WHax.set_ylabel("$W_i$")
    WHax.set_title("Unstructured Outer Weight Estimates")
    WHax.grid()
    WHplot.savefig(path+"/WHat.pdf")

    #plot the parameter estiamtes
    VHplot,VHax = plot.subplots()
    for ii in range(5*l):
        VHax.plot(t,VHHist[ii,:],color=np.random.rand(3),linewidth=2,linestyle='--')
    VHax.set_xlabel("$t$ $(sec)$")
    VHax.set_ylabel("$V_i$")
    VHax.set_title("Unstructured Inner Weight Estimates")
    VHax.grid()
    VHplot.savefig(path+"/VHat.pdf")

    #plot the approx
    fplot,fax = plot.subplots()
    fax.plot(t,fHist[0,:],color='orange',linewidth=2,linestyle='-')
    fax.plot(t,fHHist[0,:],color='orange',linewidth=2,linestyle='--')
    fax.plot(t,fHist[1,:],color='blue',linewidth=2,linestyle='-')
    fax.plot(t,fHHist[1,:],color='blue',linewidth=2,linestyle='--')
    fax.set_xlabel("$t$ $(sec)$")
    fax.set_ylabel("$function$")
    fax.set_title("Function Approximate")
    fax.legend(["$f_1$","$\hat{f1}$","$f_2$","$\hat{f}_2$"],loc='upper right')
    fax.grid()
    fplot.savefig(path+"/fapprox.pdf")

    #plot the approx norm
    fdplot,fdax = plot.subplots()
    fdax.plot(t,fDiffNormHist,color='orange',linewidth=2,linestyle='-')
    fdax.set_xlabel("$t$ $(sec)$")
    fdax.set_ylabel("$function$")
    fdax.set_title("Function Difference Norm RMS = "+str(np.around(np.sqrt(np.mean(fDiffNormHist**2)),decimals=2)))
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
    

    # tsigmas = np.linspace(0.0,5,int(5/dt),dtype=np.float64)
    # phidsigmas = np.zeros((2,len(tsigmas)),dtype=np.float64)
    # phiDdsigmas = np.zeros((2,len(tsigmas)),dtype=np.float64)
    # sigmaHist = np.zeros((L,len(tsigmas)),dtype=np.float64)
    # for jj in range(0,len(tsigmas)):
    #     # get the state and input data
    #     phidj,phiDdj,_ = dyn.getDesiredState(tsigmas[jj])
    #     sigmaj = dyn.getsigma(phidj,phiDdj)
    #     sigmaHist[:,jj] = sigmaj
    #     phidsigmas[:,jj] = phidj
    #     phiDdsigmas[:,jj] = phiDdj

    # #plot the sigmas estiamtes
    # sigmaplot,sigmaax = plot.subplots()
    # for ii in range(L):
    #     sigmaax.plot(phidsigmas[0,:],sigmaHist[4*ii,:],color=np.random.rand(3),linewidth=2,linestyle='-')
    # # sigmaax.plot(tsigmas,phidsigmas[0,:],color='orange',linewidth=2,linestyle='-')
    # sigmaax.set_xlabel("$position$")
    # sigmaax.set_ylabel("$sigma$")
    # sigmaax.set_title("Sigmas Position1")
    # sigmaax.grid()
    # sigmaplot.savefig(path+"/sigmasPosition1.pdf")

    # #plot the sigmas estiamtes
    # sigmaplot,sigmaax = plot.subplots()
    # for ii in range(L):
    #     sigmaax.plot(phidsigmas[1,:],sigmaHist[4*ii+1,:],color=np.random.rand(3),linewidth=2,linestyle='-')
    # # sigmaax.plot(tsigmas,phidsigmas[1,:],color='orange',linewidth=2,linestyle='-')
    # sigmaax.set_xlabel("$position$")
    # sigmaax.set_ylabel("$sigma$")
    # sigmaax.set_title("Sigmas Position2")
    # sigmaax.grid()
    # sigmaplot.savefig(path+"/sigmasPosition2.pdf")

    # #plot the sigmas estiamtes
    # sigmaplot,sigmaax = plot.subplots()
    # for ii in range(L):
    #     sigmaax.plot(phiDdsigmas[0,:],sigmaHist[4*ii+2,:],color=np.random.rand(3),linewidth=2,linestyle='-')
    # # sigmaax.plot(tsigmas,phiDdsigmas[0,:],color='orange',linewidth=2,linestyle='-')
    # sigmaax.set_xlabel("$velocity$")
    # sigmaax.set_ylabel("$sigma$")
    # sigmaax.set_title("Sigmas Velocity1")
    # sigmaax.grid()
    # sigmaplot.savefig(path+"/sigmasvelocity1.pdf")

    # #plot the sigmas estiamtes
    # sigmaplot,sigmaax = plot.subplots()
    # for ii in range(L):
    #     sigmaax.plot(phiDdsigmas[1,:],sigmaHist[4*ii+3,:],color=np.random.rand(3),linewidth=2,linestyle='-')
    # # sigmaax.plot(tsigmas,phiDdsigmas[1,:],color='orange',linewidth=2,linestyle='-')
    # sigmaax.set_xlabel("$velocity$")
    # sigmaax.set_ylabel("$sigma$")
    # sigmaax.set_title("Sigmas Velocity2")
    # sigmaax.grid()
    # sigmaplot.savefig(path+"/sigmasvelocity2.pdf")

    
    
    
                