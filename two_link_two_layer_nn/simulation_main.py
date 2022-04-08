import torch
import dynamics
import csv
import os
import datetime
import matplotlib
import matplotlib.pyplot as plot
from matplotlib import rc
from torch import rand as rand

# rc('text',usetex=True)
# matplotlib.rcParams['text.usetex'] = True

devicecuda = torch.device("cuda:0")
devicecpu = torch.device("cpu")

def usecuda(tensor: torch.Tensor,useCuda=False):
    if useCuda and torch.cuda.is_available():
        tensor = tensor.to(devicecuda)
    return tensor

def usecpu(tensor: torch.Tensor):
    tensor = tensor.to(devicecpu)
    return tensor

if __name__ == '__main__':
    tauNoise = 1.0
    phiNoise = 0.04
    phiDNoise = 0.05
    phiDDNoise = 0.06
    # tauNoise = 0.001
    # phiNoise = 0.001
    # phiDNoise = 0.001
    # phiDDNoise = 0.001
    useNN = True
    useYth = True
    useBias = False
    useCuda = False
    dt = 0.01 # time step
    tf = 30.0 # final time
    t = usecuda(torch.linspace(0.0,tf,int(tf/dt),dtype=torch.float),useCuda=useCuda) # times
    alpha = 3.0*torch.eye(2)
    betar = 1.5*torch.eye(2)
    betadel = 0.001*torch.eye(2)
    gammath = 0.1
    gammaw = 0.1
    

    memorySize = 1000
    batchSize = 200
    numberEpochs = 200
    learnRate = 0.01
    
    l = 20
    L = l+1
    if not useBias:
        L -= 1
    
    dyn = dynamics.Dynamics(alpha=alpha,betar=betar,betadel=betadel,gammath=gammath,gammaw=gammaw,tauN=tauNoise,phiN=phiNoise,phiDN=phiDNoise,phiDDN=phiDDNoise,l=l,useNN=useNN,useYth=useYth,useBias=useBias,useCuda=useCuda,memorySize=memorySize,batchSize=batchSize,numberEpochs=numberEpochs,learnRate=learnRate)
    phiHist = usecuda(torch.zeros((2,len(t)),dtype=torch.float),useCuda=useCuda)
    phidHist = usecuda(torch.zeros((2,len(t)),dtype=torch.float),useCuda=useCuda)
    eHist = usecuda(torch.zeros((2,len(t)),dtype=torch.float),useCuda=useCuda)
    eNormHist = usecuda(torch.zeros(len(t),dtype=torch.float),useCuda=useCuda)
    rHist = usecuda(torch.zeros((2,len(t)),dtype=torch.float),useCuda=useCuda)
    rNormHist = usecuda(torch.zeros(len(t),dtype=torch.float),useCuda=useCuda)
    thetaHHist = usecuda(torch.zeros((5,len(t)),dtype=torch.float),useCuda=useCuda)
    WHHist = usecuda(torch.zeros((2*L,len(t)),dtype=torch.float),useCuda=useCuda)
    tauHist = usecuda(torch.zeros((2,len(t)),dtype=torch.float),useCuda=useCuda)
    tauffHist = usecuda(torch.zeros((2,len(t)),dtype=torch.float),useCuda=useCuda)
    taufbHist = usecuda(torch.zeros((2,len(t)),dtype=torch.float),useCuda=useCuda)
    fHist = usecuda(torch.zeros((2,len(t)),dtype=torch.float),useCuda=useCuda)
    fHHist = usecuda(torch.zeros((2,len(t)),dtype=torch.float),useCuda=useCuda)
    fDiffNormHist = usecuda(torch.zeros(len(t),dtype=torch.float),useCuda=useCuda)
    lossHist = usecuda(torch.zeros(len(t),dtype=torch.float),useCuda=useCuda)

    #start save file
    savePath = "C:/Users/bell_/OneDrive/Documents/_teaching/AdaptiveControlSpring2022/projects/project4/two_link_two_layer_nn"
    now = datetime.datetime.now()
    nownew = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = savePath+"/sim-"+nownew
    os.mkdir(path)
    
    # loop through
    for jj in range(0,len(t)):
        print("time "+str(tf*jj/len(t)))
        # get the state and input data
        phidj,phiDdj,phiDDdj = dyn.getDesiredState(t[jj])
        phidj = usecuda(phidj,useCuda=useCuda)
        phiDdj = usecuda(phiDdj,useCuda=useCuda)
        phiDDdj = usecuda(phiDDdj,useCuda=useCuda)
        phimj,phiDmj,phiDDmj,thetaHj,WHj = dyn.getState(t[jj])
        emj,eDmj,rmj = dyn.getErrorState(t[jj])
        tauj,tauffj,taufbj,_,_,_ = dyn.getTau(t[jj],phi=phimj,phiD=phiDmj,thetaH=thetaHj,WH=WHj)
        fj,fHj = dyn.getfuncComp(phi=phimj,phiD=phiDmj,phiDD=phiDDmj,tau=tauj,thetaH=thetaHj,WH=WHj)
        fDiffNormj = torch.linalg.norm(fj-fHj)
        lossj = usecuda(dyn.getLoss(),useCuda=useCuda)
        
        # save the data to the buffers
        phiHist[:,jj] = phimj
        phidHist[:,jj] = phidj
        eHist[:,jj] = emj
        rHist[:,jj] = rmj
        eNormHist[jj] = torch.linalg.norm(emj)
        eNormHist[jj] = torch.linalg.norm(rmj)
        thetaHHist[:,jj] = thetaHj
        WHHist[:,jj] = torch.reshape(WHj.T,(2*L,))
        tauHist[:,jj] = tauj
        tauffHist[:,jj] = tauffj
        taufbHist[:,jj] = taufbj
        fHist[:,jj] = fj
        fHHist[:,jj] = fHj
        fDiffNormHist[jj] = fDiffNormj
        lossHist[jj] = lossj

        if torch.linalg.norm(phimj) > 5.0*torch.linalg.norm(phidj) or torch.linalg.norm(tauj) > 1000:
            print("GOING UNSTABLE")
            break
        
        # step the internal state of the dyanmics
        dyn.step(dt,t[jj])

    # plot the data
    if useCuda:
        t = usecpu(t)
        phiHist = usecpu(phiHist)
        phidHist = usecpu(phidHist)
        eHist = usecpu(eHist)
        eNormHist = usecpu(eNormHist)
        rHist = usecpu(rHist)
        rNormHist = usecpu(rNormHist)
        thetaHHist = usecpu(thetaHHist)
        WHHist = usecpu(WHHist)
        tauHist = usecpu(tauHist)
        tauffHist = usecpu(tauffHist)
        taufbHist = usecpu(taufbHist)
        fHist = usecpu(fHist)
        fHHist = usecpu(fHHist)
        fDiffNormHist = usecpu(fDiffNormHist)
        lossHist = usecpu(lossHist)

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
    eNax.set_title("Error Norm RMS = "+"{:.2f}".format(torch.round(torch.sqrt(torch.mean(eNormHist**2)),decimals=2)))
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
        thHax.plot(t,thetaHHist[ii,:],color=rand(3).numpy(),linewidth=2,linestyle='--')
    thHax.set_xlabel("$t$ $(sec)$")
    thHax.set_ylabel("$\\theta_"+str(ii)+"$")
    thHax.set_title("Structured Parameter Estimates")
    thHax.grid()
    thHplot.savefig(path+"/thetaHat.pdf")

    #plot the parameter estiamtes
    WHplot,WHax = plot.subplots()
    for ii in range(2*L):
        WHax.plot(t,WHHist[ii,:],color=rand(3).numpy(),linewidth=2,linestyle='--')
    WHax.set_xlabel("$t$ $(sec)$")
    WHax.set_ylabel("$W_i$")
    WHax.set_title("Unstructured Outer Weight Estimates")
    WHax.grid()
    WHplot.savefig(path+"/WHat.pdf")

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
    fdax.set_title("Function Difference Norm RMS = "+"{:.2f}".format(torch.round(torch.sqrt(torch.mean(fDiffNormHist**2)),decimals=2)))
    fdax.grid()
    fdplot.savefig(path+"/fdiffnorm.pdf")

    #plot the loss
    lsplot,lsax = plot.subplots()
    lsax.plot(t,lossHist,color='orange',linewidth=2,linestyle='-')
    lsax.set_xlabel("$t$ $(sec)$")
    lsax.set_ylabel("$loss$")
    lsax.set_title("Average Loss Norm RMS = "+"{:.2f}".format(torch.round(torch.sqrt(torch.mean(lossHist**2)),decimals=2)))
    lsax.grid()
    lsplot.savefig(path+"/loss.pdf")