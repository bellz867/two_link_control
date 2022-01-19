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
    dt = 0.005 # time step
    tf = 60.0 # final time
    t = np.linspace(0.0,tf,int(tf/dt),dtype=np.float32) # times
    alpha = 5.0*np.ones(2,dtype=np.float32)
    beta = 2.5*np.ones(2,dtype=np.float32)
    gamma = 1.25*np.ones(5,dtype=np.float32)
    dyn = dynamics.Dynamics(alpha=alpha,beta=beta,gamma=gamma)
    phiHist = np.zeros((2,len(t)),dtype=np.float32)
    phidHist = np.zeros((2,len(t)),dtype=np.float32)
    phiDHist = np.zeros((2,len(t)),dtype=np.float32)
    phiDdHist = np.zeros((2,len(t)),dtype=np.float32)
    phiDDHist = np.zeros((2,len(t)),dtype=np.float32)
    phiDDdHist = np.zeros((2,len(t)),dtype=np.float32)
    eHist = np.zeros((2,len(t)),dtype=np.float32)
    eNormHist = np.zeros_like(t)
    rHist = np.zeros((2,len(t)),dtype=np.float32)
    rNormHist = np.zeros_like(t)
    thetaTildeHist = np.zeros((5,len(t)),dtype=np.float32)
    thetaTildeNormHist = np.zeros_like(t)
    tauHist = np.zeros((2,len(t)),dtype=np.float32)
    tauffHist = np.zeros((2,len(t)),dtype=np.float32)
    taufbHist = np.zeros((2,len(t)),dtype=np.float32)

    #start save file
    savePath = "C:/Users/bell_/OneDrive/Documents/_teaching/AdaptiveControlSpring2022/projects/project1"
    now = datetime.datetime.now()
    nownew = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = savePath+"/sim-"+nownew
    os.mkdir(path)
    file = open(path+"/data.csv","w",newline='')
    # write the header into the file
    with file:
        write = csv.writer(file)
        write.writerow(["time","e1","e2","r1","r2","tau1","tau2"])
    file.close()
    
    # loop through
    for jj in range(0,len(t)):
        # get the state and input data
        phij,phiDj,phiDDj,thetaHj = dyn.getState(t[jj])
        phidj,phiDdj,phiDDdj = dyn.getDesiredState(t[jj])
        ej,_,rj,thetaTildej = dyn.getErrorState(t[jj])
        tauj,_,tauffj,taufbj = dyn.getTauThetaHD(t[jj])
        
        # save the data to the buffers
        phiHist[:,jj] = phij
        phidHist[:,jj] = phidj
        phiDHist[:,jj] = phiDj
        phiDdHist[:,jj] = phiDdj
        phiDDHist[:,jj] = phiDDj
        phiDDdHist[:,jj] = phiDDdj
        eHist[:,jj] = ej
        eNormHist[jj] = np.linalg.norm(ej)
        rHist[:,jj] = rj
        rNormHist[jj] = np.linalg.norm(rj)
        thetaTildeHist[:,jj] = thetaTildej
        thetaTildeNormHist[jj] = np.linalg.norm(thetaTildej)
        tauHist[:,jj] = tauj
        tauffHist[:,jj] = tauffj
        taufbHist[:,jj] = taufbj

        #save select data to file
        file = open(path+"/data.csv","a",newline='')
        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerow([t[jj],eHist[0,jj],eHist[1,jj],rHist[0,jj],rHist[1,jj],tauHist[0,jj],tauHist[1,jj]])
        file.close()

        # step the internal state of the dyanmics
        dyn.step(dt,t[jj])

    # plot the data
    #plot the angles
    phiplot,phiax = plot.subplots()
    phiax.plot(t,phidHist[0,:],color='orange',linewidth=2,linestyle='--')
    phiax.plot(t,phiHist[0,:],color='orange',linewidth=2,linestyle='-')
    phiax.plot(t,phidHist[1,:],color='blue',linewidth=2,linestyle='--')
    phiax.plot(t,phiHist[1,:],color='blue',linewidth=2,linestyle='-')
    phiax.set_xlabel("$t$ $(sec)$")
    phiax.set_ylabel("$\phi_i$ $(rad)$")
    phiax.set_title("Angle")
    phiax.legend(["$\phi_{1d}$","$\phi_1$","$\phi_{2d}$","$\phi_2$"],loc='upper right')
    phiax.grid()
    phiplot.savefig(path+"/angles.pdf")

    #plot the error
    eplot,eax = plot.subplots()
    eax.plot(t,eHist[0,:],color='orange',linewidth=2,linestyle='-')
    eax.plot(t,eHist[1,:],color='blue',linewidth=2,linestyle='-')
    eax.set_xlabel("$t$ $(sec)$")
    eax.set_ylabel("$e_i$ $(rad)$")
    eax.set_title("Error")
    eax.legend(["$e_1$","$e_2$"],loc='upper right')
    eax.grid()
    eplot.savefig(path+"/error.pdf")

    #plot the error norm
    eNplot,eNax = plot.subplots()
    eNax.plot(t,eNormHist,color='orange',linewidth=2,linestyle='-')
    eNax.set_xlabel("$t$ $(sec)$")
    eNax.set_ylabel("$\Vert e \Vert$ $(rad)$")
    eNax.set_title("Error Norm")
    eNax.grid()
    eNplot.savefig(path+"/errorNorm.pdf")

    #plot the anglular velocity
    phiDplot,phiDax = plot.subplots()
    phiDax.plot(t,phiDdHist[0,:],color='orange',linewidth=2,linestyle='--')
    phiDax.plot(t,phiDHist[0,:],color='orange',linewidth=2,linestyle='-')
    phiDax.plot(t,phiDdHist[1,:],color='blue',linewidth=2,linestyle='--')
    phiDax.plot(t,phiDHist[1,:],color='blue',linewidth=2,linestyle='-')
    phiDax.set_xlabel("$t$ $(sec)$")
    phiDax.set_ylabel("$anglular velocity$ $(rad/sec)$")
    phiDax.set_title("Anglular Velocity")
    phiDax.legend(["$\dot{\phi}_{1d}$","$\dot{\phi}_1$","$\dot{\phi}_{2d}$","$\dot{\phi}_2$"],loc='upper right')
    phiDax.grid()
    phiDplot.savefig(path+"/anglularVelocity.pdf")

    #plot the filtered error
    rplot,rax = plot.subplots()
    rax.plot(t,rHist[0,:],color='orange',linewidth=2,linestyle='-')
    rax.plot(t,rHist[1,:],color='blue',linewidth=2,linestyle='-')
    rax.set_xlabel("$t$ $(sec)$")
    rax.set_ylabel("$r_i$ $(rad/sec)$")
    rax.set_title("Filtered Error")
    rax.legend(["$r_1$","$r_2$"],loc='upper right')
    rax.grid()
    rplot.savefig(path+"/filteredError.pdf")

    #plot the filtered error norm
    rNplot,rNax = plot.subplots()
    rNax.plot(t,rNormHist,color='orange',linewidth=2,linestyle='-')
    rNax.set_xlabel("$t$ $(sec)$")
    rNax.set_ylabel("$\Vert r \Vert$ $(rad)$")
    rNax.set_title("Filtered Error Norm")
    rNax.grid()
    rNplot.savefig(path+"/filteredErrorNorm.pdf")

    #plot the anglular acceleration
    phiDDplot,phiDDax = plot.subplots()
    phiDDax.plot(t,phiDDdHist[0,:],color='orange',linewidth=2,linestyle='--')
    phiDDax.plot(t,phiDDHist[0,:],color='orange',linewidth=2,linestyle='-')
    phiDDax.plot(t,phiDDdHist[1,:],color='blue',linewidth=2,linestyle='--')
    phiDDax.plot(t,phiDDHist[1,:],color='blue',linewidth=2,linestyle='-')
    phiDDax.set_xlabel("$t$ $(sec)$")
    phiDDax.set_ylabel("$anglular acceleration$ $(rad/sec^2)$")
    phiDDax.set_title("Anglular Acceleration")
    phiDDax.legend(["$\ddot{\phi}_{1d}$","$\ddot{\phi}_1$","$\ddot{\phi}_{2d}$","$\ddot{\phi}_2$"],loc='upper right')
    phiDDax.grid()
    phiDDplot.savefig(path+"/anglularAcceleration.pdf")

    #plot the inputs
    tauplot,tauax = plot.subplots()
    tauax.plot(t,tauHist[0,:],color='orange',linewidth=2,linestyle='-')
    tauax.plot(t,tauHist[1,:],color='blue',linewidth=2,linestyle='-')
    tauax.plot(t,tauffHist[0,:],color='orange',linewidth=2,linestyle='--')
    tauax.plot(t,tauffHist[1,:],color='blue',linewidth=2,linestyle='--')
    tauax.plot(t,taufbHist[0,:],color='orange',linewidth=2,linestyle='-.')
    tauax.plot(t,taufbHist[1,:],color='blue',linewidth=2,linestyle='-.')
    tauax.set_xlabel("$t$ $(sec)$")
    tauax.set_ylabel("$input$ $(Nm)$")
    tauax.set_title("Control Input")
    tauax.legend(['$\\tau_1$',"$\\tau_2$","$\\tau_{ff1}$","$\\tau_{ff2}$","$\\tau_{fb1}$","$\\tau_{fb2}$"],loc='upper right')
    tauax.grid()
    tauplot.savefig(path+"/input.pdf")

    #plot the parameter estiamtes
    thetaplot,thetaax = plot.subplots()
    thetaax.plot(t,thetaTildeHist[0,:],color='red',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildeHist[1,:],color='green',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildeHist[2,:],color='blue',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildeHist[3,:],color='orange',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildeHist[4,:],color='magenta',linewidth=2,linestyle='-')
    thetaax.set_xlabel("$t$ $(sec)$")
    thetaax.set_ylabel("$\\tilde{\\theta}_i$")
    thetaax.set_title("Parameter Error")
    thetaax.legend(["$\\tilde{\\theta}_1$","$\\tilde{\\theta}_2$","$\\tilde{\\theta}_3$","$\\tilde{\\theta}_4$","$\\tilde{\\theta}_5$"],loc='upper right')
    thetaax.grid()
    thetaplot.savefig(path+"/thetaTilde.pdf")

    #plot the parameter estiamtes norm
    thetaNplot,thetaNax = plot.subplots()
    thetaNax.plot(t,thetaTildeNormHist,color='orange',linewidth=2,linestyle='-')
    thetaNax.set_xlabel("$t$ $(sec)$")
    thetaNax.set_ylabel("$\Vert \\tilde{\\theta} \Vert$")
    thetaNax.set_title("Parameter Error Norm")
    thetaNax.grid()
    thetaNplot.savefig(path+"/thetaTildeNorm.pdf")