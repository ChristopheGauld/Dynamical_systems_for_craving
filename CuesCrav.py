import matplotlib.pyplot as plt
import numpy as np
from random import random
from random import seed
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})



# cycle :
Par={'Smax':10, 'Rs':1, 'lambdas':0.1, 'taux':14, 'P': 10, 'Rb':1.2, 'lambdab': 0.05, 'L':1.01*0.75, 'tauy':14, 'S':10, 'alpha':0.5, 'beta':0.5, 'tauz':1 }

Luse=[]
Lcrav=[]
Lcues=[]
Lsim= 200000
cutT=int(Lsim/1.2)

Nsim = 100
for nsim in range(Nsim):
    print(nsim)
    seed(nsim)

    
    Li=[o for o in range(Lsim)]
    dt=0.01
    Lt=[dt*i for i in Li]
    y= 0.1*(random()+0.5)
    x= 0.1*(random()+0.5) 
    z= 0.1*(random()+0.5)
    f=0.0
    Ly=[]
    Lx=[]
    Lz=[]
    Lf=[]



    Smax = Par['Smax']
    Rs = Par['Rs']
    lambdas = Par['lambdas']
    taux = Par['taux']
    P = Par['P']
    Rb = Par['Rb']
    lambdab = Par['lambdab']
    L = Par['L']*(random()/2 + 0.75)
    tauy = Par['tauy']
    S = Par['S']
    alpha = Par['alpha']
    beta = Par['beta']
    tauz = Par['tauz']

    Lzi=[np.sin(6e-4*i) for i in Li]
    Lzi=[i*5  if i>0 else 0 for i in Lzi]
    #Lzi=[0  if i>0 else 0 for i in Lzi]
   # plt.plot(Lzi)
   # plt.show()
    for i in Li:

        #b=random()
        a=(random()-0.5)
    # La.append(3.4+a)
        #print(a)
        dx=(Smax/(1+np.exp((Rs-y)/lambdas))-x)/taux
        dy=(P/(1+np.exp((Rb-y)/lambdab))+L-y*x-z)/tauy
        dz= -Lzi[i] + (S*(alpha*x+beta*y)*(a)-z)/tauz
        #df=(y-1.0*f)/720
        y=y+dy*dt#+a
        x=x+dx*dt
        z=z+dz*dt
        #f=f+df*dt

    # changes in parameters during simulation
    # pharmacological intervention
    # if i>547500:
    #    P = 7.



        Ly.append(y)
        Lx.append(x)
        Lz.append(z)
        #Lf.append(f)
    Luse.append(Lx[cutT::])
    Lcrav.append(Ly[cutT::])
    Lcues.append(Lz[cutT::])
data = [Luse, Lcrav, Lcues]
np.save('CuesCrav.npy', data)    



fig, axs = plt.subplots(4)
axs[0].plot(Lt[cutT::],Lx[cutT::], 'b')
axs[0].axes.xaxis.set_ticklabels([])
#axs[0].tick_params(axis='both', left='on', top='off', right='off', bottom='off', labelleft='on', labeltop='off', labelright='off', labelbottom='off')
axs[0].set_ylabel(r'$\bf{x}$') #+'\n"symptom rate"')
axs[1].plot(Lt[cutT::],Ly[cutT::], 'magenta')
axs[1].axes.xaxis.set_ticklabels([])
#axs[1].text(5400, 1.4, r'$\bf{|}$', transform=axs[1].transData)
#axs[1].text(5401, 1.4, r'$\bf{-}$', transform=axs[1].transData)
#axs[1].text(5402, 1.4, r'$\bf{>}$', transform=axs[1].transData)
#axs[1].tick_params(axis='both', left='on', top='off', right='off', bottom='off', labelleft='on', labeltop='off', labelright='off', labelbottom='off')
axs[1].set_ylabel(r'$\bf{y}$') #+'\n"internal potentiation"')
#axs[1].text(1500, 1.2, '|'+r"intervention on $P$", transform=axs[1].transData)
axs[2].plot(Lt[cutT::],Lz[cutT::], 'r')
axs[2].set_ylabel(r'$\bf{z}$') #+'\n"perceived environment"')

#axs[2].tick_params(axis='both', left='on', top='off', right='off', bottom='off', labelleft='on', labeltop='off', labelright='off', labelbottom='off')
axs[2].set_ylim(-5, 5)
axs[2].axes.xaxis.set_ticklabels([])
#axs[3].plot(Lt[cutT::],Lf[cutT::], 'g')
axs[3].set_ylabel(r'$\bf{f}$')
axs[3].set_xlabel('Time (days)')
axs[3].set_ylim(0,1.5)
fig.align_ylabels()
#plt.tight_layout()

plt.show()
