#%% Packages %%#
import importlib
from scipy.special import binom
import pandas as pd
import Parameters as par
import matplotlib.pyplot as plt
import numpy as np
import time
import math
#%% AUXILIAR FUNCTIONS %%#
def printSparseMatrix(M,rownames,colnames):
    df = pd.DataFrame(M.toarray(), columns=colnames, index=rownames)
    print(df)
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))    
PRINT=True
#%% DESCRIPTION STATES%%##
N=par.global_N
print('|------------ Number of states ------------|')
print('Total number of states: ',binom(N+4,4))
print('Transient States: ',binom(N+4,4)-binom(N+3,3))
print('Absorbing States: ',binom(N+3,3))
print('|------------ Infinitesimal Generator ------------|')
print('Q: ',binom(N+4,4),'x',binom(N+4,4))
print('QTT: ',binom(N+4,4)-binom(N+3,3),'x',binom(N+4,4)-binom(N+3,3))
print('QTA: ',binom(N+4,4)-binom(N+3,3),'x',binom(N+3,3))
#%% DISTRIBUTION OF THE MAX NUMBER OF INDIVIDUALS SIMULTANEOUSLY INFECTED %%#
import DistrMaxNumInfectives as DMI
importlib.reload(DMI)
start = time.time()
F_max,P_max,HP,E_max=DMI.Fmax()
end = time.time()
print('|-----------------------------|')
print('Total computation time of the PDF Max # of individuals simultaneously infected : ')
timer(start,end)
#%% NUMBER OF VACCINATIONS %%#
import NumVaccinations as NV
importlib.reload(NV)
start = time.time()
F_V,P_V=NV.FV(0.99)
end = time.time()
print('|-----------------------------|')
print('Total Computation time of the PDF # of vaccinations: ')
timer(start,end)   
#%% EXPECTED NUMBER OF VACCINATIONS %%#     
start = time.time()
E_V=NV.M()
end = time.time()
print('|-----------------------------|')
print('Total Computation time of the expected # of vaccinations ')
timer(start,end)
#%% DISTRIBUTION R0 EXACT %%#
import R0
importlib.reload(R0)
start = time.time()
F_R0,P_R0=R0.FR0(0.99)
end = time.time()
print('|-----------------------------|')
print('Total Computation time of the PDF of R0: ')
timer(start,end)
#%% EXPECTED R0 EXACT %%#
start = time.time()
E_R0=R0.M()
end = time.time()
print('|-----------------------------|')
print('Total Computation time of the expected R0: ')
timer(start,end)
plt.figure()
plt.plot(range(0,N),list(reversed(E_R0)))
plt.ylabel('R0p')
plt.xlabel('p')
plt.axis([-0.1, 22.1, 0, 3.5])
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.yticks([0,0.5,1,1.5,2,2.5,3,3.5])
#plt.legend(['Model','Simulation'])
plt.show()
#%% Simulación %%#
#import Simulaciones as SIM
#imp.reload(SIM)
#import random

#random.seed(2019)

#Xmax=[]
#V=[]
#Tmax=[]
#R0_sim=[]
#start = time.time()
#for i in range(0,40000):
#    t,x,v=SIM.Sim_Trayectoria()
#    r=SIM.Sim_Trayectoria_R0()
#    Tmax.append(t)
#    Xmax.append(x)
#    V.append(v)
#    R0_sim.append(r)
#end = time.time()
#print('|-----------------------------|')
#print('Total Computation time of the simulation: ')
#timer(start,end)
#%% RESUMEN DE LOS RESULTADOS #%%
print('')
print('|------------------- SUMMARY OF THE RESULTS ---------------------------|')
print('')

print('')
print('|--- MAX # OF INDIVIDUALS SIMULTANEOUSLY INFECTED ---|')
print('')

#val, cnt = np.unique(Xmax, return_counts=True)
#pmf = cnt / len(Xmax)
plt.figure()
plt.bar(range(1,N+1), P_max,color='b',width = 0.5)
#plt.bar(np.array(range(1,N+1))+0.3,np.concatenate([pmf,np.array([0]*(len(P_max)-len(pmf)))])
#,color = 'r', width = 0.25)
#plt.title('PDF Max # of individuals simultaneously infected:')
plt.ylabel('Probability')
plt.xlabel('Number of Infectives')
plt.axis([0.5, 20.5, 0, 0.3])
plt.xticks([2,4,6,8,10,12,14,16,18,20])
plt.yticks([0,0.05,0.1,0.15,0.2,0.25,0.3])
#plt.legend(['Model','Simulation'])
plt.show()

#-----------------------------------------------------------#
print('Model: Expected Max # of individuals simultaneously infected:'
      ,round(sum(np.array(P_max)*range(1,len(P_max)+1)),5),'±',
      round(math.sqrt(sum(np.array(P_max)*np.array(range(1,len(P_max)+1))**2)-
                (sum(np.array(P_max)*range(1,len(P_max)+1)))**2),5))
#print('Simulation: Expected Max # of individuals simultaneously infected:',np.mean(Xmax),'±',np.std(Xmax))

print('')
print('|--- HITTING PROBABILITIES ---|')
print('')

plt.figure()
plt.bar(range(1,int(binom(N+3,3))+1),HP,color='b',width=3)
plt.title('Hitting Probabilities')
plt.ylabel('Probability')
plt.xlabel('Absorbing States')
s=0
for j in range(0,N):
        s=s+(N-j+1)*(j+1)
        plt.axvline(x=s,color='k', linestyle='--',linewidth=0.8)
plt.show()

EHP=[0]*4
s=0
for j in range(0,N+1):
    for k in range(0,j+1):
        for m in range(0,N-j+1):
            EHP[0]=EHP[0]+k*HP[s]
            EHP[1]=EHP[1]+(j-k)*HP[s]
            EHP[2]=EHP[2]+m*HP[s]
            EHP[3]=EHP[3]+(N-j-m)*HP[s]
            s=s+1
HPlu=[0]*(N+1)
HPlv=[0]*(N+1)
HPsu=[0]*(N+1)
HPsv=[0]*(N+1)
s=0
for j in range(0,N+1):
    for k in range(0,j+1):
        for m in range(0,N-j+1):
            HPlu[k]=HPlu[k]+HP[s]
            HPlv[j-k]=HPlv[j-k]+HP[s]
            HPsu[m]=HPsu[m]+HP[s]
            HPsv[N-j-m]=HPsv[N-j-m]+HP[s]
            s=s+1
            
plt.plot(range(0,len(HPlu)), HPlu,color='green')
plt.plot(np.array(range(0,len(HPlu))), HPlv,color='lime')            
plt.plot(np.array(range(0,len(HPsu))), HPsu,color='mediumblue')
plt.plot(np.array(range(0,len(HPsu))), HPsv,color='deepskyblue')
plt.axis([-0.5, 20.5, 0, 1])
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.yticks([0,0.2,0.4,0.6,0.8,1])
plt.ylabel('Probability')
plt.legend(['Lu','Lv','Su','Sv']) 


print('Expected number of latents and susceptibles at the end of the epidemic')
print(round(EHP[0],5),round(EHP[1],5),round(EHP[2],5),round(EHP[3],5))

print('')
print('|--- LENGTH OF THE OUTBREAK ---|')
print('')

print('Model: Expected length of the outbreak: ',round(E_max[0],5),'±',round(math.sqrt(E_max[1]-E_max[0]**2),5))
#print('Simulation: Expected length of the outbreak: ',np.mean(Tmax),'±',np.std(Tmax))

print('')
print('|--- NUMBER OF VACCINATIONS ---|')
print('')

#-----------------------------------------------------------#
plt.figure()
#val, cnt = np.unique(V, return_counts=True)
#pmf = cnt / len(V)
plt.bar(range(0,len(P_V)), P_V,color='b', width = 3)
#plt.bar(np.array(range(0,len(P_V)))+0.3,pmf[0:len(P_V)],color = 'r', width = 0.25)
#plt.title('PDF # of vaccinations during an outbreak:')
plt.axis([-3, 410, 0, 0.12])
plt.xticks([0,50,100,150,200,250,300,350,400])
plt.yticks([0,0.02,0.04,0.06,0.08,0.1,0.12])
plt.ylabel('Probability')
plt.xlabel('Number of Vaccinations')
#plt.legend(['Model','Simulation'])
print('Model: Expected number of vaccinations: ',round(E_V[0],5),'±',round(math.sqrt(E_V[1]+E_V[0]-E_V[0]**2),5))
#print('Simulation: Expected number of vaccinations:',np.mean(V),'±',np.std(V))
plt.show()



print('')
print('|--- R0 Exact ---|')
print('')

#-----------------------------------------------------------#
plt.figure()
#val, cnt = np.unique(R0_sim, return_counts=True)
#pmf = cnt / len(R0_sim)
plt.bar(range(0,len(P_R0)), P_R0,color='b',width = 0.25)
#plt.bar(np.array(range(0,len(P_R0)))+0.3,pmf[0:len(P_R0)],color = 'r', width = 0.25)
plt.title('PDF of R0:')
plt.ylabel('Probability')
plt.xlabel('Number of infectives due to the marked individual')
#plt.legend(['Model','Simulation'])
plt.show()

print('Model: Expected R0 exact: ',round(E_R0[0],5),'±',round(math.sqrt(E_R0[1]+E_R0[0]-E_R0[0]**2),5))
#print('Simulation: Expected R0 exact: ',np.mean(R0_sim),'±',np.std(R0_sim))