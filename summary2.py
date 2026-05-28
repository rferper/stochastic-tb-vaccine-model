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
#%% EXPECTED R0 EXACT %%#
start = time.time()
E_R0=R0.M()
end = time.time()
print('|-----------------------------|')
print('Total Computation time of the expected R0: ')
timer(start,end)
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
plt.bar(range(1,N+1), P_max,color='b',width = 0.25)
#plt.bar(np.array(range(1,N+1))+0.3,np.concatenate([pmf,np.array([0]*(len(P_max)-len(pmf)))])
#,color = 'r', width = 0.25)
plt.title('PDF Max # of individuals simultaneously infected:')
plt.ylabel('Probability')
plt.xlabel('Number of Infectives')
#plt.legend(['Model','Simulation'])
plt.show()

#-----------------------------------------------------------#
print('Model: Expected Max # of individuals simultaneously infected:'
      ,sum(np.array(P_max)*range(1,len(P_max)+1)),'±',
      math.sqrt(sum(np.array(P_max)*np.array(range(1,len(P_max)+1))**2)-
                (sum(np.array(P_max)*range(1,len(P_max)+1)))**2))

print('')
print('|--- HITTING PROBABILITIES ---|')
print('')

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
            
print('Expected number of latents and susceptibles at the end of the epidemic')
print(EHP)

print('')
print('|--- LENGTH OF THE OUTBREAK ---|')
print('')

print('Model: Expected length of the outbreak: ',E_max[0],'±',math.sqrt(E_max[1]-E_max[0]**2))

print('')
print('|--- NUMBER OF VACCINATIONS ---|')
print('')

print('Model: Expected number of vaccinations: ',E_V[0],'±',math.sqrt(E_V[1]+E_V[0]-E_V[0]**2))


print('')
print('|--- R0 Exact ---|')
print('')

print('Model: Expected R0 exact: ',E_R0[0],'±',math.sqrt(E_R0[1]+E_R0[0]-E_R0[0]**2))
