#%% Packages %%#
import importlib
from scipy.special import binom
import pandas as pd
import Parameters as par
importlib.reload(par)
import matplotlib.pyplot as plt
import numpy as np
import time
N=par.global_N
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
import StateGeneration as st  # Functions for the generation of subsets of states of different levels
print()
print('Total number of states: ',binom(N+4,4))
print()

if(binom(N+4,4)>1000):
    PRINT=False
print('Transient States: ',binom(N+4,4)-binom(N+3,3))
if(PRINT==True):
    print(st.TransientStates())
    print()

print('Absorbing States: ',binom(N+3,3))
if(PRINT==True):
    print(st.AbsorbingStates())


#%% INFINITESIMAL GENERATOR %##
import InfinitesimalGenerator as IG
importlib.reload(IG)
    
start = time.time()
print('State transitions between transient states')
print('QTT matrix. Rows: ',binom(N+4,4)-binom(N+3,3) ,'Columns: ',binom(N+4,4)-binom(N+3,3))
QTT=IG.QTT()
print(type(QTT))
if(PRINT==True):
    printSparseMatrix(QTT,st.TransientStates(),st.TransientStates())

print('---------------------------------------------')
print('State transitions from transient states to absorbing states')
print('QTA matrix. Rows: ',binom(N+4,4)-binom(N+3,3) ,'Columns: ',binom(N+3,3))
QTA=IG.QTA()
print(type(QTA))
if(PRINT==True):
    printSparseMatrix(QTA,st.TransientStates(),st.AbsorbingStates())
end = time.time()
print()
print('Computation time: ')
timer(start,end) # approx 12 min with N=100

start = time.time()
print('State transitions')
print('Q matrix. Rows: ',binom(N+4,4) ,'Columns: ',binom(N+4,4))
Q=IG.Q()
print(type(Q))
if(PRINT==True):
    printSparseMatrix(Q,st.States(),st.States())
print('Computation time: ')
end = time.time()
timer(start,end) 

# Comprobation
#Q.sum(axis=1) # Las entradas no son exactamente 0
#%% DISTRIBUTION OF THE MAX NUMBER OF INDIVIDUALS SIMULTANEOUSLY INFECTED %%#
import DistrMaxNumInfectives as DMI
importlib.reload(DMI)

start = time.time()
F,P,HP,E=DMI.Fmax()
end = time.time()
print('Computation time: ')
timer(start,end)

#------------------------- PLOTS ----------------------------------#
print('PDF Max # of individuals simultaneously infected:')
#print(P)
plt.figure()
plt.bar(range(1,N+1), P)
plt.title('PDF Max # of individuals simultaneously infected:')
plt.ylabel('Probability')
plt.xlabel('Number of Infectives')
print('CDF Max # of individuals simultaneously infected:')
#print(F)
plt.figure()
data = range(0,N+2)
y = F
yn = np.insert(F, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='red', zorder=1)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='red',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='red', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='red')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])

plt.title('CDF Max # of individuals simultaneously infected:')
plt.ylabel('Probability')
plt.xlabel('Number of Infectives')
#-----------------------------------------------------------#
#%% HITTING PROBABILITIES %%#
print('Hitting Probabilities:')
#print(HP)
plt.figure()
plt.bar(range(1,int(binom(N+3,3))+1),HP)
plt.title('Hitting Probabilities')
plt.ylabel('Probability')
plt.xlabel('Absorbing States')
s=0
for j in range(0,N):
        s=s+(N-j+1)*(j+1)
        plt.axvline(x=s,color='k', linestyle='--')
#%% LENGTH OF THE OUTBREAK%%#
print('Expected length of the outbreak: ',E[0],'±',E[1]-E[0]**2)
#%% NUMBER OF VACCINATIONS %%#
import NumVaccinations as NV
importlib.reload(NV)
start = time.time()
F,P=NV.FV(0.9999)
end = time.time()
print('Computation time: ')
timer(start,end)
#-----------------------------------------------------------#
print('PDF # of vaccinations during an outbreak:')
#print(P)
plt.figure()
plt.bar(range(0,len(P)), P)
plt.title('PDF # of vaccinations during an outbreak:')
plt.ylabel('Probability')
plt.xlabel('Number of Vaccinations')
print('CDF # of vaccinations during an outbreak:')
#print(F)
plt.figure()
data = range(0,len(P)+2)
y = F
yn = np.insert(F, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='red', zorder=1)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='red',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='red', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='red')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])

plt.title('CDF # of vaccinations during an outbreak:')
plt.ylabel('Probability')
plt.xlabel('Number of Vaccinations')
#-----------------------------------------------------------# 
#%% EXPECTED NUMBER OF VACCINATIONS %%#     
start = time.time()
E=NV.M()
end = time.time()
print('Computation time: ')
timer(start,end)
print('Expected number of vaccinations: ',E[0],'±',E[1]+E[0]-E[0]**2)
print('Approximated number of vaccinations:',sum(np.array(P)*np.array(range(0,len(P)))),'±',sum(np.array(P)*(np.array(range(0,len(P)))**2)))