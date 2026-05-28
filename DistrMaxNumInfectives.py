#%% Packages %%#
import InfinitesimalGenerator as IG
from numpy.linalg import inv
from scipy.special import binom
from numpy import hstack, vstack, bmat, array,zeros
from numpy.linalg import multi_dot,matrix_power
import numpy as np
import Parameters as par
import math
import time
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))    
PRINT=True
#%% Parameters %%#
N=par.global_N

#%% Initial Distribution -- Invasion %%#
InitialDistribution=[0]*(int(binom(N+4,4)-binom(N+3,3)))
InitialDistribution[N-1]=1

#%% Auxiliar functions #%%
def J(i):
    S=0
    for k in range(1,i+1):
        S=S+binom(N-k+3,3)
    return(int(S))
def J1(i):
    return(int(binom(N-i+3,3)))

def pi_T(i):
    n=J(i)
    return(array(InitialDistribution[0:(n-1)] + [sum(InitialDistribution[n:])]))
#%% Computation of the distribution function Fmax given pi_T and Hitting Probabilities %%#
def Fmax():
    print('|--------Computing Fmax---------|')
    F=[]
    P=[]
    i=1
    print('Step',i,'/',N)
    print('...')
    print('Computing an inverse of size',J1(i),'x',J1(i))
    start = time.time()
    T =-inv(IG.Aii(i))
    end = time.time()
    print('Computation time: ')
    timer(start,end)
    print('Computing matrix product of order',T.shape)
    start = time.time()
    t = IG.A1ii(1).sum(axis=1)
    p=T.dot(t)
    end = time.time()
    print('Computation time: ')
    timer(start,end)
    F.append(float(pi_T(i).dot(p)))
    P.append(F[0])
    for i in range(2,N+1):
        print('Step',i,'/',N,)
        print('...')
        U21= hstack([zeros([J1(i),J(i-2)]),IG.A1ii(i)])
        U12= vstack([zeros([J(i-2),J1(i)]),IG.Aii1(i-1)])
        print('Computing an inverse of size',J1(i),'x',J1(i))
        start = time.time()
        V22=-inv(IG.Aii(i)+multi_dot([U21,T,U12]))
        end = time.time()
        print('Computation time: ')
        timer(start,end)
        print('Multiplying matrices of order',T.shape)
        start = time.time()
        V12=multi_dot([T,U12,V22])
        V21=multi_dot([V22,U21,T])
        V11=T+multi_dot([T,U12,V21])
        T=bmat([[V11,V12],[V21,V22]])
        if i==N:
            p=np.concatenate((np.squeeze(np.asarray(p+multi_dot([V12,U21,p]))),np.squeeze(np.asarray(multi_dot([V22,U21,p])),axis=(1,))))
        else:
            p=np.concatenate((np.squeeze(np.asarray(p+multi_dot([V12,U21,p]))),np.squeeze(np.asarray(multi_dot([V22,U21,p])))))
        end = time.time()
        print(psutil.virtual_memory())
        del U21,U12,V22,V12,V21,V11
        print('Computation time: ')
        timer(start,end)
        F.append(float(pi_T(i).dot(p)))
        P.append(F[i-1]-F[i-2])
    QTA=IG.QTA().toarray()
    HP=multi_dot([pi_T(N),T,QTA])
    E=[]
    for r in range(1,3):
        e=math.factorial(r)*pi_T(N).dot(matrix_power(T,r).sum(axis=1))
        E.append(float(e))
    return F,P,np.squeeze(np.asarray(HP)),E

