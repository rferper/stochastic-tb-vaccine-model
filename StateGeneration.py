# Functions for the generation of subsets of states of different levels #
import Parameters as par
N=par.global_N
#%% Compute states given N,i,j,k %%#
def States3(i,j,k):
    S=[]
    for m in range(0,N-i-j+1):
        S.append(''.join(map(str,[i,k,j-k,m,N-i-j-m])))
    return(S)
#%% Compute states given N,i,j %%#
def States2(i,j):
    S=[]
    for k in range(0,j+1):
        for m in range(0,N-i-j+1):
            S.append(''.join(map(str,[i,k,j-k,m,N-i-j-m])))
    return(S)
#%% Compute states given N,i %%#
def States1(i):
    S=[]
    for j in range(0,N-i+1):
        for k in range(0,j+1):
            for m in range(0,N-i-j+1):
                S.append(''.join(map(str,[i,k,j-k,m,N-i-j-m])))
    return(S)
#%% Compute states given N %%#
def States():
    S=[]
    for i in range(0,N+1):
        for j in range(0,N-i+1):
            for k in range(0,j+1):
                for m in range(0,N-i-j+1):
                    S.append(''.join(map(str,[i,k,j-k,m,N-i-j-m])))
    return(S)
#%% Compute transient states given N %%#
def TransientStates():
    S=[]
    for i in range(1,N+1):
        for j in range(0,N-i+1):
            for k in range(0,j+1):
                for m in range(0,N-i-j+1):
                    S.append(''.join(map(str,[i,k,j-k,m,N-i-j-m])))
    return(S)
#%% Compute absorbing states given N %%#
def AbsorbingStates():
    S=[]
    for j in range(0,N+1):
        for k in range(0,j+1):
            for m in range(0,N-j+1):
                S.append(''.join(map(str,[0,k,j-k,m,N-j-m])))
    return(S)
