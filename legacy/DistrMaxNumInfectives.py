#%% Packages %%#
import InfinitesimalGenerator as IG
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix,csr_matrix
from scipy.special import binom
from scipy.sparse import hstack, vstack,bmat
import Parameters as par
import math
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
    return(csc_matrix(InitialDistribution[0:(n-1)] + [sum(InitialDistribution[n:])]))
#%% Computation of the distribution function Fmax given pi_T and Hitting Probabilities %%#
def Fmax():
    print('|--------Computing Fmax---------|')
    F=[]
    P=[]
    i=1
    print('Step',i,'<--> Computing an inverse of size',J1(i),'x',J1(i))
    print('...')
    T =csr_matrix(-inv(IG.Aii(i)))
    t = csc_matrix(IG.A1ii(1)).dot(csc_matrix([[1]]*J1(0)))
    p=T.dot(t)
    F.append(float(pi_T(i).dot(p).toarray()))
    P.append(F[0])
    for i in range(2,N+1):
        print('Step',i,'/',N,'<--> Computing an inverse of size',J1(i),'x',J1(i))
        print('...')
        U21= csr_matrix(hstack([csr_matrix((J1(i),J(i-2))),IG.A1ii(i)]))
        U12= csr_matrix(vstack([csr_matrix((J(i-2),J1(i))),IG.Aii1(i-1)]))
        print('Inverse')
        V22=csr_matrix(inv(csc_matrix(-IG.Aii(i)-U21.dot(T).dot(U12))))
        print('Size of T',T.shape,'. Computing products of this order')
        V12=T.dot(U12.dot(V22))
        V21=V22.dot(U21.dot(T))
        V11=T+T.dot(U12.dot(V21))
        T=csr_matrix(bmat([[V11,V12],[V21,V22]]))
        print('p')
        p=csr_matrix(vstack([p+V12.dot(U21.dot(p)),V22.dot(U21.dot(p))]))
        F.append(float(pi_T(i).dot(p).toarray()))
        P.append(F[i-1]-F[i-2])
    QTA=IG.QTA()
    HP=pi_T(N).dot(T.dot(QTA))
    E=[]
    for r in range(1,N):
        e=math.factorial(r)*pi_T(N).dot(T.power(r).dot(csc_matrix([[1]]*J(N))))
        E.append(float(e.toarray()[0]))
    return F,P,HP.toarray()[0],E

