# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:43:19 2019

@author: rfern
"""
import InfinitesimalGenerator as IG
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.special import binom
from scipy.sparse import hstack, vstack,bmat
import Parameters as par
N=par.global_N


InitialDistribution=[0]*(int(binom(N+4,4)-binom(N+3,3)))
InitialDistribution[0]=1

def J(i):
    S=0
    for k in range(1,i+1):
        S=S+binom(N-k+3,3)
    return(int(S))
def J1(i):
    return(int(binom(N-i+3,3)))

def pi_T(i):
    n=J(i)
    return(csr_matrix(InitialDistribution[0:(n-1)] + [sum(InitialDistribution[n:])]))

def Fmax():
    F=[]
    P=[]
    i=1
    T =-inv(csc_matrix(IG.Aii(i)))
    t = csc_matrix(IG.A1ii(1)).dot(csc_matrix([[1]]*J1(0)))
    p=T.dot(t)
    F.append(float(pi_T(i).dot(p).toarray()))
    P.append(F[0])
    for i in range(2,N+1):
        U21= csc_matrix(hstack([csc_matrix((J1(i),J(i-2))),IG.A1ii(i)]))
        U12= csc_matrix(vstack([csc_matrix((J(i-2),J1(i))),IG.Aii1(i-1)]))
        V22=csc_matrix(inv(-csc_matrix(IG.Aii(i))-U21.dot(T).dot(U12)))
        V12=T.dot(U12.dot(V22))
        V21=V22.dot(U21.dot(T))
        V11=T+T.dot(U12.dot(V21))
        T=bmat([[V11,V12],[V21,V22]])
        p=csc_matrix(vstack([p+V12.dot(U21.dot(p)),V22.dot(U21.dot(p))]))
        F.append(float(pi_T(i).dot(p).toarray()))
        P.append(F[i-1]-F[i-2])
    return F,P

