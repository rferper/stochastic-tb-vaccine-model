from BlockDiagonalSparseMatrices import SupDiagBlockMatrix,TriBlockMatrix,SubDiagBlockMatrix
import numpy as np
from scipy.sparse import diags,vstack,block_diag,hstack,csc_matrix
from scipy.special import binom
import Parameters as par

N=par.global_N
gamma=par.global_gamma 
q = par.global_q 
C = par.global_C 
eta=par.global_eta
alpha_u=par.global_alpha_u 
alpha_v=par.global_alpha_v 
p_u=par.global_p_u 
p_v=par.global_p_v 
r_u=par.global_r_u
r_v=par.global_r_v
delta_TB = par.global_delta_TB
delta_R = par.global_delta_R 
delta_D=par.global_delta_D 
xi = par.global_xi 
sigma_l_u=par.global_sigma_l_u 
sigma_s_u=par.global_sigma_s_u

#%% Aii %%#
#%% #%%
def Ekk1(i,j,k):
    return(vstack([csc_matrix((1,N-i-j)),
             diags(np.array(range(1,N-i-j+1))*alpha_u*(1-p_u)*i,shape=(N-i-j, N-i-j),format='csc')],format='csc'))
#%% %%#
def Ekk(i,j,k):
    return(vstack([diags(np.array(list(reversed(range(1,N-i-j+1))))*alpha_v*(1-p_v)*i,shape=(N-i-j, N-i-j),format='csc'),
                   csc_matrix((1,N-i-j))],format='csc'))
#%% %%#  
def Fkk(i,j,k):
    if(i < N):
        return(diags([np.array(range(1,N-i-j+1))*delta_D*eta(q,C),
                    -(np.array(range(0,N-i-j+1))*(alpha_u*i+delta_D*eta(q,C))+
                    (np.array(list(reversed(range(0,N-i-j+1))))*(gamma+alpha_v*i+delta_D*(1-eta(q,C)))
                    +np.array([1]*(N-i-j+1))*(delta_D*j+r_v(i)*(j-k)+r_u(i)*k+(delta_D+delta_R+delta_TB)*i))),
                   np.array(list(reversed(range(1,N-i-j+1))))*(gamma+delta_D*(1-eta(q,C)))]
                  ,[-1,0,1],format='csc'))
    else:
        return(diags([np.array(range(1,N-i-j+1))*delta_D*eta(q,C),
                    -(np.array([1]*(N-i-j+1))*((delta_D+delta_R+delta_TB)*i)),
                   np.array(list(reversed(range(1,N-i-j+1))))*(gamma+delta_D*(1-eta(q,C)))]
                  ,[-1,0,1],format='csc'))
#%% %%#
def Gkk(i,j,k):
    return(diags([np.array([j-k]*(N-i-j+1))*delta_D*eta(q,C),
           np.array([j-k]*(N-i-j+1))*delta_D*(1-eta(q,C))]
          ,[0,1],
          shape=(N-i-j+1, N-i-j+2),format='csc'))
#%% %%#
def G1kk(i,j,k):
    return(diags([np.array([k]*(N-i-j+1))*delta_D*eta(q,C),
           np.array([k]*(N-i-j+1))*delta_D*(1-eta(q,C))]
          ,[0,1],
          shape=(N-i-j+1, N-i-j+2),format='csc'))
#%% %%#
def Bjj1(i,j):
    Diagonal=[]
    SuperDiagonal=[]

    for k in range(0,j+1):
        Diagonal.append(Ekk(i,j,k))
        SuperDiagonal.append(Ekk1(i,j,k))
    
    Bjj1=SupDiagBlockMatrix(Diagonal,SuperDiagonal)
    return(Bjj1)
#%% %%#
def Bjj(i,j):
    Diagonal=[]
    for k in range(0,j+1):
        Diagonal.append(Fkk(i,j,k))
    Bjj=block_diag(Diagonal,dtype='float64',format='csc')
    return(Bjj)
#%% %%#
def B1jj(i,j):
    Diagonal=[]
    SubDiagonal=[]
    for k in range(0,j+1):
        if k<j:
            Diagonal.append(Gkk(i,j,k))
        if k>0:
            SubDiagonal.append(G1kk(i,j,k))
    B1jj=SubDiagBlockMatrix(SubDiagonal,Diagonal)
    return(B1jj)
#%% %%#
def Aii(i):
    SuperDiagonal=[]
    Diagonal=[]
    SubDiagonal=[]
    for j in range(0,N-i+1):
        if j<N-i:
            SuperDiagonal.append(Bjj1(i,j))
        Diagonal.append(Bjj(i,j))
        if j>0:
            SubDiagonal.append(B1jj(i,j))
    Aii=TriBlockMatrix(SubDiagonal,Diagonal,SuperDiagonal)
    return(Aii.toarray())
#%% Ai(i+1) %%#
#%% %%#
def Lkk(i,j,k):
    return(diags([np.array(range(1,N-i-j+1))*alpha_u*p_u*i,
           np.array(list(reversed(range(1,N-i-j+1))))*alpha_v*p_v*i]
          ,[-1,0],
            shape=(N-i-j+1,N-i-j),format='csc'))
def Mkk(i,j,k):
    return(diags([np.array([j-k]*(N-i-j+1))*r_v(i)]
          ,[0],format='csc'))
def M1kk(i,j,k):
     return(diags([np.array([k]*(N-i-j+1))*r_u(i)]
          ,[0],format='csc'))   
#%% %%#
def Djj(i,j):
    Diagonal=[]
    for k in range(0,j+1):
        Diagonal.append(Lkk(i,j,k))
    Djj=block_diag(Diagonal,dtype='float64',format='csc')
    return(Djj)
def D1jj(i,j):
    Diagonal=[]
    SubDiagonal=[]
    for k in range(0,j+1):
        if k<j:
            Diagonal.append(Mkk(i,j,k))
        if k>0:
            SubDiagonal.append(M1kk(i,j,k))
    D1jj=SubDiagBlockMatrix(SubDiagonal,Diagonal)
    return(D1jj)
#%% %%#
def Aii1(i):
    Diagonal=[]
    SubDiagonal=[]
    for j in range(0,N-i+1):
        if j < N-i:
            Diagonal.append(Djj(i,j))
        if j>0:
            SubDiagonal.append(D1jj(i,j))
    Aii1=SubDiagBlockMatrix(SubDiagonal,Diagonal)
    return(Aii1.toarray())
#%% Ai(i-1) %%#
#%% %%#
def Jkk(i,j,k):
    return(diags([np.array([i]*(N-i-j+1))*(delta_R*(1-xi)*(1-sigma_s_u)+(delta_TB+delta_D)*eta(q,C)),
       np.array([i]*(N-i-j+1))*(delta_R*(1-xi)*(sigma_s_u)+(delta_TB+delta_D)*(1-eta(q,C)))]
      ,[0,1],
      shape=(N-i-j+1, N-i-j+2),format='csc'))
def Hkk(i,j,k):
    return(diags([np.array([i]*(N-i-j+1))*(delta_R*xi*(1-sigma_l_u))]
          ,[0],format='csc'))  
def Hkk1(i,j,k):
    return(diags([np.array([i]*(N-i-j+1))*(delta_R*xi*(sigma_l_u))]
          ,[0],format='csc'))      
    
#%% %%#
    
def Cjj1(i,j):
    Diagonal=[]
    SuperDiagonal=[]

    for k in range(0,j+1):
        Diagonal.append(Hkk(i,j,k))
        SuperDiagonal.append(Hkk1(i,j,k))
    Cjj1=SupDiagBlockMatrix(Diagonal,SuperDiagonal)
    return(Cjj1)

def Cjj(i,j):
    Diagonal=[]
    for k in range(0,j+1):
        Diagonal.append(Jkk(i,j,k))
    Cjj=block_diag(Diagonal,dtype='float64',format='csc')
    return(Cjj)    
#%% %%#
def A1ii(i):
    Diagonal=[]
    SuperDiagonal=[]
    for j in range(0,N-i+1):
        Diagonal.append(Cjj(i,j))
        SuperDiagonal.append(Cjj1(i,j))
    A1ii=SupDiagBlockMatrix(Diagonal,SuperDiagonal)
    return(A1ii.toarray())    
#%% Infinitesimal Generator %%#
#%% %%#
def QTT():
    SuperDiagonal=[]
    Diagonal=[]
    SubDiagonal=[]
    for i in range(1,N+1):
        if i<N:
            SuperDiagonal.append(Aii1(i))
        Diagonal.append(Aii(i))
        if i>1:
            SubDiagonal.append(A1ii(i))
    QTT=TriBlockMatrix(SubDiagonal,Diagonal,SuperDiagonal)
    return(QTT)
    
def QTA():
    A=A1ii(1)
    Zeros=csc_matrix((int(binom(N+4,4)-binom(N+3,3)-binom(N+2,3)),A.shape[1]))
    QTA=vstack([A,Zeros],format='csc')
    return(QTA)    
#%% %%#

def Q():
    Q1=QTA()
    Q2=QTT()
    Zeros=csc_matrix((int(binom(N+3,3)), int(binom(N+4,4))))
    Q=vstack([Zeros,hstack([Q1,Q2],format='csc')],format='csc')
    return Q
    
            