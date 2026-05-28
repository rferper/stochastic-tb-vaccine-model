#%% Packages
import numpy as np
import matplotlib.pyplot as plt
from BlockDiagonalSparseMatrices import SupDiagBlockMatrix,TriBlockMatrix,SubDiagBlockMatrix
from scipy.sparse import diags,coo_matrix,vstack,block_diag
from numpy.linalg import inv
from numpy.linalg import multi_dot
#%% Parameters

N=20 # Size of the population
delta_D=(3/5) # per capita death rate from causes beyond TB
gamma=1/5 # per capita duration rate of the vaccine protection
q = 0.1# take (fraction of vaccinated individuals in whom the vaccione induces
      # some degree of protection)
C = 0.8 # Coverage level (fraction of individuals that are vaccinated)
def eta(q,C): #Probability that a newborn will be vaccinated succesfuly (with some degree of protection)
   return q*C
alpha_u=5.2 # per capita infectious contact rate on unvaccinated susceptible individuals
alpha_v=5.2 # per capita infectious contact rate on vaccinated susceptible individuals
p_u=0.15 # probability that a non-vaccinated susceptible becomes infective
p_v=0 # probability that a vaccinated susceptible becomes infective
a_u = 1.5*10**(-4) # per capita reactivation rate of a non-vaccinated latenly infected individual
b_u = 0.35*alpha_u # per capita reinfection rate of a non-vaccinated latenly infected individual
def r_u(I): #per capita rate of reactivation and reinfection of a non-vaccinated latently infected individual
   return a_u+b_u*I
a_v = 0 # per capita reactivation rate of a vaccinated latenly infected individual
b_v = 0 # per capita reinfection rate of a vaccinated latenly infected individual
def r_v(I): #per capita rate of reactivation and reinfection of a vaccinated latently infected individual
   return a_v+b_v*I
delta_TB = 0.2 # per capita rate of death due to tuberculosis
delta_R = 4/3 # per capita effective treatment rate
xi = 1 # probability that a recovered individual is considered as latent
sigma_l_u=0.5 # probability that a recovered latent individual is considered as an unvaccinated individual
sigma_s_u=1 # probability that a recovered susceptible individual is considered as an unvaccinated individual
def M(q):
    def Ekk1(i,j,k):
        return(vstack([coo_matrix((1,N-i-j)),
                 diags(np.array(range(1,N-i-j+1))*alpha_u*(1-p_u)*i,shape=(N-i-j, N-i-j))]))
    def Ekk(i,j,k):
        return(vstack([diags(np.array(list(reversed(range(1,N-i-j+1))))*alpha_v*(1-p_v)*i,shape=(N-i-j, N-i-j)),
                       coo_matrix((1,N-i-j))]))  
    def Fkk(i,j,k):
        if(i < N):
            return(diags([np.array(range(1,N-i-j+1))*delta_D*eta(q,C),
                        -(np.array(range(0,N-i-j+1))*(alpha_u*i+delta_D*eta(q,C))+
                        (np.array(list(reversed(range(0,N-i-j+1))))*(gamma+alpha_v*i+delta_D*(1-eta(q,C)))
                        +np.array([1]*(N-i-j+1))*(delta_D*j+r_v(i)*(j-k)+r_u(i)*k+(delta_D+delta_R+delta_TB)*i))),
                       np.array(list(reversed(range(1,N-i-j+1))))*(gamma+delta_D*(1-eta(q,C)))]
                      ,[-1,0,1]))
        else:
            return(diags([np.array(range(1,N-i-j+1))*delta_D*eta(q,C),
                        -(np.array([1]*(N-i-j+1))*((delta_D+delta_R+delta_TB)*i)),
                       np.array(list(reversed(range(1,N-i-j+1))))*(gamma+delta_D*(1-eta(q,C)))]
                      ,[-1,0,1]))
    def Fkk_v(i,j,k):
        if(i < N):
            return(diags([np.array(range(1,N-i-j+1))*delta_D*eta(q,C),
                        (np.array(list(reversed(range(0,N-i-j+1))))*(delta_D*eta(q,C)))]
                      ,[-1,0]))
        else:
            return(diags([np.array(range(1,N-i-j+1))*delta_D*eta(q,C),
                        np.array([0]*(N-i-j+1))]
                      ,[-1,0]))
    def Fkk_star(i,j,k):
        if(i < N):
            return(diags([-(np.array(range(0,N-i-j+1))*(alpha_u*i+delta_D*eta(q,C))+
                        (np.array(list(reversed(range(0,N-i-j+1))))*(gamma+alpha_v*i+delta_D*(1-eta(q,C)))
                        +np.array([1]*(N-i-j+1))*(delta_D*j+r_v(i)*(j-k)+r_u(i)*k+(delta_D+delta_R+delta_TB)*i))
                        +(np.array(list(reversed(range(0,N-i-j+1))))*(delta_D*eta(q,C))))
                        ]
                      ,[0]))
        else:
            return(diags([np.array(range(1,N-i-j+1))*delta_D*eta(q,C),
                        -(np.array([1]*(N-i-j+1))*((delta_D+delta_R+delta_TB)*i)),
                       np.array(list(reversed(range(1,N-i-j+1))))*(gamma+delta_D*(1-eta(q,C)))]
                      ,[-1,0,1]))
    def Gkk(i,j,k):
        return(diags([np.array([j-k]*(N-i-j+1))*delta_D*eta(q,C),
               np.array([j-k]*(N-i-j+1))*delta_D*(1-eta(q,C))]
              ,[0,1],
              shape=(N-i-j+1, N-i-j+2)))
    def Gkk_v(i,j,k):
        return(diags([np.array([j-k]*(N-i-j+1))*delta_D*eta(q,C)]
              ,[0],
              shape=(N-i-j+1, N-i-j+2)))    
    def G1kk(i,j,k):
        return(diags([np.array([k]*(N-i-j+1))*delta_D*eta(q,C),
               np.array([k]*(N-i-j+1))*delta_D*(1-eta(q,C))]
              ,[0,1],
              shape=(N-i-j+1, N-i-j+2)))
    def G1kk_v(i,j,k):
        return(diags([np.array([k]*(N-i-j+1))*delta_D*eta(q,C)]
              ,[0],
              shape=(N-i-j+1, N-i-j+2)))
    def Bjj1(i,j):
        Diagonal=[]
        SuperDiagonal=[]
    
        for k in range(0,j+1):
            Diagonal.append(Ekk(i,j,k))
            SuperDiagonal.append(Ekk1(i,j,k))
        
        Bjj1=SupDiagBlockMatrix(Diagonal,SuperDiagonal)
        return(Bjj1)
    def Bjj(i,j):
        Diagonal=[]
        for k in range(0,j+1):
            Diagonal.append(Fkk(i,j,k))
        Bjj=block_diag(Diagonal,dtype='float64')
        return(Bjj)
    def Bjj_v(i,j):
        Diagonal=[]
        for k in range(0,j+1):
            Diagonal.append(Fkk_v(i,j,k))
        Bjj_v=block_diag(Diagonal,dtype='float64')
        return(Bjj_v)
    def Bjj_star(i,j):
        Diagonal=[]
        for k in range(0,j+1):
            Diagonal.append(Fkk_star(i,j,k))
        Bjj_star=block_diag(Diagonal,dtype='float64')
        return(Bjj_star)
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
    def B1jj_v(i,j):
        Diagonal=[]
        SubDiagonal=[]
        for k in range(0,j+1):
            if k<j:
                Diagonal.append(Gkk_v(i,j,k))
            if k>0:
                SubDiagonal.append(G1kk_v(i,j,k))
        B1jj_v=SubDiagBlockMatrix(SubDiagonal,Diagonal)
        return(B1jj_v)
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
    def Aii_v(i):
        SuperDiagonal=[]
        Diagonal=[]
        SubDiagonal=[]
        for j in range(0,N-i+1):
            if j<N-i:
                SuperDiagonal.append(coo_matrix(((j+1)*(N-i-j+1),(j+2)*(N-i-j))))
            Diagonal.append(Bjj_v(i,j))
            if j>0:
                SubDiagonal.append(B1jj_v(i,j))
        Aii_v=TriBlockMatrix(SubDiagonal,Diagonal,SuperDiagonal)
        return(Aii_v.toarray())
    def Aii_star(i):
        SuperDiagonal=[]
        Diagonal=[]
        SubDiagonal=[]
        for j in range(0,N-i+1):
            if j<N-i:
                SuperDiagonal.append(coo_matrix(((j+1)*(N-i-j+1),(j+2)*(N-i-j))))
            Diagonal.append(Bjj_star(i,j))
            if j>0:
                SubDiagonal.append(coo_matrix(((j+1)*(N-i-j+1),j*(N-i-j+2))))
        Aii_star=TriBlockMatrix(SubDiagonal,Diagonal,SuperDiagonal)
        return(Aii_star.toarray())    
    def Aii_u(i):
        return(Aii(i)-Aii_star(i)-Aii_v(i))
    def Lkk(i,j,k):
        return(diags([np.array(range(1,N-i-j+1))*alpha_u*p_u*i,
               np.array(list(reversed(range(1,N-i-j+1))))*alpha_v*p_v*i]
              ,[-1,0],
                shape=(N-i-j+1,N-i-j)))
    def Mkk(i,j,k):
        return(diags([np.array([j-k]*(N-i-j+1))*r_v(i)]
              ,[0]))
    def M1kk(i,j,k):
         return(diags([np.array([k]*(N-i-j+1))*r_u(i)]
              ,[0]))   
    def Djj(i,j):
        Diagonal=[]
        for k in range(0,j+1):
            Diagonal.append(Lkk(i,j,k))
        Djj=block_diag(Diagonal,dtype='float64')
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
    def Jkk(i,j,k):
        return(diags([np.array([i]*(N-i-j+1))*(delta_R*(1-xi)*(1-sigma_s_u)+(delta_TB+delta_D)*eta(q,C)),
           np.array([i]*(N-i-j+1))*(delta_R*(1-xi)*(sigma_s_u)+(delta_TB+delta_D)*(1-eta(q,C)))]
          ,[0,1],
          shape=(N-i-j+1, N-i-j+2)))
    def Jkk_v(i,j,k):
        return(diags([np.array([i]*(N-i-j+1))*((delta_TB+delta_D)*eta(q,C))]
          ,[0],
          shape=(N-i-j+1, N-i-j+2)))
        
    def Hkk(i,j,k):
        return(diags([np.array([i]*(N-i-j+1))*(delta_R*xi*(1-sigma_l_u))]
              ,[0]))  
    def Hkk1(i,j,k):
        return(diags([np.array([i]*(N-i-j+1))*(delta_R*xi*(sigma_l_u))]
              ,[0]))      
        
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
        Cjj=block_diag(Diagonal,dtype='float64')
        return(Cjj)
    def Cjj_v(i,j):
        Diagonal=[]
        for k in range(0,j+1):
            Diagonal.append(Jkk_v(i,j,k))
        Cjj_v=block_diag(Diagonal,dtype='float64')
        return(Cjj_v)     
    def A1ii(i):
        Diagonal=[]
        SuperDiagonal=[]
        for j in range(0,N-i+1):
            Diagonal.append(Cjj(i,j))
            SuperDiagonal.append(Cjj1(i,j))
        A1ii=SupDiagBlockMatrix(Diagonal,SuperDiagonal)
        return(A1ii.toarray())
    def A1ii_v(i):
        Diagonal=[]
        SuperDiagonal=[]
        for j in range(0,N-i+1):
            Diagonal.append(Cjj_v(i,j))
            SuperDiagonal.append(coo_matrix(((j+1)*(N-i-j+1),(j+2)*(N-i-j+1))))
        A1ii_v=SupDiagBlockMatrix(Diagonal,SuperDiagonal)
        return(A1ii_v.toarray())
    def A1ii_u(i):
        return(A1ii(i)-A1ii_v(i))
    K=1
    i=1
    k=1
    H=[]
    d=[]
    print('Inverse Step',i)
    H.append(-inv(Aii(i)))
    d.append(H[i-1].dot(A1ii_v(i).sum(axis=1)+Aii_v(i).sum(axis=1)))
    while i<N:
        i=i+1
        #print('Inverse Step',i)
        H.append(-inv(Aii(i)+multi_dot([A1ii(i),H[i-2],Aii1(i-1)])))
        d.append(H[i-1].dot(A1ii(i).dot(d[i-2])+A1ii_v(i).sum(axis=1)+Aii_v(i).sum(axis=1)))
    M=[[]]*N
    M_invasion=[]
    M[i-1]=d[i-1]
    while i>1:
        i=i-1
        M[i-1]=d[i-1]+multi_dot([H[i-1],Aii1(i),M[i]])
    M_invasion.append(float(M[0][N-1]))
    while k<K:
        k=k+1
        d[i-1]=k*multi_dot([H[i-1],Aii_v(i),M[i-1]])
        while i<N:
            i=i+1
            d[i-1]=H[i-1].dot(A1ii(i).dot(d[i-2])+k*A1ii_v(i).dot(M[i-2])+k*Aii_v(i).dot(M[i-1]))
        M[i-1]=d[i-1]
        while i>1:
            i=i-1
            M[i-1]=d[i-1]+multi_dot([H[i-1],Aii1(i),M[i]])
        M_invasion.append(float(M[0][N-1]))
    return -M_invasion[0]
#%% Optimization: find q that maximizes expected vaccinations
qlist=np.arange(0, 1+0.05, 0.05)
Ev=[0]*len(qlist)
i=0
for q0 in qlist:
    q=q0
    Ev[i]=M(q)
    i+=1

import scipy.optimize
sol1=scipy.optimize.minimize(M,[0.1], method="SLSQP",bounds=[(0,1)])

etamax=sol1.x*C
#%%
def Convert(lst): 
    return [ -i for i in lst ] 
plt.xlabel('q')
plt.ylabel('Ev')
plt.plot(qlist,Convert(Ev),sol1.x,-sol1.fun,'o')
   