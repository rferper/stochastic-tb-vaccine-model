#%% PARAMETERS -- GLOBAL VARIABLES %%#

N=6 # Size of the population


gamma=1 # per capita duration rate of the vaccine protection


q = 0.5 # take (fraction of vaccinated individuals in whom the vaccione induces
      # some degree of protection)

C = 1 # Coverage level (fraction of individuals that are vaccinated)


def eta(q,C): #Probability that a newborn will be vaccinated succesfuly (with some degree of protection)
   return q*C


alpha_u=1 # per capita infectious contact rate on unvaccinated susceptible individuals


alpha_v=1 # per capita infectious contact rate on vaccinated susceptible individuals


p_u=0.5 # probability that a non-vaccinated susceptible becomes infective

global p_v
p_v=0.5 # probability that a vaccinated susceptible becomes infective


a_u = 1 # per capita reactivation rate of a non-vaccinated latenly infected individual
b_u = 1 # per capita reinfection rate of a non-vaccinated latenly infected individual


def r_u(I): #per capita rate of reactivation and reinfection of a non-vaccinated latently infected individual
   return a_u+b_u*I

a_v = 1 # per capita reactivation rate of a vaccinated latenly infected individual
b_v = 1 # per capita reinfection rate of a vaccinated latenly infected individual


def r_v(I): #per capita rate of reactivation and reinfection of a vaccinated latently infected individual
   return a_v+b_v*I


delta_TB = 1 # per capita rate of death due to tuberculosis


delta_R = 1 # per capita effective treatment rate


delta_D=1 # per capita death rate from causes beyond TB


xi = 0.5 # probability that a recovered individual is considered as latent

sigma_l_u=0.5 # probability that a recovered latent individual is considered as an unvaccinated individual

sigma_s_u=0.5 # probability that a recovered susceptible individual is considered as an unvaccinated individual



#%% STATES %%#

# Transient states
print('Transient states')
for i in range(1,N+1):
    for j in range(0,N-i+1):
        for k in range(0,j+1):
            for m in range(0,N-i-j+1):
                print(''.join(map(str,[i,k,j-k,m,N-i-j-m])))
from scipy.special import binom
print('Number of transient states', binom(N+4,4)-binom(N+3,3))
# Absorbing states
print('Absorbing states')
for j in range(0,N+1):
    for k in range(0,j+1):
        for m in range(0,N-j+1):
            print(''.join(map(str,[0,k,j-k,m,N-j-m])))
from scipy.special import binom
print('Number of absorbing states', binom(N+3,3))
#%% Compute states given N,i,j,k %%#
def States3(i,j,k):
    S=[]
    for m in range(0,N-i-j+1):
        S.append(''.join(map(str,[i,k,j-k,m,N-i-j-m])))
    return(S)
def States2(i,j):
    S=[]
    for k in range(0,j+1):
        for m in range(0,N-i-j+1):
            S.append(''.join(map(str,[i,k,j-k,m,N-i-j-m])))
    return(S)
def States1(i):
    S=[]
    for j in range(0,N-i+1):
        for k in range(0,j+1):
            for m in range(0,N-i-j+1):
                S.append(''.join(map(str,[i,k,j-k,m,N-i-j-m])))
    return(S)
def States():
    S=[]
    for i in range(1,N+1):
        for j in range(0,N-i+1):
            for k in range(0,j+1):
                for m in range(0,N-i-j+1):
                    S.append(''.join(map(str,[i,k,j-k,m,N-i-j-m])))
    return(S)
#%% Construction of Aii %%#
import numpy as np
from scipy.sparse import diags,coo_matrix,vstack
from scipy.sparse import csr_matrix
import pandas as pd
N=6
i=2 # infected
j=2 # total of latents
k=1 # of unvaccinated latents

print('Construction of Aii')
print('The number of infectives remains constant')
print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('Level 4')
print('---------------------------------------------------------')
print('---------------------------------------------------------')

# Ekk+1 (N-i-j+1)x(N-i-j)
print('---------------------------------------------------------')
print('One unvaccinated susceptible becomes an unvaccinated latent')
print('Ekk+1 matrix. Rows: ',N-i-j+1,'Columns:',N-i-j)
rownames = States3(i,j,k)
colnames = States3(i,j+1,k+1)
df = pd.DataFrame(Ekk1(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)
print('---------------------------------------------------------')
# Ekk (N-i-j+1)x(N-i-j)
print('One vaccinated susceptible becomes a vaccinated latent')
print('Ekk matrix. Rows: ',N-i-j+1,'Columns',N-i-j)
rownames = States3(i,j,k)
colnames = States3(i,j+1,k)
df = pd.DataFrame(Ekk(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)
print('---------------------------------------------------------')
print('One vaccinated susceptible dies & is replaced by an unvaccinated susceptible or the effect of a vaccine wanes')
print('Fkk matrix. Rows: ',N-i-j+1,'Columns:',N-i-j+1)

rownames = States3(i,j,k)
colnames = States3(i,j,k)
df = pd.DataFrame(Fkk(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)
print('---------------------------------------------------------')
print('One vaccinated latent dies and is replaced by a susceptible')
print('Gkk matrix. Rows: ',N-i-j+1,'Columns:',N-i-j+2)
rownames = States3(i,j,k)
colnames = States3(i,j-1,k)
df = pd.DataFrame(Gkk(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)
print('---------------------------------------------------------')
print('One unvaccinated latent dies and is replaced by a susceptible')
print('G1kk matrix. Rows: ',N-i-j+1,'Columns:',N-i-j+2)
rownames = States3(i,j,k)
colnames = States3(i,j-1,k-1)
df = pd.DataFrame(G1kk(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)
print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('Level 3')
print('---------------------------------------------------------')
print('---------------------------------------------------------')
#%% %%#
from scipy.sparse import coo_matrix, block_diag, hstack, vstack, dia_matrix
print('---------------------------------------------------------')
print('One susceptible becomes latent')
print('Bjj1 matrix. Rows: ',(j+1)*(N-i-j+1),'Columns:',(j+2)*(N-i-j))

rownames = States2(i,j)
colnames = States2(i,j+1)
df = pd.DataFrame(Bjj1(i,j).toarray(), columns=colnames, index=rownames)
print(df)
    
print('---------------------------------------------------------')
print('One susceptible dies and is replaced by another susceptible or the effect of a vaccine wanes')
print('Bjj matrix. Rows: ',(j+1)*(N-i-j+1),'Columns:',(j+1)*(N-i-j+1))

rownames = States2(i,j)
colnames = States2(i,j)
df = pd.DataFrame(Bjj(i,j).toarray(), columns=colnames, index=rownames)
print(df)
print('---------------------------------------------------------')
print('One latent dies and is replaced by a susceptible')
print('B1jj matrix. Rows: ',(j+1)*(N-i-j+1),'Columns:',j*(N-i-j+2))

rownames = States2(i,j)
colnames = States2(i,j-1)
df = pd.DataFrame(B1jj(i,j).toarray(), columns=colnames, index=rownames)
print(df)

#%% %%#
print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('Level 2')
print('---------------------------------------------------------')
print('---------------------------------------------------------')
from scipy.special import binom
print('The number of infectives remains constant')
print('Aii matrix. Rows: ',binom(N-i+3,3) ,'Columns: ',binom(N-i+3,3))

rownames = States1(i)
colnames = States1(i)

df = pd.DataFrame(Aii(i).toarray(), columns=colnames, index=rownames)
print(df)
#%% %%#

print('Construction of Aii1')
print('One more infective')
print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('Level 4')
print('---------------------------------------------------------')
print('---------------------------------------------------------')

print('Infectious infection of a suceptible')
print('Lkk matrix. Rows: ',(N-i-j+1),'Columns:',(N-i-j))
rownames = States3(i,j,k)
colnames = States3(i+1,j,k)
df = pd.DataFrame(Lkk(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)

print('---------------------------------------------------------')
print('Diasease development on a vaccinated latent')
print('Mkk matrix. Rows: ',(N-i-j+1),'Columns:',(N-i-j+1))

rownames = States3(i,j,k)
colnames = States3(i+1,j-1,k)
df = pd.DataFrame(Mkk(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)

print('---------------------------------------------------------')
print('Diasease development on an unvaccinated latent')
print('M1kk matrix. Rows: ',(N-i-j+1),'Columns:',(N-i-j+1))

rownames = States3(i,j,k)
colnames = States3(i+1,j-1,k-1)
df = pd.DataFrame(M1kk(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)
print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('Level 3')
print('---------------------------------------------------------')
print('---------------------------------------------------------')

print('Infectious infection of a suceptible')
print('Djj matrix. Rows: ',(j+1)*(N-i-j+1),'Columns:',(j+1)*(N-i-j))
rownames = States2(i,j)
colnames = States2(i+1,j)
df = pd.DataFrame(Djj(i,j).toarray(), columns=colnames, index=rownames)
print(df)
print('Diasease development on a latent')
print('D1jj matrix. Rows: ',(j+1)*(N-i-j+1),'Columns:',j*(N-i-j+1))

rownames = States2(i,j)
colnames = States2(i+1,j-1)
df = pd.DataFrame(D1jj(i,j).toarray(), columns=colnames, index=rownames)
print(df)

print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('Level 2')
print('---------------------------------------------------------')
print('---------------------------------------------------------')

print('One more infective')
print('Aii1 matrix. Rows: ',binom(N-i+3,3) ,'Columns: ',binom(N-i+2,3))

rownames = States1(i)
colnames = States1(i+1)

df = pd.DataFrame(Aii1(i).toarray(), columns=colnames, index=rownames)
print(df)

#%% %%#

print('Construction of A1ii')
print('One less infective')
print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('Level 4')
print('---------------------------------------------------------')
print('---------------------------------------------------------')



print('An infective recovers and becomes a susceptible or an infective dies and is replaced by a susceptible')
print('Jkk matrix. Rows: ',(N-i-j+1),'Columns:',(N-i-j+2))
rownames = States3(i,j,k)
colnames = States3(i-1,j,k)
df = pd.DataFrame(Jkk(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)

print('---------------------------------------------------------')
print('An infective recovers and becomes an unvaccinated latent')
print('Hkk1 matrix. Rows: ',(N-i-j+1),'Columns:',(N-i-j+1))
rownames = States3(i,j,k)
colnames = States3(i-1,j+1,k+1)
df = pd.DataFrame(Hkk1(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)

print('---------------------------------------------------------')
print('An infective recovers and becomes a vaccinated latent')
print('Hkk matrix. Rows: ',(N-i-j+1),'Columns:',(N-i-j+1))
rownames = States3(i,j,k)
colnames = States3(i-1,j+1,k)
df = pd.DataFrame(Hkk(i,j,k).toarray(), columns=colnames, index=rownames)
print(df)

print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('Level 3')
print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('An infective recovers and becomes a latent')
print('Cjj1 matrix. Rows: ',(j+1)*(N-i-j+1) ,'Columns: ',(j+2)*(N-i-j+1))

rownames = States2(i,j)
colnames = States2(i-1,j+1)
df = pd.DataFrame(Cjj1(i,j).toarray(), columns=colnames, index=rownames)
print(df)

print('An infective recovers and becomes a susceptible or an infective dies and is replaced by a susceptible')
print('Cjj matrix. Rows: ',(j+1)*(N-i-j+1) ,'Columns: ',(j+1)*(N-i-j+2))

rownames = States2(i,j)
colnames = States2(i-1,j)
df = pd.DataFrame(Cjj(i,j).toarray(), columns=colnames, index=rownames)
print(df)

print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('Level 2')
print('---------------------------------------------------------')
print('---------------------------------------------------------')

print('One less infective')
print('A1ii matrix. Rows: ',binom(N-i+3,3) ,'Columns: ',binom(N-i+4,3))

rownames = States1(i)
colnames = States1(i-1)

df = pd.DataFrame(A1ii(i).toarray(), columns=colnames, index=rownames)
print(df)

#%% %%#


print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('Level 1')
print('---------------------------------------------------------')
print('---------------------------------------------------------')

print('State transitions between transient states')
print('QTT matrix. Rows: ',binom(N+4,4)-binom(N+3,3) ,'Columns: ',binom(N+4,4)-binom(N+3,3))

rownames = States()
colnames = States()

df = pd.DataFrame(QTT().toarray(), columns=colnames, index=rownames)
print(df)


