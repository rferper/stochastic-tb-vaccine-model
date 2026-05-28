#%% Parameters %%#
import Parameters as par
from numpy import random
import matplotlib.pyplot as plt
import math
import numpy
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
theta_l_u=par.global_sigma_l_u 
theta_s_u=par.global_sigma_s_u
a_u=par.global_a_u
a_v=par.global_a_v
b_u=par.global_b_u
b_v=par.global_b_v
def r_u_bar(I): 
   return b_u
def r_u_tilde(I): 
   return a_u+b_u*(I-1)
def r_v_bar(I): 
   return b_v
def r_v_tilde(I): 
   return a_v+b_v*(I-1)
#%% Simulation %%#
def Sim_Trayectoria(X0=[1,0,0,N-1,0],plot=0):
    X=[]
    T=[]
    Xmax=0
    V=0
    t=0
    i=X0[0]
    lu=X0[1]
    lv=X0[2]
    su=X0[3]
    sv=X0[4]
    X.append([i,lu,lv,su,sv])
    T.append(t)
    Pi=[gamma*sv,
        alpha_u*(1-p_u)*su*i,
        alpha_v*(1-p_v)*sv*i,
        r_u(i)*lu,
        r_v(i)*lv,
        alpha_u*p_u*su*i,
        alpha_v*p_v*sv*i,
        delta_R*i*xi*(1-theta_l_u),
        delta_R*i*xi*theta_l_u,
        delta_R*i*(1-xi)*(1-theta_s_u),
        delta_R*(1-xi)*i*theta_s_u,
        (delta_TB+delta_D)*i*(1-eta(q,C)),
        (delta_TB+delta_D)*i*eta(q,C),
        delta_D*su*(1-eta(q,C)),
        delta_D*su*eta(q,C),
        delta_D*sv*(1-eta(q,C)),
        delta_D*sv*eta(q,C),
        delta_D*lu*(1-eta(q,C)),
        delta_D*lu*eta(q,C),
        delta_D*lv*(1-eta(q,C)),
        delta_D*lv*eta(q,C)]
    Spi=sum(Pi)
    Xmax=max(Xmax,i)
    while i!=0 and t<1000:
        u=random.uniform(0,1)
        t=-math.log(u)/Spi+t
        v=random.uniform(0,1)
        j=0
        S=Pi[j]
        while S/Spi<=v:
            j=j+1
            S=S+Pi[j]
        if j==0:
            su=su+1
            sv=sv-1
        elif j==1:
            lu=lu+1
            su=su-1
        elif j==2:
            lv=lv+1
            sv=sv-1
        elif j==3:
            i=i+1
            lu=lu-1
            Xmax=max(Xmax,i)
        elif j==4:
            i=i+1
            lv=lv-1
            Xmax=max(Xmax,i)
        elif j==5:
            i=i+1
            su=su-1
            Xmax=max(Xmax,i)
        elif j==6:
            i=i+1
            sv=sv-1
            Xmax=max(Xmax,i)
        elif j==7:
            i=i-1
            lv=lv+1
        elif j==8:
            i=i-1
            lu=lu+1
        elif j==9:
            i=i-1
            sv=sv+1
        elif j==10:
            i=i-1
            su=su+1
        elif j==11:
            i=i-1
            su=su+1
        elif j==12:
            i=i-1
            sv=sv+1
            V=V+1
        elif j==13:
            i=i
        elif j==14:
            su=su-1
            sv=sv+1
            V=V+1
        elif j==15:
            su=su+1
            sv=sv-1
        elif j==16:
           V=V+1
        elif j==17:
            lu=lu-1
            su=su+1
        elif j==18:
            lu=lu-1
            sv=sv+1
            V=V+1
        elif j==19:
            lv=lv-1
            su=su+1
        else:
            lv=lv-1
            sv=sv+1
            V=V+1
        X.append([i,lu,lv,su,sv])
        T.append(t)
        Pi=[gamma*sv,alpha_u*(1-p_u)*su*i,alpha_v*(1-p_v)*sv*i,r_u(i)*lu,r_v(i)*lv,
            alpha_u*p_u*su*i,alpha_v*p_v*sv*i,delta_R*i*xi*(1-theta_l_u),
            delta_R*i*xi*theta_l_u,delta_R*i*(1-xi)*(1-theta_s_u),
            delta_R*(1-xi)*i*theta_s_u,(delta_TB+delta_D)*i*(1-eta(q,C)),
            (delta_TB+delta_D)*i*eta(q,C),delta_D*su*(1-eta(q,C)),delta_D*su*eta(q,C),
            delta_D*sv*(1-eta(q,C)),delta_D*sv*eta(q,C),delta_D*lu*(1-eta(q,C)),
            delta_D*lu*eta(q,C),delta_D*lv*(1-eta(q,C)),delta_D*lv*eta(q,C)]
        Spi=sum(Pi)
    if plot==1:
        X=numpy.array(X)
        I=X[:,0]
        Lu=X[:,1]
        Lv=X[:,2]
        Su=X[:,3]
        Sv=X[:,4]       
        plt.plot(T, I, 'r') # plotting t, a separately 
        plt.plot(T, Lu, 'b') # plotting t, b separately 
        plt.plot(T, Lv, 'c') # plotting t, c separately
        plt.plot(T, Su, 'g') # plotting t, b separately 
        plt.plot(T, Sv, 'y') # plotting t, c separately 
        plt.show()
    return(t,Xmax,V)

def Sim_Trayectoria_R0(X0=[1,0,0,N-1,0],plot=0):
    R0=0
    t=0
    i_marcado=1
    i=X0[0]-1
    lu=X0[1]
    lv=X0[2]
    su=X0[3]
    sv=X0[4]
    Pi=[gamma*sv,
        alpha_u*(1-p_u)*su*(i+1),
        alpha_v*(1-p_v)*sv*(i+1),
        r_u_bar(i)*lu,
        r_u_tilde(i)*lu,
        r_v_bar(i)*lv,
        r_v_tilde(i)*lv,
        alpha_u*p_u*su*1,
        alpha_u*p_u*su*i,
        alpha_v*p_v*sv*1,
        alpha_v*p_v*sv*i,
        delta_R*1*xi*(1-theta_l_u),
        delta_R*i*xi*(1-theta_l_u),
        delta_R*1*xi*theta_l_u,
        delta_R*i*xi*theta_l_u,
        delta_R*1*(1-xi)*(1-theta_s_u),
        delta_R*i*(1-xi)*(1-theta_s_u),
        delta_R*(1-xi)*1*theta_s_u,
        delta_R*(1-xi)*i*theta_s_u,
        (delta_TB+delta_D)*1*(1-eta(q,C)),
        (delta_TB+delta_D)*i*(1-eta(q,C)),
        (delta_TB+delta_D)*1*eta(q,C),
        (delta_TB+delta_D)*i*eta(q,C),
        delta_D*su*(1-eta(q,C)),
        delta_D*su*eta(q,C),
        delta_D*sv*(1-eta(q,C)),
        delta_D*sv*eta(q,C),
        delta_D*lu*(1-eta(q,C)),
        delta_D*lu*eta(q,C),
        delta_D*lv*(1-eta(q,C)),
        delta_D*lv*eta(q,C)]
    Spi=sum(Pi)
    while i_marcado!=0 and t<1000:
        u=random.uniform(0,1)
        t=-math.log(u)/Spi+t
        v=random.uniform(0,1)
        j=0
        S=Pi[j]
        while S/Spi<=v:
            j=j+1
            S=S+Pi[j]
        if j==0:
            su=su+1
            sv=sv-1
        elif j==1:
            lu=lu+1
            su=su-1
        elif j==2:
            lv=lv+1
            sv=sv-1
        elif j==3:
            R0=R0+1
            i=i+1
            lu=lu-1
        elif j==4:
            i=i+1
            lu=lu-1
        elif j==5:
            R0=R0+1
            i=i+1
            lv=lv-1
        elif j==6:
            i=i+1
            lv=lv-1
        elif j==7:
            R0=R0+1
            i=i+1
            su=su-1
        elif j==8:
            i=i+1
            su=su-1
        elif j==9:
            R0=R0+1
            i=i+1
            sv=sv-1
        elif j==10:
            i=i+1
            sv=sv-1
        elif j==11:
            i_marcado=0
            lv=lv+1
        elif j==12:
            i=i-1
            lv=lv+1
        elif j==13:
            i_marcado=0
            lu=lu+1
        elif j==14:
            i=i-1
            lu=lu+1
        elif j==15:
            i_marcado=0
            sv=sv+1
        elif j==16:
            i=i-1
            sv=sv+1
        elif j==17:
            i_marcado=0
            su=su+1
        elif j==18:
            i=i-1
            su=su+1
        elif j==19:
            i_marcado=0
            su=su+1
        elif j==20:
            i=i-1
            su=su+1
        elif j==21:
            i_marcado=0
            sv=sv+1
        elif j==22:
            i=i-1
            sv=sv+1
        elif j==23:
            i=i
        elif j==24:
            su=su-1
            sv=sv+1
        elif j==25:
            su=su+1
            sv=sv-1
        elif j==26:
           i=i
        elif j==27:
            lu=lu-1
            su=su+1
        elif j==28:
            lu=lu-1
            sv=sv+1
        elif j==29:
            lv=lv-1
            su=su+1
        else:
            lv=lv-1
            sv=sv+1
        Pi=[gamma*sv,
            alpha_u*(1-p_u)*su*(i+1),
            alpha_v*(1-p_v)*sv*(i+1),
            r_u_bar(i)*lu,
            r_u_tilde(i)*lu,
            r_v_bar(i)*lv,
            r_v_tilde(i)*lv,
            alpha_u*p_u*su*1,
            alpha_u*p_u*su*i,
            alpha_v*p_v*sv*1,
            alpha_v*p_v*sv*i,
            delta_R*1*xi*(1-theta_l_u),
            delta_R*i*xi*(1-theta_l_u),
            delta_R*1*xi*theta_l_u,
            delta_R*i*xi*theta_l_u,
            delta_R*1*(1-xi)*(1-theta_s_u),
            delta_R*i*(1-xi)*(1-theta_s_u),
            delta_R*(1-xi)*1*theta_s_u,
            delta_R*(1-xi)*i*theta_s_u,
            (delta_TB+delta_D)*1*(1-eta(q,C)),
            (delta_TB+delta_D)*i*(1-eta(q,C)),
            (delta_TB+delta_D)*1*eta(q,C),
            (delta_TB+delta_D)*i*eta(q,C),
            delta_D*su*(1-eta(q,C)),
            delta_D*su*eta(q,C),
            delta_D*sv*(1-eta(q,C)),
            delta_D*sv*eta(q,C),
            delta_D*lu*(1-eta(q,C)),
            delta_D*lu*eta(q,C),
            delta_D*lv*(1-eta(q,C)),
            delta_D*lv*eta(q,C)]
        Spi=sum(Pi)
    return(R0)
        
        