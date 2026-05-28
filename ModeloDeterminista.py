import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#%% Parameters %%#
N=20
gamma=1/5
q = 0.8
C = 0.8
eta=q*C
alpha_u=5.2
alpha_v=5.2
p_u=0.15
p_v=0
a_u=1.5*10**(-4)
b_u=0.35*alpha_u
delta_TB = 0.2
delta_R = 4/3
delta_D=3/5
theta_u=0.5

#%% ODE System %%#
def f(t,y):
    Su=y[0]
    Sv=y[1]
    Lu=y[2]
    Lv=y[3]
    I=N-y[0]-y[1]-y[2]-y[3]
    return[gamma*Sv-alpha_u*Su*I-delta_D*Su*eta+(I+Sv+Lu+Lv)*delta_D*(1-eta)+delta_TB*I*(1-eta),
           -gamma*Sv-alpha_v*Sv*I-delta_D*Sv*(1-eta)+(I+Su+Lu+Lv)*delta_D*eta+delta_TB*I*eta,
           alpha_u*(1-p_u)*Su*I+delta_R*I*theta_u-(a_u+b_u*I)*Lu-delta_D*Lu,
           alpha_v*Sv*I+delta_R*I*(1-theta_u)-delta_D*Lv,
           ]

#%% Equilibrium Points %%#
def g(y):
    I=y[0]
    Su=y[1]
    Sv=y[2]
    Lu=y[3]
    Lv=y[4]
    return[gamma*Sv-alpha_u*Su*I-delta_D*Su*eta+(I+Sv+Lu+Lv)*delta_D*(1-eta)+delta_TB*I*(1-eta),
           -gamma*Sv-alpha_v*Sv*I-delta_D*Sv*(1-eta)+(I+Su+Lu+Lv)*delta_D*eta+delta_TB*I*eta,
           alpha_u*(1-p_u)*Su*I+delta_R*I*theta_u-(a_u+b_u*I)*Lu-delta_D*Lu,
           alpha_v*Sv*I+delta_R*I*(1-theta_u)-delta_D*Lv
           ]

#%% Numerical Solution %%#
y0=[N-1,0,0,0]
sol = solve_ivp(f, [0, 500],y0)

t=sol.t
Su=sol.y[0]
Sv=sol.y[1]
Lu=sol.y[2]
Lv=sol.y[3]
I=N-Su-Sv-Lu-Lv
plt.plot(t,I)
peq=[I[len(sol.y[0])-1],Su[len(sol.y[0])-1],Sv[len(sol.y[0])-1],Lu[len(sol.y[0])-1],Lv[len(sol.y[0])-1]]
print('Equilibrium point:', peq)
print('Equilibrium residual:', g(peq))
