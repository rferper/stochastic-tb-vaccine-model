global_N=20 # Size of the population

global_delta_D=(3/5) # per capita death rate from causes beyond TB
global_gamma=1/5 # per capita duration rate of the vaccine protection

global_q = 0.8# take (fraction of vaccinated individuals in whom the vaccione induces
      # some degree of protection)

global_C = 0.8 # Coverage level (fraction of individuals that are vaccinated)

def global_eta(q,C): #Probability that a newborn will be vaccinated succesfuly (with some degree of protection)
   return q*C

global_alpha_u=5.2 # per capita infectious contact rate on unvaccinated susceptible individuals

global_alpha_v=5.2 # per capita infectious contact rate on vaccinated susceptible individuals

global_p_u=0.15 # probability that a non-vaccinated susceptible becomes infective

global_p_v=0 # probability that a vaccinated susceptible becomes infective

global_a_u = 1.5*10**(-4) # per capita reactivation rate of a non-vaccinated latenly infected individual

global_b_u = 0.35*global_alpha_u # per capita reinfection rate of a non-vaccinated latenly infected individual

def global_r_u(I): #per capita rate of reactivation and reinfection of a non-vaccinated latently infected individual
   return global_a_u+global_b_u*I

global_a_v = 0 # per capita reactivation rate of a vaccinated latenly infected individual

global_b_v = 0 # per capita reinfection rate of a vaccinated latenly infected individual

def global_r_v(I): #per capita rate of reactivation and reinfection of a vaccinated latently infected individual
   return global_a_v+global_b_v*I

global_delta_TB = 0.2 # per capita rate of death due to tuberculosis

global_delta_R = 4/3 # per capita effective treatment rate

global_xi = 1 # probability that a recovered individual is considered as latent

global_sigma_l_u=0.5 # probability that a recovered latent individual is considered as an unvaccinated individual

global_sigma_s_u=1 # probability that a recovered susceptible individual is considered as an unvaccinated individual

