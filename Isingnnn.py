import numpy as np
import matplotlib.pyplot as plt
from Sweep import Ising
import time



Time = time.time()
T_initial = 1.5
T_final   = 1.9
step = 0.02
separation = int((T_final-T_initial)/step) +1
temperature = np.linspace(T_initial,T_final,separation)
separation = len(temperature)
#defines an array, consists of temperature we want to calculate.

a = 20 
b = 20 # defines the size of spin field, size = (a,b)

spin = np.ones((a,b),dtype=int) #generates the spin field


Magnetization = np.zeros((separation,2),dtype=float) 
# M[t,0] record the average value at temperature t, and M[t,1] record the error of this value
Susceptibility =np.zeros((separation,2),dtype=float)
Energy = np.zeros((separation,2),dtype=float)
Heat_Capacitance = np.zeros((separation,2),dtype=float)


for t_ind in range(separation):
    spin, Magnetization[t_ind,:], Susceptibility[t_ind,:],Energy[t_ind,:],Heat_Capacitance[t_ind,:]= Ising(temperature[t_ind],spin)


fig, axs = plt.subplots(2,2)

#####First plotting, avg(M)

axs[0,0].errorbar(temperature,Magnetization[:,0],yerr = Magnetization[:,1])
axs[0,0].set_xlim(T_initial,T_final)
axs[0,0].set_title('Magnetization')

##second plotting, chi
axs[0,1].errorbar(temperature,Susceptibility[:,0],yerr = Susceptibility[:,1])
axs[0,1].set_xlim(T_initial,T_final)
axs[0,1].set_title('Susceptibility')

###Third plotting, Energy

axs[1,0].errorbar(temperature,Energy[:,0],yerr= Energy[:,1])
axs[1,0].set_xlim(T_initial,T_final)
axs[1,0].set_title('Average energy per site')


#####Fourth plotting, C_v

axs[1,1].errorbar(temperature,Heat_Capacitance[:,0],yerr= Heat_Capacitance[:,1])
axs[1,1].set_xlim(T_initial,T_final)
axs[1,1].set_title('Heat Capacitance')
print('Total time used is',time.time()-Time)

plt.show()