import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit

Time = time.time()
T_initial = 1
T_final   = 2
step = 0.1
separation = int((T_final-T_initial)/step) +1
temperature = np.linspace(T_initial,T_final,separation)

a = 20 
b = 20 # defines the size, size = (a,b)
Nbin = 5000
Ntest = 100

Nsweep = 15
num = Nsweep*a*b
spin = np.ones((a,b),dtype=int) #generates the spin field
J_1 = 1
J_2 = 1
J_3 = -0.2

@njit
def sweep(t,spin,position_x,position_y,random_number):
    M = np.zeros(Nbin)
    E = np.zeros(Nbin)
    for n in range(num*Nbin):
        x = position_x[n]
        y = position_y[n]
        dE = 2*spin[x,y]*(
            J_1*(spin[(x+1)%a,y]+spin[(x-1)%a,y])
            +J_2*(spin[x,(y+1)%b]+spin[x,(y-1)%b])
            +J_3*(spin[(x+1)%a,(y+1)%a]+spin[(x-1)%a,(y-1)%a]
             +spin[(x+1)%a,(y+1)%a]+spin[(x-1)%a,(y-1)%a]))
        #Calculate the energy
        if np.exp(-dE/t) > random_number[n]:
            spin[x,y] *=-1 #Reject or accept the flip with probability exp(-dE/t)
        if n%num==0:
            rows = np.arange(a)
            cols = np.arange(b)
            M[n//num] = np.abs(np.average(spin))
            row_shifted_spin = spin[(rows+1)%a,:]
            E[n//num] = np.average(spin*
                                (J_1*row_shifted_spin+J_2*spin[:,(cols+1)%b]
                                +J_3*(
                                    row_shifted_spin[:,(cols+1)%b]+row_shifted_spin[:,(cols-1)%b]
                                    )
                                ))
    return spin, M , E

def Ising(t,spin):
    Time = time.time()
    #generate the random number outside the loop
    position_x = np.random.randint(a,size=num*Nbin)
    position_y = np.random.randint(b,size=num*Nbin)
    random_number = np.random.uniform(0,1,size=(num*Nbin))
    ##sweeping
    spin, M , E = sweep(t,spin,position_x,position_y,random_number)
    #Finish sweeping    
    

    M = np.reshape(M,(Ntest,Nbin//Ntest))
    E = np.reshape(E,(Ntest,Nbin//Ntest))

    Magnetization = np.average(M,axis=1)
    Susceptibility = (np.average(M**2,axis=1) - np.average(M,axis=1)**2)/t
    Energy = np.average(E,axis=1)
    Heat_Capacitance = (np.average(E**2,axis=1) - np.average(E,axis=1)**2)/ t**2
    print('Time used for each tempreture is',time.time()-Time)
    return Magnetization,Susceptibility , Energy,Heat_Capacitance #returns the M and E at given temperature

def Confidence_interval(arr):
    std = np.std(arr,ddof=1,axis= 1)
    n = np.size(arr,axis = 1)
    return std/np.sqrt(n)

Magnetization = np.zeros((separation,Ntest),dtype=float)
Susceptibility =np.zeros((separation,Ntest),dtype=float)
Energy = np.zeros((separation,Ntest),dtype=float)
Heat_Capacitance = np.zeros((separation,Ntest),dtype=float)


for t_ind in range(separation):
    Magnetization[t_ind,:], Susceptibility[t_ind,:],Energy[t_ind,:],Heat_Capacitance[t_ind,:]= Ising(temperature[t_ind],spin)


fig, axs = plt.subplots(2,2)

#####First plotting, avg(M)

axs[0,0].errorbar(temperature,np.average(Magnetization,axis=1),yerr = Confidence_interval(Magnetization))
axs[0,0].set_xlim(T_initial,T_final)
axs[0,0].set_title('Magnetization')

##second plotting, chi
axs[0,1].errorbar(temperature,np.average(Susceptibility,axis=1),yerr = Confidence_interval(Susceptibility))
axs[0,1].set_xlim(T_initial,T_final)
axs[0,1].set_title('Susceptibility')

###Third plotting, Energy

axs[1,0].errorbar(temperature,np.average(Energy,axis=1),yerr = Confidence_interval(Energy))
axs[1,0].set_xlim(T_initial,T_final)
axs[1,0].set_title('Average energy per site')


#####Fourth plotting, C_v

axs[1,1].errorbar(temperature,np.average(Heat_Capacitance,axis=1),yerr = Confidence_interval(Heat_Capacitance))
axs[1,1].set_xlim(T_initial,T_final)
axs[1,1].set_title('Heat Capacitance')
print('Total time used is',time.time()-Time)

plt.show()