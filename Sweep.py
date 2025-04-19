import numpy as np
from time import time
from Equilibrium_check import is_Equilibrium
from numba import njit
#couplings J1 is along x axis, J2 along y axis nn hopping and J3 is nnn hopping
J_1 = 1
J_2 = 1
J_3 = -0.2
Nbin = 100
Ntest = 10
Nsweep = 10


def Confidence_interval(arr): #returns the confidence interval of given array. 
    #Only used in the end
    std = np.std(arr,ddof=1)
    n = np.size(arr)
    return std/np.sqrt(n)*1.96


@njit(parallel = True)
def sweep(t:float,
          spin,
          position_x,
          position_y,
          random_number,
          Nbin:int=100,
          Nsweep:int=10):
    # the latter four variables are not important, only to speed up the program by generating the random numbers outside the loop. 
    # t is the temperature.
    # calculate Magnetization and Energy for Nbin = 100 configuration, then calculate the average Magnetization, Susceptibility, Energy and Heat capacitance
    a,b = np.shape(spin)
    num = Nsweep*a*b
    
    M = np.zeros(Nbin)
    E = np.zeros(Nbin)
    for n in range(num*Nbin):
        x = position_x[n]
        y = position_y[n]
        dE = 2*spin[x,y]*(
            J_1*(spin[(x+1)%a,y]+spin[(x-1)%a,y])
            +J_2*(spin[x,(y+1)%b]+spin[x,(y-1)%b])
            +J_3*(spin[(x+1)%a,(y+1)%a]+spin[(x-1)%a,(y-1)%a]
            +spin[(x-1)%a,(y+1)%a]+spin[(x+1)%a,(y-1)%a])
            )
        #Calculate the energy
        if np.exp(-dE/t) > random_number[n]:
            spin[x,y] *=-1 #Reject or accept the flip with probability exp(-dE/t)
        
        if n%num==0: #record E and M after sweeping 
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

    # Processing the recorded data 
    Magnetization = np.average(M)
    Susceptibility = (np.average(M**2) - np.average(M)**2)/t
    Energy = np.average(E)
    Heat_Capacitance = (np.average(E**2) - np.average(E)**2)/ t**2

    return spin, Magnetization, Susceptibility, Energy, Heat_Capacitance

def Ising(t,spin):
    Time = time()
    a,b = np.shape(spin)
    num = Nsweep*a*b
    Magnetization    = []
    Susceptibility   = []
    Energy           = []
    Heat_Capacitance = []
    
    while True:
        #generate the random number outside the loop
        position_x = np.random.randint(a,size=num*Nbin)
        position_y = np.random.randint(b,size=num*Nbin)
        random_number = np.random.uniform(0,1,size=(num*Nbin))

    ##sweeping
        spin, Magnetization_val ,Susceptibility_val,Energy_val,Heat_Capacitance_val = sweep(t,spin,position_x,position_y,random_number)
    #Finish sweeping, returns updated spin and physics quantity.
    
        Magnetization.append(Magnetization_val)
        Susceptibility.append(Susceptibility_val)
        Energy.append(Energy_val)
        Heat_Capacitance.append(Heat_Capacitance_val)

        if len(Magnetization)>100:
            if is_Equilibrium(Magnetization) and is_Equilibrium(Susceptibility) and is_Equilibrium(Energy) and is_Equilibrium(Heat_Capacitance): #Do KS sampling
                break

    Magnetization = np.array([np.average(Magnetization),Confidence_interval(Magnetization)])
    Susceptibility = np.array([np.average(Susceptibility),Confidence_interval(Susceptibility)])
    Energy = np.array([np.average(Energy),Confidence_interval(Energy)])
    Heat_Capacitance = np.array([np.average(Heat_Capacitance),Confidence_interval(Heat_Capacitance)])

    print('Time used for each temperature is',time()-Time)

    return spin ,Magnetization ,Susceptibility , Energy,Heat_Capacitance #returns physical quantities and their errors