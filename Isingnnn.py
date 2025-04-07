import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
import statsmodels.api as sm
from scipy.stats import ks_2samp

def is_Equilibrium(data: np.array, 
                     auto_corr_thresh: float = 1e-2,
                     stat_threshold: float = 0.1,
                     p_threshold: float = 0.01,
                     min_data_points: int = 200) -> bool:
    
    
    # auto-corr analysis
    def _auto_correlation(vec):
        nlags = int(len(vec) * 3/4)
        if nlags < 1:
            return True, -1
        
        try:
            acf = sm.tsa.acf(vec, nlags=nlags, fft=False)
        except:  
            return True, -1
            
        acf_abs = np.abs(acf)
        lag_candidates = np.where(acf_abs <= auto_corr_thresh)[0]
        
        if len(lag_candidates) == 0:
            return False, -1  # 未找到合适滞后长度
        return False, lag_candidates[0]  # 返回第一个满足条件的滞后
    
    same, lag = _auto_correlation(data)
    if same or lag < 1:
        return False
    
    # slicing
    sampled_data = data[::lag]
    
    # Make sure there is enough sample
    if len(sampled_data) < 2:
        return False
    
    # separate into two parts
    split_idx = len(sampled_data) // 2
    part1 = sampled_data[:split_idx]
    part2 = sampled_data[split_idx:]
    
    # KS sampling
    stat, p_value = ks_2samp(part1, part2)
    
    # returns bool
    return (
        (p_value >= p_threshold or stat <= stat_threshold) 
        and len(sampled_data) >= min_data_points
    )



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
Nbin = 100
Ntest = 10

Nsweep = 10
num = Nsweep*a*b
spin = np.ones((a,b),dtype=int) #generates the spin field

#couplings J1 is along x axis, J2 along y axis nn hopping and J3 is nnn hopping
J_1 = 1
J_2 = 1
J_3 = -0.2

def Confidence_interval(arr): #returns the confidence interval of given array. 
    #Only used in the end
    std = np.std(arr,ddof=1)
    n = np.size(arr)
    return std/np.sqrt(n)*1.96


@njit
def sweep(t,spin,position_x,position_y,random_number):
    # the latter four variables are not important, only to speed up the program by generating the random numbers outside the loop. 
    # t is the temperature.
    # calculate Magnetization and Energy for Nbin = 100 configuration, then calculate the average Magnetization, Susceptibility, Energy and Heat capacitance
    M = np.zeros(Nbin)
    E = np.zeros(Nbin)
    for n in range(num*Nbin):
        x = position_x[n]
        y = position_y[n]
        dE = 2*spin[x,y]*(
            J_1*(spin[(x+1)%a,y]+spin[(x-1)%a,y])
            +J_2*(spin[x,(y+1)%b]+spin[x,(y-1)%b])
            +J_3*(spin[(x+1)%a,(y+1)%a]+spin[(x-1)%a,(y-1)%a]
            +spin[(x+1)%a,(y+1)%a]+spin[(x-1)%a,(y-1)%a])
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
    Time = time.time()
    
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

    print('Time used for each temperature is',time.time()-Time)

    return Magnetization ,Susceptibility , Energy,Heat_Capacitance #returns physical quantities and their errors




Magnetization = np.zeros((separation,2),dtype=float) 
# M[t,0] record the average value at temperature t, and M[t,1] record the error of this value
Susceptibility =np.zeros((separation,2),dtype=float)
Energy = np.zeros((separation,2),dtype=float)
Heat_Capacitance = np.zeros((separation,2),dtype=float)


for t_ind in range(separation):
    Magnetization[t_ind,:], Susceptibility[t_ind,:],Energy[t_ind,:],Heat_Capacitance[t_ind,:]= Ising(temperature[t_ind],spin)


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