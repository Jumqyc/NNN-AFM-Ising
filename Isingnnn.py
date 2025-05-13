import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
from Equilibrium_check import is_Equilibrium
#couplings J1 is along x axis, J2 along y axis nn hopping and J3 is nnn hopping
J_1 = 1
J_2 = 1
J_3 = -0.2
Ntest = 2


def Confidence_interval(arr,axis): #returns the confidence interval of given array. 
    #Only used in the end
    std = np.std(arr,ddof=1,axis= axis)
    n = np.size(arr,axis=axis)
    return std/np.sqrt(n)*1.96


def sweep(t:float,spin):
    # the latter four variables are not important, only to speed up the program by generating the random numbers outside the loop. 
    # t is the temperature.
    a,b = np.shape(spin)
    num = a*b
    
    #record the first M and E
    #!!! Unnormalized !!!
    M = [np.abs(np.sum(spin))]
    E = [np.sum(spin*(
        J_1*np.roll(spin,1,1)+
        J_2*np.roll(spin,1,0)+
        J_3*np.roll(np.roll(spin,1,1),1,0)
        +J_3*np.roll(np.roll(spin,1,1),-1,0)))]
    n = 0
    while True:
        acc_E = 0
        acc_M = 0
        position_x = np.random.randint(a,size=num)
        position_y = np.random.randint(b,size=num)
        random_number = np.random.uniform(0,1,size=(num))
        random_number = t*np.log(random_number)
        spin,acc_M,acc_E = single_sweep(spin,position_x,position_y,random_number,a,b)

        M.append(M[-1]+acc_M) #record it after each sweep
        E.append(E[-1]+acc_E)
        n+=1
        if n > 500 and n%1000 == 0: # do not test if the length is too small
            if is_Equilibrium(M) and is_Equilibrium(E):
                break
    # turn it into array and mean over site
    M = np.abs(np.array(M) / num)
    E = np.array(E) / num 
    # Processing the recorded data 
    Magnetization = np.mean(M)
    Susceptibility = (np.mean(M**2) - np.mean(M)**2)/t
    Energy = np.mean(E)
    Heat_Capacitance = (np.mean(E**2) - np.mean(E)**2)/ t**2

    return spin, [Magnetization, Susceptibility, Energy, Heat_Capacitance]

@njit
def single_sweep(spin,position_x,position_y,random_number,a,b):
    num = a*b
    acc_E = 0
    acc_M = 0
    for n in range(num):
            x = position_x[n]
            y = position_y[n]
            dE = 2*spin[x,y]*(
            J_1*(spin[(x+1)%a,y]+spin[(x-1)%a,y])
            +J_2*(spin[x,(y+1)%b]+spin[x,(y-1)%b])
            +J_3*(spin[(x+1)%a,(y+1)%a]+spin[(x-1)%a,(y-1)%a]
            +spin[(x-1)%a,(y+1)%a]+spin[(x+1)%a,(y-1)%a])
            )
        #Calculate the energy
            if -dE > random_number[n]:
                spin[x,y] *= -1 #Reject or accept the flip with probability exp(-dE/t)
                acc_E += dE
                acc_M += 2*spin[x,y]
    return spin,acc_M,acc_E

Time = time.time()
temperature = np.linspace(1.5,1.9,3)
separation = len(temperature)
#defines an array, consists of temperature we want to calculate.

a = 20 
b = 20 # defines the size of spin field, size = (a,b)

res = np.zeros((separation,4,Ntest),dtype=float)
spin = np.ones((a,b),dtype=np.int8) #generates the spin field

for t_ind,t in enumerate(temperature):
    for n in range(Ntest):
        spin,res[t_ind,:,n] = sweep(t,spin)
        print('temperature = ',t,'test = ',n)

fig, axs = plt.subplots(2,2)

#####First plotting, avg(M)

axs[0,0].errorbar(temperature,np.mean(res[:,0,:],axis = 1),yerr = Confidence_interval(res[:,0,:],axis = 1))
axs[0,0].set_xlim(temperature[0],temperature[-1])
axs[0,0].set_title('Magnetization')

##second plotting, chi
axs[0,1].errorbar(temperature,np.mean(res[:,1,:],axis = 1),yerr = Confidence_interval(res[:,1,:],axis = 1))
axs[0,1].set_xlim(temperature[0],temperature[-1])
axs[0,1].set_title('Susceptibility')

###Third plotting, Energy

axs[1,0].errorbar(temperature,np.mean(res[:,2,:],axis = 1),yerr = Confidence_interval(res[:,2,:],axis = 1))
axs[1,0].set_xlim(temperature[0],temperature[-1])
axs[1,0].set_title('mean energy per site')


#####Fourth plotting, C_v

axs[1,1].errorbar(temperature,np.mean(res[:,3,:],axis = 1),yerr = Confidence_interval(res[:,3,:],axis = 1))
axs[1,1].set_xlim(temperature[0],temperature[-1])
axs[1,1].set_title('Heat Capacitance')
print('Total time used is',time.time()-Time)

plt.show()