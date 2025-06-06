import ising as cising
from collections import namedtuple
from typing import Callable
import numpy as np
result = namedtuple("result", ["avg", "err"])

def rtime(data:np.ndarray,method = "acf"):
    if method == "binning":
        Nbin = 64
        N = len(data)
        k = N // Nbin
        data = data[:k*Nbin]  # Ensure data length is a multiple of Nbin
        Bin = data.reshape(k, Nbin)
        Bin = np.mean(Bin, axis=0)
        return k*np.var(Bin,ddof=1)/(2*np.var(data,ddof= 1))

    if method == "acf":
        N = len(data)
        data = data - np.mean(data)  # calculate variance around mean
        var = np.var(data, ddof=1)

        for search in range(100, N, N//100):
            acf = np.array([np.mean((data[:N-k]) * (data[k:])) for k in range(search)])
            if np.sum(acf < 0) > 0:
                break
        else:
            print("The data is too short!")
            return N

        acf /= var
        k_max = np.argmax(acf < 0)
        if k_max == 0:
            k_max = len(acf)

        return int(0.5 + np.sum(acf[1:k_max+1]))+1


def confidence_interval(data):
    return 1.96*np.std(data) / np.sqrt(len(data) - 1)

class Ising():
    def __init__(self, size:tuple[int,int],coupling:tuple[float,float,float,float]):
        """
        Construct the Ising object with given size and coupling parameters.
        Args:
            size (tuple[int, int]): Size of the lattice (Lx*Ly). it is advised to use square lattices.
            coupling (tuple[float, float, float, float]): Coupling parameters (t, J1, J2, J3).
        """
        self.L = int((size[0]*size[1])**0.5)
        self.t = coupling[0]

        self.model = cising.Ising(size[0], size[1])
        self.model.set_parameters(coupling[0], coupling[1], coupling[2], coupling[3])
        

    
    def run(self, Nsample:int, spacing:int,method:str):
        """
        Run the Ising model simulation with specified parameters.
        Args:
            Nsample (int): Number of samples to run.
            spacing (int): Spacing between samples.
            method (str): Method to use for simulation ('local' or 'cluster').
        """

        if method == "local":
            self.model.run_local(Nsample=Nsample, spacing=spacing*self.L)
        elif method == "cluster":
            self.model.run_cluster(Nsample=Nsample, spacing=spacing)
        else:
            raise ValueError("Method must be 'local' or 'cluster'.")

        
        print(f"The simulation for size {self.L}, temperature {self.model.temperature()} is done.")



    @property
    def size(self)-> tuple[int, int]:
        """
        Get the size of the Ising model.
        Returns:
            tuple[int, int]: Size of the lattice (Lx, Ly).
        """
        return self.model.size_x(), self.model.size_y()
    @property
    def couplings(self)-> tuple[float, float, float, float]:
        """
        Get the coupling parameters of the Ising model.
        Returns:
            tuple[float, float, float, float]: Coupling parameters (t, J1, J2, J3).
        """
        return (self.model.temperature(), self.model.J1(), self.model.J2(), self.model.J3())

    def __ensemble_avg(self,arr:np.ndarray,sep:int,f:Callable):
        res = []
        sep_arr = range(0, len(arr), (int(sep)+1)*6)  # create a range of indices to separate the array
        # first separate the array into parts

        for i in range(len(sep_arr)-1):
            res.append(f(arr[sep_arr[i]:sep_arr[i+1]]))
        res.append(f(arr[sep_arr[-1]:]))

        return np.array(res) # f should be a vectorized function


    def __update_rtime(self):
        self.e_rtime = rtime(self.model.get_energy())
        self.afm_rtime = rtime(self.model.get_afm())


    @property
    def afm(self):
        """
        Get the antiferromagnetic order parameter.
        Returns:
            result: Antiferromagnetic order parameter and its error
        """
        self.__update_rtime()

        afm = self.__ensemble_avg(np.abs(self.model.get_afm())/self.L**2, self.afm_rtime, lambda x: np.abs(np.mean(x)))
        return result(avg=np.mean(afm), err=confidence_interval(afm))

    @property
    def energy(self):
        """
        Get the energy of the Ising model.
        Returns:
            tuple: Energy and its error
        """
        self.__update_rtime()

        energy = self.__ensemble_avg(self.model.get_energy()/self.L**2, self.e_rtime, np.mean)
        return result(avg=np.mean(energy), err=confidence_interval(energy))

    @property
    def specific_heat(self):
        """
        Get the specific heat of the Ising model.
        Returns:
            tuple: Specific heat and its error
        """
        self.__update_rtime()
        # specific heat is defined as C = (1/T^2) * (avg(E**2) - avg(E)**2)

        specific_heat = self.__ensemble_avg(self.model.get_energy() / self.L, self.e_rtime, lambda arr: (np.mean(arr**2) - np.mean(arr)**2) / (self.t**2))
        return result(avg=np.mean(specific_heat), err=confidence_interval(specific_heat))

    @property
    def susceptibility(self):
        """
        Get the susceptibility of the Ising model.
        Returns:
            tuple: Susceptibility and its error
        """
        self.__update_rtime()
        # susceptibility is defined as chi = (1/T) * std(M)**2
        

        susceptibility = self.__ensemble_avg(np.abs(self.model.get_afm())/self.L, self.afm_rtime, lambda arr: (np.mean(arr**2) - np.mean(arr)**2) / self.t)
        return result(avg=np.mean(susceptibility), err=confidence_interval(susceptibility))

    @property
    def binder_ratio(self):
        """
        Get the Binder ratio of the Ising model.
        Returns:
            tuple: Binder ratio and its error
        """
        self.__update_rtime()
        # Binder ratio is defined as B = 1 - avg(M^4) / (3 * avg(M^2)^2)

        binder_ratio = self.__ensemble_avg(np.abs(self.model.get_afm())/self.L**2, self.afm_rtime, lambda arr: 1 - np.mean(arr**4) / (3 * np.mean(arr**2)**2))
        return result(avg=np.mean(binder_ratio), err=confidence_interval(binder_ratio))

