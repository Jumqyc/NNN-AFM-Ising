# import ising_wrapper as ising
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
from collections import namedtuple
result = namedtuple("result", ["avg", "err"])


def read_data(filename):
    with open(filename, "rb") as f:
        model_list = pkl.load(f)
    # Now close the file
    f.close()
    return model_list


def plot_result(filename: str, quantity: str,scaling: str = "False"):
    model_list = read_data(filename)

    L = model_list[0].size[0]
    temperatures = []
    data = []
    data_err = []
    for model in model_list:
        result = getattr(model, quantity)

        t = model.couplings[0]
        L = model.size[0]


        # codes for finite scale scaling

        if scaling == "True":
            temperatures.append((t-2.896)*L)
            data.append(result.avg*L**0.125)
            data_err.append(result.err*L**0.125)
        else:
            temperatures.append(t)
            data.append(result.avg)
            data_err.append(result.err)

    # for different methods, we use different markers, however, we use the same color for the same L
    color_map = {
        16: 'blue',
        20: 'green',
        32: 'brown',
        48: 'red'
    }
    if 'cluster' in filename:
        method = "Cluster"
        plt.errorbar(temperatures, data,yerr = data_err, label=method+", L="+str(L), fmt='x',lw=0.2, color=color_map[L],markersize = 2.5,capsize=2)
    else:
        method = "Local"
        plt.errorbar(temperatures, data,yerr = data_err, label=method+", L="+str(L), fmt='o',lw=0.2, color=color_map[L],markersize = 2.5,capsize=2)

for filename in os.listdir("data"):
    if filename.endswith("pkl"):

        plt.subplot(2,3,1)
        plot_result(os.path.join("data", filename), quantity="afm", scaling="True")
        plt.title("Finite size scaling")
        plt.xlabel("$(T-T_c)L^{1}$")
        plt.ylabel("$M_{\mathrm{AFM}}L^{1/8}$")

        plt.subplot(2,3,4)
        plot_result(os.path.join("data", filename), quantity="binder_ratio")
        plt.title("Binder ratio")
        plt.xlabel("Temperature")

        plt.subplot(2,3,2)
        plot_result(os.path.join("data", filename), quantity="afm")
        plt.title("AFM order parameter")
        plt.xticks([])

        plt.subplot(2,3,5)
        plot_result(os.path.join("data", filename), quantity="energy")
        plt.title("Energy")

        plt.subplot(2,3,3)
        plot_result(os.path.join("data", filename), quantity="susceptibility")
        plt.title("Susceptibility")
        plt.xticks([])

        plt.subplot(2,3,6)
        plot_result(os.path.join("data", filename), quantity="specific_heat")
        plt.title("Specific heat")



plt.legend()
plt.show()
