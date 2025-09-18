import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

# from formulas import find_max_value_location
from trajectory import fit_GMM
from supMethods import load_outputs, read_json
from fitness import norm_fitness, phage_growth
from mutation import mutation

def compute_entropy(array, dim = 2, k=1):
    if dim == 2:    
        x_ind, y_ind = np.nonzero(array)    
        
        if scipy.sparse.issparse(array):
            non_zero_values = array[x_ind, y_ind].toarray().squeeze()
        else:
            non_zero_values = array[x_ind, y_ind]
    
    elif dim == 1:
        x_ind = np.nonzero(array)    
        
        if scipy.sparse.issparse(array):
            non_zero_values = array[x_ind].toarray().squeeze()
        else:
            non_zero_values = array[x_ind]

    entropy = k*np.sum(non_zero_values*np.log(non_zero_values))
    return (-1)*entropy

def compute_fitness_variance(fitness, n, params, sim_params):
    if scipy.sparse.issparse(fitness) and scipy.sparse.issparse(n):
        var_f_n = np.sum(np.power(fitness.toarray(), 2)*n.toarray())
    else:
        raise NotImplementedError("Not implemented for non sparse fitness variance")
    return var_f_n*sim_params["dt"]

def compute_entropy_production_Gaussian(var, Diff_const, t, N = 1, dim = 2):
    term = 2*dim*Diff_const/(2*var + 4*Diff_const*t)
    return N*term

def compute_entropy_production_2D_mutation(n_old, params, sim_params):
    _, cov, count = fit_GMM(n_old, params, sim_params)

    eigval, _ = np.linalg.eigh(cov.squeeze())
    Diff_const = params["D"]
    entropy_mutation = compute_entropy_production_Gaussian(eigval[0], Diff_const, 1, N = count, dim = 1) \
                        + compute_entropy_production_Gaussian(eigval[1], Diff_const, 1, N = count, dim = 1)
    return entropy_mutation

def compute_entropy_Gaussian(var, Diff_const, t, N = 1, dim =2):
    size_cor = N*np.log(N) if N !=1 else 0 
    return -size_cor + N*(dim/2)*(1+np.log((2*var+4*Diff_const*t)*np.pi))

def compute_entropy_change(n_new, n_old, dim=2, dt=1):
    return (compute_entropy(n_new, dim) - compute_entropy(n_old, dim))*dt

def get_entropy_change(t_domain, foldername, to_plot = True, to_save_folder = None):
    entropy_change_time = []
    entropy_change_mutation_time = []
    entropy_change_growth_time = []
    entropy_change_remainder_time = []
    fitness_time = []
    t_range = []

    n_old, nh_old, f_old = load_outputs(foldername, t_domain[0], True)
    params, sim_params = read_json(foldername)

    for t in t_domain:
        n_new, nh_new, f_new = load_outputs(foldername, t, True)
        f_norm = norm_fitness(f_new, n_old, params, sim_params)

        x_ind, y_ind = f_norm.nonzero()
        mean_f = np.mean(f_norm[x_ind, y_ind])
        fitness_time.append(mean_f)

        n_intermediate = phage_growth(n_old, f_norm, params, sim_params)
        n_mutated = mutation(n_intermediate, params, sim_params)

        entropy_change_growth = compute_entropy_change(n_intermediate, n_old)
        entropy_change_remainder = compute_entropy_change(n_new, n_intermediate)
        entropy_change_mutation = compute_entropy_change(n_mutated, n_intermediate)
        entropy_change = compute_entropy_change(n_new, n_old)

        entropy_change_time.append(entropy_change)
        entropy_change_mutation_time.append(entropy_change_mutation)
        entropy_change_growth_time.append(entropy_change_growth)
        entropy_change_remainder_time.append(entropy_change_remainder)
        t_range.append(t)
        n_old = n_new

    if to_plot:
        plt.figure()
        plt.title("Entropy Production")
        plt.plot(t_range[1:], entropy_change_time[1:], alpha = 0.2, label = "Actual Entropy")

        plt.plot(t_range[1:], entropy_change_mutation_time[1:], alpha = 0.5, label = "Mutation")
        plt.plot(t_range[1:], entropy_change_growth_time[1:], alpha = 0.5, label = "Growth")

        plt.title("Entropy Production")
        plt.ylabel("Entropy Change[1]")
        plt.xlabel("Time")
        plt.legend()

        if to_save_folder is not None:
            np.save(to_save_folder + "/entropy_total.npy", entropy_change_time)
            np.save(to_save_folder + "/entropy_mutation.npy", entropy_change_mutation_time)
            np.save(to_save_folder + "/entropy_growth.npy", entropy_change_growth_time)
            np.save(to_save_folder + "/fitness.npy", fitness_time)
            plt.savefig(to_save_folder + "/entropy.png")
            plt.close("all")

    return entropy_change_time, entropy_change_mutation_time, entropy_change_growth_time, fitness_time


