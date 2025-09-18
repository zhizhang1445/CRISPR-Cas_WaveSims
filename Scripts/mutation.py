import numpy as np
import numpy.ma as ma
import scipy
from scipy.ndimage import convolve
from scipy import signal
from scipy.sparse import issparse
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from concurrent.futures import as_completed
from supMethods import sum_parallel, timeit

def num_mutants_per_loc_2D(n, params, sim_params): #TODO PARALLELIZE THIS
    mu = params["mu"]
    dt = sim_params["dt"]

    x_ind, y_ind = np.nonzero(n)
    map = np.zeros(n.shape, dtype=np.int32)

    p = 1-np.exp(-1*mu*dt)
    if scipy.sparse.issparse(n):
        map[x_ind, y_ind] = np.random.binomial(n[x_ind, y_ind].todense(), p) #   so this is really prob of finding k mutation in n possible virus with prob p in 1-e^-mudt
    else:
        map[x_ind, y_ind] = np.random.binomial(n[x_ind, y_ind], p)
    
    return map
                                    #   so p to not mutated is really e^-mu*dt

def num_mutants_per_loc_1D(n, params, sim_params): #TODO PARALLELIZE THIS
    mu = params["mu"]
    dt = sim_params["dt"]

    x_ind = np.nonzero(n)[0]
    map = np.zeros(n.shape, dtype=int)

    p = 1-np.exp(-1*mu*dt)
    if scipy.sparse.issparse(n):
        map[x_ind] = np.random.binomial(n[x_ind], p) #   so this is really prob of finding k mutation in n possible virus with prob p in 1-e^-mudt
    else:
        map[x_ind] = np.random.binomial(n[x_ind], p)
    
    return map

def num_mutation(params, sim_params): #Sparse is not needed
    mu = params["mu"]
    dt = sim_params["dt"]

    out = np.random.poisson(mu*dt)
    if out >= 1 :
        return out #This is what mu, is the average rate of mutation
    else:
        try:
            return num_mutation(params, sim_params) #conditioned as to have at least one mutation
        except RecursionError:
            return 0

def mutation_jump1D(m, params, sim_params):
    shape_param = params["gamma_shape"]
    dx = sim_params["dx"]

    jump = 0
    mean = 2*dx
    theta = mean/shape_param
    
    for i in range(m):
        angle = np.random.choice([-1, 1]) #Goodammit couldn't this have been clearer??? It's supposed to be isotropic
        jump = jump + np.random.gamma(shape_param, theta)*angle

    jump = np.round(jump)
    return jump

def mutation_jump2D(m, params, sim_params):
    shape_param = params["gamma_shape"]
    dx = sim_params["dx"]

    jump = np.zeros(2)
    mean = 2*dx
    theta = mean/shape_param
    
    for i in range(m):
        angle = np.random.uniform(0, 2*np.pi) #Goodammit couldn't this have been clearer??? It's supposed to be isotropic
        jump = jump + np.random.gamma(shape_param, theta)*np.array([np.cos(angle), np.sin(angle)])
        #The distribution of jump is a sum of gamma distribution. 

    jump = np.round(jump)
    return jump

def mutation1D(n, params, sim_params):
    mutation_map = num_mutants_per_loc_1D(n, params, sim_params) # The mutation maps tells you how many virus have mutated at each location
    x_ind = np.nonzero(mutation_map)[0]

    n_to_add = np.zeros_like(n)
    for x_i in x_ind:
        num_mutants_at_site = mutation_map[x_i]
        n_to_add[x_i] -= num_mutants_at_site
        for j in range(num_mutants_at_site):
            num_mutation_at_site = num_mutation(params, sim_params)
            jump = mutation_jump1D(num_mutation_at_site, params, sim_params)

            try:
                new_x_loc = (x_i + jump).astype(int)
                n_to_add[new_x_loc] += 1
            except IndexError: #Array Out of Bounds
                if new_x_loc < 0:
                    new_x_loc = 0

                if new_x_loc >= len(n):
                    new_x_loc = len(n)-1 #lmao this is gonna be a pain in cpp
                n_to_add[new_x_loc] += 1

    return n+n_to_add

def mutation2D(n, params, sim_params):
    num_threads = sim_params["num_threads"]
    checksum = np.sum(n)

    mutation_map = num_mutants_per_loc_2D(n, params, sim_params) # The mutation maps tells you how many virus have mutated at each location
    x_ind, y_ind = np.nonzero(mutation_map) #finding out where the mutations happends

    x_ind_subsets = np.array_split(x_ind, num_threads)
    y_ind_subsets = np.array_split(y_ind, num_threads)

    def mutation_single(x_ind, y_ind):
        n_to_add = scipy.sparse.dok_matrix(n.shape, dtype=np.int64)
        for x_i, y_i in zip(x_ind, y_ind):
            num_mutants_at_site = mutation_map[x_i, y_i] # unpacking the number of mutated virus
            n_to_add[x_i, y_i] -= num_mutants_at_site

            for j in range(num_mutants_at_site): #Find out where those virus have moved to
                num_mutation_at_site = num_mutation(params, sim_params) #Sampling how many single mutation for a single virus (num of jumps) 
                jump = mutation_jump2D(num_mutation_at_site, params, sim_params) #Sampling the jump distance

                new_x_loc = (x_i + jump[0]).astype(int)
                new_y_loc = (y_i + jump[1]).astype(int)

                if new_x_loc < 0:
                    new_x_loc = 0
                if new_y_loc < 0:
                    new_y_loc = 0

                if new_x_loc >= n.shape[0]:
                    new_x_loc = n.shape[0]-1
                if new_y_loc >= n.shape[1]:
                    new_y_loc = n.shape[1]-1
                
                n_to_add[new_x_loc, new_y_loc] += 1

        return n_to_add

    results = Parallel(n_jobs=num_threads, backend='threading')(delayed(mutation_single)(x_ind, y_ind) for x_ind, y_ind in zip(x_ind_subsets, y_ind_subsets))
    # results = mutation_single(x_ind, y_ind)
    # n = n+results

    n = n + sum_parallel(results, num_threads)
    if checksum != np.sum(n):
        raise ValueError("Cries cuz Bacteria died during mutation")
    return n

def mutation(n, params, sim_params):
    ndim = sim_params["ndim"]
    if ndim == 1 and isinstance(n, np.ndarray):
        return mutation1D(n, params, sim_params)
    elif ndim == 2 and issparse(n):
        return mutation2D(n, params, sim_params)
    else:
        raise TypeError(f"Something went wrong with Mutation| n_dim: {ndim} but type is {type(n)}")