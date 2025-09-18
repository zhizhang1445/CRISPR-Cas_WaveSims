import numpy as np
import numpy.ma as ma
import scipy
from scipy.ndimage import convolve
from scipy import signal
from scipy.sparse import issparse
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from supMethods import timeit
from concurrent.futures import as_completed
from copy import deepcopy
from formulas import p_infection, binomial_pdf, gaussian1D, theoretical_c, fitness_simple_M

def fitness(n, p, params, sim_params):
    ndim = sim_params["ndim"]
    if ndim == 1 and isinstance(p, np.ndarray):
        return fitness_1D(p, params, sim_params)
    elif ndim == 2 and issparse(n):
        return fitness_2D(n, p, params, sim_params)
    else:
        raise TypeError(f"Something went wrong with fitness n_dim: {ndim} but type is {type(n)}")

def fitness_2D(n, p_sparse, params, sim_params): #TODO PARALLELIZE THIS
    R0 = params["R0"]
    res = scipy.sparse.dok_matrix(n.shape, dtype=float)

    x_ind, y_ind = np.nonzero(p_sparse) #also == np.nonzero(n)
    p_dense = np.array(p_sparse[x_ind, y_ind].todense()).squeeze()

    if np.sum(p_dense) == 0:
        raise ValueError("Zero spacer probability")
    
    p_tt = p_infection(p_dense, params, sim_params)

    if np.min(p_tt) < 0:
        raise ValueError("Negative Probability")
        
    res = scipy.sparse.dok_matrix(n.shape, dtype=float)
    res[x_ind, y_ind] = np.log(R0*p_tt)
    return res

def fitness_1D(p_coverage, params, sim_params):
    R0 = params["R0"]

    p_inf = p_infection(p_coverage, params, sim_params)
    fit = np.log(R0*p_inf)
    return fit

def norm_fitness(f, n, params, sim_params, return_avg = False):
    ndim = sim_params["ndim"]
    if ndim == 1 and isinstance(f, np.ndarray):
        f_avg = np.sum(f*n)/np.sum(n)
        new_f = f-f_avg
        if return_avg:
            return new_f, f_avg

        return new_f
    elif ndim == 2 and issparse(f):
        return norm_fitness_2D(f, n, params, sim_params, return_avg=return_avg)
    else:
        raise TypeError(f"Something went wrong with Norm_F| n_dim: {ndim} but type is {type(n)}")

def norm_fitness_2D(f_sparse, n, params, sim_params, return_avg = False):
    f_avg = np.sum(f_sparse.multiply(n))/np.sum(n)

    x_ind, y_ind = f_sparse.nonzero()
    new_f = scipy.sparse.dok_matrix(f_sparse.shape, dtype=float)
    new_f[x_ind, y_ind] = f_sparse[x_ind, y_ind].toarray() - f_avg
    if return_avg:
        return new_f, f_avg
    else:
        return new_f

def phage_growth(n, f, params, sim_params, det_growth = False):
    ndim = sim_params["ndim"]
    if ndim == 1 and isinstance(f, np.ndarray):
        return phage_growth_1D(n, f, params, sim_params, det_growth)
    elif ndim == 2 and issparse(f):
        return phage_growth_2D(n, f, params, sim_params, det_growth)
    else:
        raise TypeError(f"Something went wrong with Growth| n_dim: {ndim} but type is {type(n)}")

def phage_growth_1D(n, f, params, sim_params, det_growth = False):
    dt = sim_params["dt"]

    # n_new = np.zeros_like(n, dtype=int)
    if not det_growth:
        mean = np.clip((1+f*dt), a_min = 0, a_max=None)*n
        n_new = np.rint(np.random.poisson(mean)).astype(int)
    else:
        n_new = np.rint(np.clip(n + f*n, a_min=0, a_max=None)).astype(int)
    return n_new

def phage_growth_2D(n, f_sparse, params, sim_params, deterministric_growth = False): #TODO PARALLELIZE THIS
    dt = sim_params["dt"]
    x_ind, y_ind = n.nonzero()
    if scipy.sparse.issparse(f_sparse):
        f_dense = np.array(f_sparse[x_ind, y_ind].todense())
    else:
        f_dense = f_sparse[x_ind, y_ind]

    if scipy.sparse.issparse(n):
        n_dense = np.array(n[x_ind, y_ind].todense())
        n_new = scipy.sparse.dok_matrix(n.shape, dtype = int)
    else:
        n_dense = n[x_ind, y_ind]
        n_new = np.zeros_like(n)

    if not deterministric_growth:
        mean = np.clip((1+f_dense*dt), a_min = 0, a_max=None)*n_dense
        n_new[x_ind, y_ind] = np.random.poisson(mean)
    else:
        n_new[x_ind, y_ind] = np.clip(n_dense + f_dense*n_dense, a_min=0, a_max=None)
    return  n_new

def fitness_n_spacers(p_coverage, params, sim_params):
    M = params["M"]
    R0 = params["R0"]
    Np = params["Np"]
    dc = params["dc"]

    def p_n_infection(p_coverage, M, Np, dc):
        p_infection = (1-p_coverage)**M

        for n in range(1, M+1):
            p_n_spacer = binomial_pdf(M, n, p_coverage)
            for d in range(0, dc+1):
                p_infection += binomial_pdf(Np, d, n/M)*p_n_spacer
        return p_infection
    
    p_inf = p_n_infection(p_coverage, M, Np, dc)
    return np.log(R0*p_inf)

def derivative_p_infection(p_coverage, params, sim_params):
    M = params["M"]
    R0 = params["R0"]
    Np = params["Np"]
    dc = params["dc"]
    n_order_spacer = params["n_spacer"]
    if n_order_spacer > M:
        n_order_spacer = M

    derivative_p_infection = 0
    for n in range(0, n_order_spacer+1):
        derivative_p_n_spacer = n*(binomial_pdf(M, n, p_coverage)/p_coverage)
        derivative_p_n_spacer -= (M-n)*(binomial_pdf(M, n, p_coverage)/(1-p_coverage))

        for d in range(0, dc+1):
            derivative_p_infection += binomial_pdf(Np, d, n/M)*derivative_p_n_spacer
    return derivative_p_infection

def derivative_fitness(p_coverage, params, sim_params):
    p_inf = p_infection(p_coverage, params, sim_params)
    derivative_p_inf = derivative_p_infection(p_coverage, params, sim_params)
    derivative_fit = (1/p_inf)*derivative_p_inf
    return derivative_fit

def find_root_fitness(params, sim_params, n_itr = 1000, err = 1e-7, to_print = False):
    x_old = 0.5

    for _ in range(n_itr):
        f0 = fitness_1D(x_old, params, sim_params)
        if to_print:
            print("New Root: ", x_old,"|  New Fitness:  ", f0)
        f_prime = derivative_fitness(x_old, params, sim_params)
        
        x_new = x_old - (f0/f_prime)
        if np.abs(x_new - x_old) < err:
            break
        x_old = x_new

    return x_new

def memory_after_event(t, params, sim_params):
    t_HGT = 1/params["rate_HGT"]
    t_m = 1/params["rate_recovery"]
    alpha = params["HGT_bonus_acq_ratio"]
    M_t = params["M0"] + alpha*(params["N0"]/params["Nh"])*np.exp(-t/t_m)
    return M_t

def del_f_impact(t, params, sim_params):
    sigma_n = params["sigma"]
    x= np.arange(-3*sigma_n, 3*sigma_n, 0.1)

    n = gaussian1D(x, 0, params, sim_params)
    ind = np.where(n>=1)
    c_0 = theoretical_c(0, 0, params, sim_params)
    a = params["HGT_bonus_acq_ratio"]
    r = params["r"]
    v = params["v0"]
    tau = params["tau"]
    A = params["A"]

    der_f = derivative_fitness(c_0, params, sim_params)
    impact = ((r*a)/(v*tau))*np.exp(-np.abs(v*t)/r)
    return der_f*impact

def f0_memory_variation(t, params, sim_params):
    M_range = memory_after_event(t, params, sim_params)

    sigma_n = params["sigma"]
    x= np.arange(-3*sigma_n, 3*sigma_n, 0.1)

    n = gaussian1D(x, 0, params, sim_params)
    ind = np.where(n>=1)
    c_0 = theoretical_c(0, 0, params, sim_params)

    # print(c_0)
    f_range = []
    for M in M_range:
        temp_params = deepcopy(params)
        temp_params["M"] = M
        f_range.append(fitness_simple_M(c_0, params, sim_params, M))

    return np.array(f_range)

def del_f_minus(t, params, sim_params):
    t_HGT = 1/params["rate_HGT"]
    t_m = 1/params["rate_recovery"]
    a = params["HGT_bonus_acq_ratio"]
    tau = params["tau"]
    # a = params["HGT_bonus_acq_ratio"]
    r = params["r"]
    v = params["v0"]

    r = params["r"]
    v = params["v0"]

    sigma_n = params["sigma"]
    x= np.arange(-3*sigma_n, 3*sigma_n, 0.1)
    n = gaussian1D(x, 0, params, sim_params)
    ind = np.where(n>=1)
    c_0 = theoretical_c(0, 0, params, sim_params)
    
    der_f = derivative_fitness(c_0, params, sim_params)

    # t_minus = 1/((-a/tau**2)+(1/tau))
    t_minus = 1/(((1/tau)-1)*(1-(a/tau))+1)
    omega_r = ((r*a)/(v*tau))*np.exp(-np.abs(v*t)/r)
    del_c = (t_minus - tau)*t
    res = del_c
    res = del_c*omega_r*der_f/50
    return res

def del_f_plus(t, params, sim_params):
    t_HGT = 1/params["rate_HGT"]
    t_m = 1/params["rate_recovery"]
    a = params["HGT_bonus_acq_ratio"]
    tau = params["tau"]
    # a = params["HGT_bonus_acq_ratio"]
    r = params["r"]
    v = params["v0"]

    sigma_n = params["sigma"]
    x= np.arange(-3*sigma_n, 3*sigma_n, 0.1)
    n = gaussian1D(x, 0, params, sim_params)
    ind = np.where(n>=1)
    c_0 = theoretical_c(0, 0, params, sim_params)
    
    der_f = derivative_fitness(c_0, params, sim_params)
    omega_r = ((r*a)/(v*tau))*np.exp(-np.abs(v*t)/r)
    t_plus = 1/((-a/tau**2)+(1/tau))
    
    del_c = tau*t - a*t_plus*np.exp(-t/t_m)
    del_c = tau*t - a*tau*np.exp(-t/t_m)
    # res = del_c
    res = del_c*omega_r*der_f/(50*12) + 1
    return res

def fitness_versus_M(params, sim_params):
    def weighted_avg_and_var(f, n):
        # Compute weighted average
        weighted_avg = np.sum(f * n) / np.sum(n)
        return weighted_avg

    sigma_n = params["sigma"]

    x = np.arange(-3*sigma_n, 3*sigma_n, 0.1)
    current_M = params["M"]
    n_order_spacers = params["n_spacer"]

    if n_order_spacers > 1:
        M_range = np.arange(1, 2*current_M, 1).astype(int)
    else:
        M_range = np.arange(1, 2*current_M, 0.2)

    n = gaussian1D(x, 0, params, sim_params)
    ind = np.where(n>=1)
    c = theoretical_c(x, 0, params, sim_params)
    c_restricted = c[ind]

    f = fitness_simple_M(c_restricted, params, sim_params)
    n_restricted = n[ind]
    # avg_f = weighted_avg_and_var(f, n_restricted)

    avg_fitness = []
    err_fitness = []

    for M in M_range:
        f = fitness_simple_M(c_restricted, params, sim_params, M)
        avg_fitness.append(weighted_avg_and_var(f, n_restricted))

    return M_range, avg_fitness