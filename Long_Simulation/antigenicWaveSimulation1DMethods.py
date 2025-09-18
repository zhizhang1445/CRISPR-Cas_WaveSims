from json import JSONDecodeError
import numpy as np
import multiprocessing
from scipy import sparse
from scipy.signal import convolve
from copy import deepcopy
import time
import os
import sys
import random

sys.path.insert(0, "../Scripts")
from coverage import elementwise_coverage, coverage_1D
from fitness import fitness, norm_fitness, phage_growth
from altImmunity import immunity_loss_uniform, immunity_gain_from_kernel
from immunity import immunity_mean_field_add, immunity_mean_field_remove
from initMethods import init_trail_nh, init_1D_kernel, init_guassian_n, init_cond
from supMethods import time_conv, write2json, read_json, load_last_output
from mutation import mutation
from randomHGT import HGT_logistic_event
from formulas import gaussian1D, semi_exact_nh
from initMethods import init_trail_nh, init_1D_kernel, init_guassian_n, init_cond

def make_paramslists1D(params, sim_params, sweep_params: str, list_to_sweep: list, n_seeds = 1, init = True, return_sweep = False, float_M = False):
    num_threads_set = 1
    foldername = sim_params["foldername"]

    sim_params["num_threads"] = 1
    sim_params["continue"] = False

    seed_list = random.sample(range(0, 255), n_seeds)
    params_list = []
    sim_params_list = []
    sweep_list = []

    print(f"Simulation to be done with Num of Threads: {num_threads_set} for Num of Seeds: {n_seeds} and Num of Points: {len(list_to_sweep)}")
    for i, sweep_itr in enumerate(list_to_sweep): 

        for seed_num, seed in enumerate(seed_list):
            params[sweep_params] = sweep_itr
            sim_params["seed"] = seed
            sim_params["foldername"] = foldername + f"/{sweep_params}_{sweep_itr}_seed{seed_num}"
            try:
                sim_params["hard_N0"] = False
                if init:
                    params, sim_params = init_cond(params, sim_params, False, False, float_M)

                params_list.append(deepcopy(params))
                sim_params_list.append(deepcopy(sim_params))

                sweep_list.append(sweep_itr)
            except ValueError:
                sim_params["hard_N0"] = True
                if init:
                    params, sim_params = init_cond(params, sim_params, False, False, float_M)

                params_list.append(deepcopy(params))
                sim_params_list.append(deepcopy(sim_params))

                continue

    if return_sweep:
        return sweep_list, params_list, sim_params_list
    
    return params_list, sim_params_list

def main(params, sim_params, normalize_f = True, reinit = True) -> int :
    np.random.seed(sim_params['seed'])
    foldername = sim_params["foldername"]

    if not os.path.exists(foldername):
        os.mkdir(foldername)

    if reinit:
        params, sim_params = init_cond(params, sim_params, False, False)
    try:
        write2json(foldername, params, sim_params)
    except FileNotFoundError:
        os.mkdir(foldername)
        write2json(foldername, params, sim_params)

    st1: float = time.time()
    n = init_guassian_n(params, sim_params)
    nh = init_trail_nh(params, sim_params)
    kernel_1D = init_1D_kernel(params, sim_params)

    if params["beta"] != 0:
        integration_kernel = init_1D_kernel(params, sim_params, "beta")
    ed = time.time()
        
    t = 0
    nh_total = params["Nh"]*params["M"]
    n_total = params["N"]
    uc = params["uc"]
    sigma = params["sigma"]
    M0 = params["M0"]

    with open(foldername+'/runtime_stats.txt','w') as file:
        file.write(f't: {t}| init_functions: {time_conv(ed-st1)}| Phage Population: {n_total:.4f}| Spacer Population: {nh_total:.4f}| Uc: {uc:.4f}| sigma: {sigma:.4f}| M0: {M0:.4f} \n')

    try:
        while(t < sim_params["tf"]):

            if (t%sim_params["dt_snapshot"] == 0):
                np.save(foldername+f"/frame_n{t}.npy", n)
                np.save(foldername+f"/frame_nh{t}.npy", nh)


            st1:float = time.time()
            p = coverage_1D(nh, kernel_1D, params, sim_params)
            st2 = time.time()
            f = fitness(n, p, params, sim_params)
            if normalize_f:
                f = norm_fitness(f, n, params, sim_params) #renormalize f
            n = phage_growth(n, f, params, sim_params, True) #update
                
            st3 = time.time()

            if (np.sum(n) <= 1) or (np.sum(n) >= (10)*params["N0"]) :
                st4 = time.time()
                with open(foldername+'/runtime_stats.txt','a') as file:
                    outstring = f"DEAD at: {t}| N: {np.sum(n)}| Coverage: {time_conv(st2-st1)}| Growth: {time_conv(st3-st2)}| Mutation: {time_conv(st3-st3)}| Immunity: {time_conv(ed-st4)} \n"
                    file.write(outstring)
                break

            n = mutation(n, params, sim_params)

            st4 = time.time()

            if params["beta"] != 0:
                integration_probability = convolve(nh, integration_kernel, mode="same")
                nh_temp = immunity_mean_field_add(nh, n, params, sim_params, int_prob=integration_probability)
            else:
                nh_temp = immunity_mean_field_add(nh, n, params, sim_params)
            nh = immunity_mean_field_remove(nh_temp, n, params, sim_params)

            diff_of_acquisition = 0
            ed = time.time()

            with open(foldername+'/runtime_stats.txt','a') as file:
                M = params["M"]
                outstring = f"t: {t}| N: {np.sum(n)}| Coverage: {time_conv(st2-st1)}| Growth: {time_conv(st3-st2)}| Mutation: {time_conv(st4-st3)}| Immunity: {time_conv(ed-st4)}| M: {M:.4f}| Net_Acq_Diff: {diff_of_acquisition:.4f} \n"
                file.write(outstring)

            t += sim_params["dt"]

    except KeyboardInterrupt or ValueError:
        write2json(foldername, params, sim_params)
        print(f"Stopped at time: {t}")
        return 0
    return 1