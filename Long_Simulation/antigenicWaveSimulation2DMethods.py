from json import JSONDecodeError
import numpy as np
import multiprocessing
from scipy import sparse
from copy import deepcopy
import time
import os
import sys
import random

sys.path.insert(0, "../Scripts")
from initMethods import init_cond, init_dict_kernel, init_exptail, init_quarter_kernel, init_guassian
from coverage import elementwise_coverage_vectorized, double_vectorized_coverage
from altImmunity import immunity_gain_from_kernel, immunity_loss_uniform
from immunity import immunity_mean_field_add, immunity_mean_field_remove
from fitness import virus_growth, norm_fitness, fitness_spacers
from mutation import mutation
from formulas import compute_shift
from supMethods import read_json, load_last_output, write2json, time_conv
from randomHGT import get_time_next_HGT, HGT_logistic_event

def make_paramslists(params, sim_params, sweep_params: str, list_to_sweep: list, n_seeds = 1):
    num_threads_set = False
    foldername = sim_params["foldername"]
    #the call is python3 antigenicWaveSimulation.py <num_cores> <num_seeds> <0 for restart or 1 for continue>

    if len(sys.argv) > 1:
        sim_params["num_threads"] = int(sys.argv[2])
        num_threads_set = True

    if len(sys.argv) > 2:
        n_seeds = int(sys.argv[2])

    if len(sys.argv) > 3:
        if int(sys.argv[3]) == 1:
            if os.path.isdir(foldername):
                print("Continuing where we left off")
                sim_params["continue"] = True

            else:
                os.mkdir(foldername)
                print("Created new folder: ", foldername)
                sim_params["continue"] = False
    
        elif int(sys.argv[3]) == 0:
            sim_params["continue"] = False
        else:
            ValueError("Error in arguments")

    if sim_params["num_threads"] == 0 or n_seeds == 0:
        raise ValueError("something wrong with the num threads or num seeds")

    seed_list = random.sample(range(0, 255), n_seeds)
    params_list = []
    sim_params_list = []

    num_cores = multiprocessing.cpu_count()
    if not num_threads_set:
        best_ratio = int(num_cores // (len(list_to_sweep)*n_seeds))
        num_cores_per_run = best_ratio if best_ratio >= 1 else 1
        sim_params["num_threads"] = num_cores_per_run
        print(f"Each Run is done with {num_cores_per_run} cores")

    print(f"Simulation to be done with Num of Threads: {num_threads_set} for Num of Seeds: {n_seeds} and Num of Points: {len(list_to_sweep)}")
    for i, sweep_itr in enumerate(list_to_sweep): 

        for seed_num, seed in enumerate(seed_list):
            params[sweep_params] = sweep_itr
            sim_params["seed"] = seed
            sim_params["foldername"] = foldername + f"/{sweep_params}_{sweep_itr}_seed{seed_num}"

            if not os.path.exists(sim_params["foldername"]):
                os.mkdir(sim_params["foldername"])

            params_list.append(deepcopy(params))
            sim_params_list.append(deepcopy(sim_params))
    return params_list, sim_params_list

def main(params, sim_params, normalize_f = True) -> int :
    np.random.seed(sim_params['seed'])
    foldername = sim_params["foldername"]
    shift_vector = [0,0] #This is unused if dt_exact_fitness is zero

    if sim_params["continue"]:
        try:
            st1 = time.time()
            params, sim_params = read_json(foldername)

            if params["beta"] != 0:
                kernel_acq_dict = init_dict_kernel(params, sim_params, "beta")

            kernel_dict = init_dict_kernel(params, sim_params)
            kernel_immunity = init_quarter_kernel(params, sim_params, type="Boltzmann")
            t, n, nh = load_last_output(foldername)
            nh_total = params["Nh"]
            n_total = params["N"]
            uc = params["uc"]
            sigma = params["sigma"]
            M0 = params["M0"]

            with open(foldername+'/runtime_stats.txt','a') as file:
                file.write(f't: {t}| Restarted  | Phage Population: {n_total:.4f}| Spacer Population: {nh_total:.4f}| Uc: {uc:.4f}| sigma: {sigma:.4f}| M: {M0:.4f} \n')

        except FileNotFoundError or (nh is None): #the folders were empty so better restart everything
            sim_params["continue"] = False


    if not sim_params["continue"]: #not an else statement because this also serves as error catch for the if statement
        params, sim_params = init_cond(params, sim_params)
        try:
            write2json(foldername, params, sim_params)
        except FileNotFoundError:
            os.mkdir(foldername)
            write2json(foldername, params, sim_params)

        st1: float = time.time()
        n = init_guassian(params["N"], sim_params, "n")
        nh = init_exptail(params["Nh"]*params["M0"], params, sim_params, "nh")

        if params["beta"] != 0:
            kernel_acq_dict = init_dict_kernel(params, sim_params, "beta")
        kernel_dict= init_dict_kernel(params, sim_params)
        kernel_immunity = init_quarter_kernel(params, sim_params, type="Boltzmann")
        ed = time.time()
            
        t = 0
        nh_total = params["Nh"]
        n_total = params["N"]
        uc = params["uc"]
        sigma = params["sigma"]
        M0 = params["M0"]

        with open(foldername+'/runtime_stats.txt','w') as file:
            file.write(f't: {t}| init_functions: {time_conv(ed-st1)}| Phage Population: {n_total:.4f}| Spacer Population: {nh_total:.4f}| Uc: {uc:.4f}| sigma: {sigma:.4f}| M0: {M0:.4f} \n')

    try:
        while(t < sim_params["tf"]):

            if t%sim_params["dt_snapshot"] == 0:
                sparse.save_npz(foldername+f"/sp_frame_n{t}",n.tocoo())
                sparse.save_npz(foldername+f"/sp_frame_nh{t}",nh.tocoo())

            st1:float = time.time()
            if params["beta"] != 0:
                p, int_prob = double_vectorized_coverage(nh, n, kernel_dict, kernel_acq_dict, params, sim_params)
            else:
                int_prob = None
                p = elementwise_coverage_vectorized(nh, n, kernel_dict, params, sim_params)

            st2 = time.time()
            f = fitness_spacers(n, nh, p, params, sim_params)
            sparse.save_npz(foldername+f"/sp_frame_f{t}", f.tocoo())

            if normalize_f:
                f = norm_fitness(f, n, params, sim_params) #renormalize f

            n = virus_growth(n, f, params, sim_params) #update
            
            if (np.sum(n) <= 0) or (np.sum(n) >= (2)*np.sum(nh)):
                with open(foldername+'/runtime_stats.txt','a') as file:
                    outstring = f"DEAD at: {t}| N: {np.sum(n)}| Coverage: {time_conv(st2-st1)}| Growth: {time_conv(st3-st2)}| Mutation: {time_conv(st4-st3)}| Immunity: {time_conv(ed-st4)} \n"
                    file.write(outstring)
                return 1

            st3 = time.time()
            n = mutation(n, params, sim_params)

            st4 = time.time()
            nh_prev = nh

            params, sim_params, num_to_add, num_to_remove = HGT_logistic_event(t, n, params, sim_params)
            nh_gain = immunity_mean_field_add(nh, n, params, sim_params, num_to_add, int_prob = int_prob) #update nh
            nh = immunity_mean_field_remove(nh_gain, n, params, sim_params, num_to_remove)
            
            ed = time.time()

            with open(foldername+'/runtime_stats.txt','a') as file:
                M = params["M"]
                outstring = f"t: {t}| N: {np.sum(n)}| Coverage: {time_conv(st2-st1)}| Growth: {time_conv(st3-st2)}| Mutation: {time_conv(st4-st3)}| Immunity: {time_conv(ed-st4)}| M: {M:.4f} \n"
                file.write(outstring)

            t += sim_params["dt"]

    except KeyboardInterrupt or ValueError:
        write2json(foldername, params, sim_params)
        print(f"Stopped at time: {t}")
        return 0
    return 1