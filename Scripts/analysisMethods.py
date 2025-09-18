from ast import Raise
from multiprocessing import Value
import numpy as np
import matplotlib.pyplot as plt
import scipy
import json
import matplotlib.pyplot as plt
import sys
import os
import re
from scipy.optimize import curve_fit


from supMethods import load_last_output, read_json
from trajsTree import make_Treelist, link_Treelists, save_Treelist
from trajectory import fit_GMM_unknown_components, get_nonzero_w_repeats, fit_unknown_GMM, fit_GMM
from trajectoryVisual import make_frame, make_Gif, plot_Ellipses
from plotStuff import get_var_single, get_count_single, plot_velocity_single, from_all_root
from entropy import get_entropy_change
from initMethods import init_cond

def get_tdomain(foldername, to_plot= True, t0 = 0, margins = (-0.4, -0.4), dt = 0):
    params, sim_params = read_json(foldername)

    t0 = 0
    if dt == 0 and (sim_params is not None):
        try:
            dt = sim_params["dt_snapshot"]
        except KeyError:
            dt = sim_params["t_snapshot"]
    elif sim_params is None:
        raise ValueError

    dim = sim_params["ndim"]
    tf, n_final, nh_final = load_last_output(foldername, dim = dim)
    t_domain = np.arange(t0, tf, dt)
    xdomain = np.arange(-sim_params["xdomain"], sim_params["xdomain"], sim_params["dx"])

    if to_plot:
        if dim == 2:
            plt.figure()
            plt.contour(n_final.toarray().transpose(), cmap = "Reds")
            plt.contour(nh_final.toarray().transpose(), cmap = "Blues")
            plt.margins(margins[0], margins[1])
            plt.show()
        
        else:
            plt.figure()
            plt.plot(xdomain, n_final, color = "Red")
            plt.plot(xdomain, nh_final, color = "Blue")

            ind_middle = np.argmax(n_final)
            x_middle = xdomain[ind_middle]
            sigma = params["sigma"]
            plt.xlim(x_middle-5*sigma, x_middle+5*sigma)
            plt.show()
    return t_domain

def gaussian(x, a, x0, sigma):
    """Gaussian function."""
    normalization = 1/(sigma*np.sqrt(2*np.pi))
    return a *normalization*np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def fit_gaussian(xdomain, data):
    # Generate an x array that matches the indices of the data
    x = xdomain

    # Initial guesses for the parameters: amplitude (max of data), mean, and std deviation
    a_guess = np.max(data)
    x0_guess = np.sum(x * data) / np.sum(data)
    sigma_guess = np.sqrt(np.sum(data * (x - x0_guess) ** 2) / np.sum(data))

    # Fit the Gaussian model to the data
    popt, _ = curve_fit(gaussian, x, data, p0=[a_guess, x0_guess, sigma_guess])

    return popt

def create_Tree(t_domain, foldername, GMM_flag = 0):
    params, sim_params = read_json(foldername)
    init_list = []

    for t in t_domain:
        try:
            n_i = scipy.sparse.load_npz(foldername+f"/sp_frame_n{t}.npz").todok()
            indexes = get_nonzero_w_repeats(n_i)
            if GMM_flag == 0:
                means, covs, counts = fit_GMM_unknown_components(n_i, params, sim_params, indexes, scale = np.sqrt(2))
            else:
                means, covs, counts = fit_GMM(n_i, params, sim_params, n_components = GMM_flag)

            next_list = make_Treelist(t, means, covs, counts)
            if t == t_domain[0]:
                init_list = next_list
                prev_list = next_list
                continue

            prev_list = link_Treelists(prev_list, next_list)
            prev_list = next_list
        except ValueError:
            print(f"Failure to find GMM at t = {t}")
            break
    else:
        save_Treelist(foldername, init_list)
    return init_list

def load_data_from_superfolder(superfolder, params_to_sweep,list_to_sweep):
    subfolders = [superfolder + f"{params_to_sweep}_{i}_seed0" for i in list_to_sweep]

    M_data = []
    mean_data = []
    error_data = []

    for folder in subfolders:

        params, sim_params = read_json(folder)
        M_data.append(params[params_to_sweep])
        x_lim = sim_params["xdomain"]
        dx = sim_params["dx"]
        x_range = np.arange(-x_lim, x_lim, dx)

        N_time = []
        var_time = []
        x_time = []

        for t in range(sim_params["tf"]):
            try:
                # nh = np.load(folder + f"/frame_nh{t}.npy")
                n = np.load(folder + f"/frame_n{t}.npy")
            except FileNotFoundError:
                break
            popt = fit_gaussian(x_range, n)
            N_time.append(popt[0])
            var_time.append(popt[2])
            x_time.append(popt[1])

        velocity = np.diff(x_time)/sim_params["dt"]
        means = [np.mean(N_time), np.mean(var_time), np.mean(velocity)]
        mean_data.append(means)

        errors = [np.std(N_time), np.std(var_time), np.std(velocity)]
        error_data.append(errors)

    M_data = np.array(M_data)
    mean_data = np.array(mean_data)
    error_data = np.array(error_data)
    return M_data, mean_data, error_data

def create_both_Gifs(t_domain, foldername, margins, GMM_flag = 0):
    t_domain_no_error = []
    init_list = []

    for t in t_domain:
        make_frame(foldername, t, save = True, margins=margins)
        t_domain_no_error.append(t)
        plt.close("all")

    make_Gif(foldername, t_domain_no_error, typename = "time_plots")
    params, sim_params = read_json(foldername)
    print("time plots made for ", foldername)

    t_domain_no_error = []
    for t in t_domain:
        try:
            n_i = scipy.sparse.load_npz(foldername+f"/sp_frame_n{t}.npz").todok()
            indexes = get_nonzero_w_repeats(n_i)

            if GMM_flag == 0:
                means, covs, counts = fit_GMM_unknown_components(n_i, params, sim_params, indexes, scale = np.sqrt(2))
            else:
                means, covs, counts = fit_GMM(n_i, params, sim_params, n_components = GMM_flag)

            next_list = make_Treelist(t, means, covs, counts)
            
            if t == t_domain[0]:
                init_list = next_list
                prev_list = next_list
                continue

            prev_list = link_Treelists(prev_list, next_list)
            prev_list = next_list
            try:
                plot_Ellipses(n_i, t, means, covs, save = True,
                        foldername = foldername, input_color = "teal", margins=margins)
            except TypeError:
                continue
            
            t_domain_no_error.append(t)

            plt.close('all')
        except ValueError:
            print(f"Failure to find GMM at t = {t}")
            break
    else:
        make_Gif(foldername, t_domain_no_error, typename = "GMM_plots")
        save_Treelist(foldername, init_list)
        print("GMM plots made for ", foldername)
    return init_list

def create_results(tdomain, foldername, init_list = None, to_plot = False, start_index = 0):
    results = {}
    params, sim_params = read_json(foldername)

    resultsfolder = foldername+"/Results"
    if not os.path.exists(resultsfolder):
        os.mkdir(resultsfolder)

    if init_list is not None:
        var_T, var_P = get_var_single(init_list, params, sim_params, 
                                      to_plot = to_plot, to_save_folder = resultsfolder)
        
        counts_all_root = get_count_single(init_list, params, sim_params, 
                                           to_plot = to_plot, to_save_folder = resultsfolder)
        
        velocity_obs = plot_velocity_single(init_list, params, sim_params, 
                                            to_plot = to_plot, to_save_folder = resultsfolder)

    ent, ent_m, ent_f, f= get_entropy_change(tdomain, foldername, to_plot=True, to_save_folder = resultsfolder)

    try:
        mean, error = from_all_root(var_T[start_index:])
        results["var_T_mean"] = mean
        results["var_T_err"] = error

        mean, error = from_all_root(var_P[start_index:])
        results["var_P_mean"] = mean
        results["var_P_err"] = error
        
        mean, error = from_all_root(counts_all_root[start_index:])
        results["count_mean"] = mean
        results["count_err"] = error

        mean, error = from_all_root(velocity_obs[start_index:])
        results["vel_mean"] = mean
        results["vel_err"] = error

        mean, error = from_all_root(ent[start_index:])
        results["Entropy_mean"] = mean
        results["Entropy_err"] = error

        mean, error = from_all_root(ent_m[start_index:])
        results["Entropy_m_mean"] = mean
        results["Entropy_m_err"] = error

        mean, error = from_all_root(ent_f[start_index:])
        results["Entropy_f_mean"] = mean
        results["Entropy_f_err"] = error

        mean, error = from_all_root(f[start_index:])
        results["fitness_mean"] = mean
        results["fitness_err"] = error

        with open(foldername + '/results.json', 'w') as fp:
            json.dump(results, fp)
    except IndexError:
        return 0
    return 1

def extract_n_total_first_line(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        match = re.search(r'Phage Population:\s*([\d.]+)', first_line)
        if match:
            return np.rint(float(match.group(1)))
    return None

def extract_t_and_N_all_but_first(file_path):
    t_vals = []
    N_vals = []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            match = re.search(r't:\s*([\d.]+)\| N:\s*([\d.]+)', line)
            if match:
                t = float(match.group(1))
                N = float(match.group(2))
                t_vals.append(t)
                N_vals.append(N)
    return t_vals, N_vals

def extract_N_from_runtime(folder_path):
    file_path = folder_path + "runtime_stats.txt"
    N_theo = extract_n_total_first_line(file_path)
    ts, Ns = extract_t_and_N_all_but_first(file_path)
    return N_theo, ts, Ns

def load_N_from_superfolder(superfolder, params_to_sweep,list_to_sweep):
    subfolders = [superfolder + f"{params_to_sweep}_{i}_seed0/" for i in list_to_sweep]

    M_data = []
    mean_data = []
    error_data = []

    for folder in subfolders:
        # params, sim_params = read_json(folder)
        N_theo, ts, Ns = extract_N_from_runtime(folder)
        M_data.append(N_theo)
        mean_data.append(np.mean(Ns))
        error_data.append(np.std(Ns))

    M_data = np.array(M_data)
    mean_data = np.array(mean_data)
    error_data = np.array(error_data)
    return M_data, mean_data, error_data