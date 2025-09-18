import numpy as np
import matplotlib.pyplot as plt
import scipy
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from trajsTree import TreeNode
from supMethods import extract_xy, average_of_pairs, calculate_velocity, calculate_var, calculate_FisherVelocity
# from formulas import calculate_FisherVelocity, calculate_velocity, calculate_var

def from_all_root(data_all_root: np.ndarray):
    mean = np.mean(data_all_root)
    error = np.sqrt(np.var(data_all_root)/np.size(data_all_root))
    return mean, error

def get_count_single(init_list, params, sim_params, to_plot = False, to_save_folder = None):
    count_all_root = []
    if to_plot:
        plt.figure()

    for root_node in init_list:
        for trajs in root_node.get_all_traversals():
            time = [x.frame for x in trajs]
            counts = np.array([x.count for x in trajs])

            if len(counts)>0:
                count_all_root.extend(counts)
                if to_plot:
                    plt.plot(time, counts)

    if to_plot:
        plt.title("Counts vs Simulation time")
        plt.xlabel("Time [1]")
        plt.ylabel("Phage Count [1]")

    if to_save_folder is not None:
        if to_plot:
            plt.savefig(to_save_folder + "/counts.png")
            plt.close("all")
        np.save(to_save_folder + "/counts.npy", np.array(counts).squeeze())

    return  np.mean(count_all_root).squeeze(), np.var(count_all_root).squeeze()

def get_var_single(init_list, params, sim_params, to_plot = False, to_save_folder = None):
    var_T_all_root = []
    var_P_all_root = []
    if to_plot:
        fig, ax = plt.subplots(2,1)

    for root_node in init_list:
        for trajs in root_node.get_all_traversals():
            time = [x.frame for x in trajs]
            var_transverse = [x.vel_var[0] for x in trajs]
            var_parallel = [x.vel_var[1] for x in trajs]


            count = np.array([x.count for x in trajs])

            if len(count)>0:
                var_T_all_root.extend(var_transverse)
                var_P_all_root.extend(var_parallel)
                if to_plot:
                    ax[0].plot(time, var_parallel)
                    ax[1].plot(time, var_transverse)

    if to_plot:
        ax[0].set_title("Parallel Variance")
        ax[1].set_title("Transverse Variance")

    if to_save_folder is not None:
        if to_plot:
            fig.savefig(to_save_folder + "/variances.png")
            plt.close("all")
        
        np.save(to_save_folder + "/variances_T.npy", np.array(var_T_all_root).squeeze())
        np.save(to_save_folder + "/variances_P.npy", np.array(var_P_all_root).squeeze())
    
    return np.array(var_T_all_root).squeeze(), np.array(var_P_all_root).squeeze()

def calc_var_single(init_list, params, sim_params):
    var_calc_all_root = []

    for root_node in init_list:
        for trajs in root_node.get_all_traversals():
            count = np.array([x.count for x in trajs])
            if len(count)>0:
                var_calc = calculate_var(count, params, sim_params)
                var_calc_all_root.extend(var_calc)

    mean = np.mean(var_calc_all_root)
    variance = np.var(var_calc_all_root)

    return  [mean, np.sqrt(1.66)*mean], [variance, 1.66*variance]


def plot_velocity_single(init_list, params, sim_params, to_plot = False, to_save_folder = None):
    velocity_obs = np.array([])
    if to_plot:
        plt.figure()

    for root_node in init_list:
        for trajs in root_node.get_all_traversals():
            positions = [x.mean for x in trajs]
            time = [x.frame for x in trajs]
            time_avg = average_of_pairs(time)

            x_val, y_val = extract_xy(positions)
            if isinstance(x_val, np.ndarray):
                dt = np.diff(time)
                v_obs = np.sqrt(np.diff(x_val)**2 + np.diff(y_val)**2)/dt
                velocity_obs = np.concatenate([velocity_obs, v_obs], axis = 0)

                if to_plot:
                    plt.plot(time_avg, v_obs, color = 'teal')
                    plt.xlabel("Time")
                    plt.ylabel("Velocity")

    if to_plot:
        plt.title("Observed Velocity")

    if to_save_folder is not None:
        if to_plot:
            plt.savefig(to_save_folder + "/velocities.png")
            plt.close("all")
        
        np.save(to_save_folder + "/velocities.npy", velocity_obs)
    return np.array(velocity_obs).squeeze()


def calc_velocity_single(init_list, params, sim_params):
    velocity_calc = np.array([])
    velocity_Fisher = np.array([])

    for root_node in init_list:
        for trajs in root_node.get_all_traversals():
            positions = [x.mean for x in trajs]
            time = [x.frame for x in trajs]
            time_avg = average_of_pairs(time)
            
            counts = np.array([x.count for x in trajs])
            counts_avg = average_of_pairs(counts)

            x_val, _ = extract_xy(positions)
            if isinstance(x_val, np.ndarray):

                v_calc = calculate_velocity(counts_avg, params, sim_params)
                velocity_calc = np.concatenate([velocity_calc, v_calc], axis = 0)
                
                v_Fisher = calculate_FisherVelocity(counts_avg, params, sim_params)
                velocity_Fisher = np.concatenate([velocity_Fisher, v_Fisher], axis = 0)

    return velocity_calc, velocity_Fisher

def plot_velocity(foldername_itr, limits = [0.07, 0.07]):
    v_obs_mean = []
    v_obs_var = []
    v_calc_mean = []
    v_calc_var = []
    v_Fisher_mean = []
    v_Fisher_var = []

    for foldername in foldername_itr:
        with open(foldername + "/params.json") as json_file:
            params = json.load(json_file)
        with open(foldername + "/sim_params.json") as json_file:
            sim_params = json.load(json_file)

        init_list = []
        tree_index = 0
        tree_path = foldername + f"/trajs_trees/tree{tree_index}.json"
        while(os.path.isfile(tree_path)):
            # print("tree loaded:", tree_index)
            init_list.append(TreeNode.load_tree(tree_path))
            tree_index += 1
            tree_path = foldername + f"/trajs_trees/tree{tree_index}.json"
    
        velocity_obs= plot_velocity_single(init_list, params, sim_params)
        velocity_calc, velocity_Fisher = calc_velocity_single(init_list, params, sim_params)

        v_obs_mean.append(np.mean(velocity_obs))
        v_obs_var.append(np.sqrt(np.var(velocity_obs)))
        v_calc_mean.append(np.mean(velocity_calc))
        v_calc_var.append(np.sqrt(np.var(velocity_calc)))
        v_Fisher_mean.append(np.mean(velocity_Fisher))
        v_Fisher_var.append(np.sqrt(np.var(velocity_Fisher)))

    plt.figure(figsize = [3,3])
    plt.errorbar(v_calc_mean, v_obs_mean, xerr= v_calc_var,
                yerr = v_obs_var, linestyle = "None", capsize = 1)
    
    line_x = np.linspace(0, limits[0], 100)
    line_y = np.linspace(0, limits[1], 100)

    # Plot the diagonal line
    plt.plot(line_x, line_y, color='red', linestyle='--', label = "Linear Fitness Velocity")
    plt.xlim(0, limits[0])
    plt.ylim(0, limits[1])
    plt.xlabel("Theoretical Velocity")
    plt.ylabel("Numerical Velocity")
    plt.legend()

def get_mean_count_single(init_list, params, sim_params):
    count_all_root = np.array([])

    for root_node in init_list:
        for trajs in root_node.get_all_traversals():
            positions = [x.mean for x in trajs]
            time = [x.frame for x in trajs]
            time_avg = average_of_pairs(time)
            
            counts = np.array([x.count for x in trajs])
            count_all_root = np.concatenate([count_all_root, np.mean(counts)], axis=0)

    return count_all_root 