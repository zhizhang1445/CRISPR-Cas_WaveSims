import numpy as np
import numpy.ma as ma
import scipy
from scipy.signal import convolve
from scipy.sparse import issparse
from joblib import Parallel, delayed, parallel_backend
from supMethods import timeit, find_max_value_location

def immunity_gain_from_kernel(nh, n, kernel, params, sim_params, num_to_add = None):
    ndim = sim_params["ndim"]
    if ndim == 1 and isinstance(n, np.ndarray):
        return immunity_gain_from_kernel_1D(nh, n, kernel, params, sim_params, num_to_add)
    elif ndim == 2 and issparse(n):
        return immunity_gain_from_kernel_2D(nh, n, kernel, params, sim_params, num_to_add)
    else:
        raise TypeError(f"Something went wrong with Immunity Gain | n_dim: {ndim} but type is {type(nh)}")

def immunity_gain_from_kernel_1D(nh, n, kernel, params, sim_params, num_to_add = None):
    Nh = params["Nh"]
    M = params["M"]
    x_ind = np.nonzero(n)[0]
    n_nonzero = n[x_ind]

    if num_to_add is None:
        num_to_add = np.sum(n)


    def generate_prob(nh, kernel):
        input_h = nh/(Nh*M)
        probability = convolve(input_h, kernel, mode = "same")
        return probability

    if kernel is None:
        beta = params["beta"]
        if beta == 0:
            prob_acquisition = n_nonzero/np.sum(n)
        else:
            print("problem with beta type, missing Kernel")
            raise ValueError
    else:
        current_prob = generate_prob(nh, kernel)
        prob_acquisition = current_prob[x_ind]*n_nonzero
        Z_partition = np.sum(prob_acquisition)
        prob_acquisition = prob_acquisition/Z_partition
    
    to_add = np.zeros_like(n, dtype = int)

    for _ in range(num_to_add):
        sample_ind = np.random.choice(x_ind, p = prob_acquisition)
        to_add[sample_ind] += 1
        
    nh_integrated = nh+to_add
    return nh_integrated    

def immunity_loss_uniform_1D(nh, n, params, sim_params, num_to_remove = None):
    Nh = params["Nh"]
    M = params["M"]

    def get_nonzero_w_repeats_1D(n_i):
        x_ind = np.nonzero(n_i)[0]
        nonzero_values = [n_i[index] for index in x_ind]
        index_nonzero_w_repeats = []
        for value, index in zip(nonzero_values, x_ind):
            for i in range(int(value)):
                index_nonzero_w_repeats.append(index)
        return index_nonzero_w_repeats

    total_number = np.sum(nh)
    x_ind_w_repeats = get_nonzero_w_repeats_1D(nh)

    if num_to_remove is None:
        num_to_remove = int(total_number - Nh*M)

    to_remove = np.zeros_like(nh, dtype = int)

    for _ in range(num_to_remove):
        sample_ind = np.random.choice(x_ind_w_repeats, replace=False)
        to_remove[sample_ind] -= 1
        
    nh_integrated = nh+to_remove
    return nh_integrated

def immunity_loss_uniform(nh, n, params, sim_params, num_to_remove = None):
    ndim = sim_params["ndim"]
    if ndim == 1 and isinstance(n, np.ndarray):
        return immunity_loss_uniform_1D(nh, n, params, sim_params, num_to_remove)
    elif ndim == 2 and issparse(n):
        return immunity_loss_uniform_2D(nh, n, params, sim_params, num_to_remove)
    else:
        raise TypeError(f"Something went wrong with Immunity Loss | n_dim: {ndim} but type is {type(n)}")

def immunity_gain_from_kernel_2D(nh, n, kernel, params, sim_params, num_to_add = None):
    Nh = params["Nh"]
    M = params["M"]
    num_threads = sim_params["num_threads"]
    conv_size = sim_params["conv_size"]

    if num_to_add is None:
        num_to_add = np.sum(n)

    x_ind, y_ind = n.nonzero()
    support_size = len(x_ind)
    x_max, y_max  = find_max_value_location(nh)

    x_nh_sets = np.array_split(x_ind, num_threads)
    y_nh_sets = np.array_split(y_ind, num_threads)

    if params["beta"] != 0:
        def generate_prob(x_ind, y_ind):
            probability = np.zeros(len(x_ind))
            for i, (x_n, y_n) in enumerate(zip(x_ind, y_ind)):
                multiplicity = n[x_n, y_n]
                # print(multiplicity)
                x_kernel = np.abs(x_n-x_max)
                y_kernel = np.abs(y_n-y_max)

                if np.any((x_kernel >= conv_size, y_kernel >= conv_size)):
                    continue
                
                try:
                    probability[i] = multiplicity*kernel[x_kernel, y_kernel]
                    # print(multiplicity, kernel[x_kernel, y_kernel])
                except(IndexError):
                    print("wtf? Immunity Convolution out of Bounds??", x_kernel, y_kernel)
                    break
            return probability

    if kernel is None:
        beta = params["beta"]
        if beta == 0:
            prob_acquisition = n[x_ind, y_ind]/np.sum(n)
        else:
            print("problem with beta type, missing Kernel")
            raise ValueError
    else:
        results = Parallel(n_jobs=num_threads)(delayed(generate_prob)
            (x_ind, y_ind) for x_ind, y_ind in zip(x_nh_sets, y_nh_sets))
        current_prob = np.concatenate(results, axis=0)
        Z_partition = np.sum(current_prob)
        
        if Z_partition == 0:
            print("No Acquisition")
            return nh
        else:
            current_prob = current_prob/Z_partition

    def add_points(itr_list, current_prob = None):
        array = scipy.sparse.dok_matrix(nh.shape, dtype=int)
        sample_ind = np.random.choice(support_size, len(itr_list), p = current_prob)
        for i in sample_ind:
            x = x_ind[i]
            y = y_ind[i]
            array[x, y] += 1
        return array
    
    main_itr = np.arange(num_to_add)
    sub_itr_sets = np.array_split(main_itr, num_threads)
    
    results = Parallel(n_jobs=num_threads)(delayed(add_points)
        (itr, current_prob) for itr in sub_itr_sets)
    
    nh_integrated = nh+np.sum(results, axis = 0)
    return nh_integrated

def immunity_loss_uniform_2D(nh, n, params, sim_params, num_to_remove = None):
    Nh = params["Nh"]
    M = params["M"]
    num_threads = sim_params["num_threads"]

    total_number = np.sum(nh)
    
    if num_to_remove is None:
        num_to_remove = int(total_number - Nh*M)

    nonzero_indices = np.transpose(nh.nonzero())
    nonzero_indices_subset = np.array_split(nonzero_indices, num_threads, axis=0)
    nonzero_values = nh[nonzero_indices[:, 0], nonzero_indices[:, 1]].toarray().squeeze()
    nonzero_values_subset = np.array_split(nonzero_values, num_threads, axis=0)

    def process_value(values, indexes):
        index_nonzero_w_repeats = []
        for value, index in zip(values, indexes):
            index_nonzero_w_repeats.extend([index for _ in range(int(value))])
        return index_nonzero_w_repeats

    set_index_w_repeats = Parallel(n_jobs=num_threads)(delayed(process_value)(values, indexes)
                                for values, indexes in zip(nonzero_values_subset, nonzero_indices_subset))

    set_num_to_remove = [int(num_to_remove*(len(set)/total_number)) 
                         for set in set_index_w_repeats]
    set_num_to_remove_ex = int(num_to_remove - np.sum(set_num_to_remove))

    if set_num_to_remove_ex < 0:
        raise ValueError("Fuck why is set_num_to_remove_ex negative")
    else:
        thread_num = np.random.choice(num_threads, set_num_to_remove_ex)
        for i in thread_num:
            set_num_to_remove[i] += 1

    def remove_points(sub_index_w_repeats, sub_num_to_remove):
        array = scipy.sparse.dok_matrix(nh.shape, dtype=int)
        if sub_num_to_remove == 0:
            return array
        
        sampled_flat_ind = np.random.choice(len(sub_index_w_repeats), 
                                            sub_num_to_remove,replace = False)
        for i in sampled_flat_ind:
            x, y = sub_index_w_repeats[i]
            array[x, y] -= 1
        return array

    results = Parallel(n_jobs=num_threads)(
        delayed(remove_points)(sub_index_w_repeats, sub_num_to_remove) 
            for sub_index_w_repeats, sub_num_to_remove 
            in zip(set_index_w_repeats, set_num_to_remove))
    
    nh = nh + np.sum(results, axis=0)

    # if np.abs(np.sum(nh) - Nh*M) > 10 and (num_to_remove == std_remove_num):
    #     raise ValueError("bacteria died/reproduced at immunity loss, np.sum(nh) = ", np.sum(nh), "M = ", M)
    
    min_val = np.min(nh.tocoo()) if (scipy.sparse.issparse(nh)) else np.min(nh)
    
    if min_val < 0:
        raise ValueError("bacteria population is negative")

    return nh

def immunity_gain_from_probability_2D(nh, n, current_prob, params, sim_params, num_to_add = None):
    Nh = params["Nh"]
    M = params["M"]
    num_threads = sim_params["num_threads"]

    if num_to_add is None:
        num_to_add = np.sum(n)

    x_ind, y_ind = n.nonzero()
    support_size = len(x_ind)

    def add_points(itr_list, current_prob):
        array = scipy.sparse.dok_matrix(nh.shape, dtype=int)
        sample_ind = np.random.choice(support_size, len(itr_list), p = current_prob)
        for i in sample_ind:
            x = x_ind[i]
            y = y_ind[i]
            array[x, y] += 1
        return array
    
    main_itr = np.arange(num_to_add)
    sub_itr_sets = np.array_split(main_itr, num_threads)
    
    results = Parallel(n_jobs=num_threads)(delayed(add_points)
        (itr,current_prob) for itr in sub_itr_sets)
    
    nh_integrated = nh+np.sum(results, axis = 0)
    return nh_integrated