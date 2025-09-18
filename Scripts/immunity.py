import numpy as np
import numpy.ma as ma
import scipy
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from concurrent.futures import as_completed
from supMethods import timeit, find_max_value_location
from scipy.sparse import issparse

def immunity_mean_field_add(nh, n, params, sim_params, num_to_add = None, int_prob = None):
    Nh = params["Nh"]
    M = params["M"]
    A = params["A"]
    ndim = sim_params["ndim"]

    if int_prob is None:
        prob = n
    else:
        prob = np.abs(int_prob*n)

    if num_to_add is None:
        num_to_add = np.rint(A*np.sum(n)).astype(int)

    if num_to_add > 1000:
        nh_new = nh + num_to_add*(prob/np.sum(prob))
    else:
        if ndim == 1:
            n_to_add = np.zeros_like(n)
            x_inds = np.nonzero(n)[0]

            nonzero_prob = prob[x_inds]/np.sum(prob)
            num_choice = len(x_inds)

            for i in range(num_to_add):
                choice = np.random.choice(num_choice, p=nonzero_prob)
                n_to_add[x_inds[choice]] += 1
            
            nh_new = nh + n_to_add

        else:
            n_to_add = scipy.sparse.dok_matrix(n.shape, dtype=np.int64)
            x_ind, y_ind = n.nonzero()
            nonzero_prob = np.array(prob[x_ind, y_ind].toarray()/np.sum(prob)).squeeze()
            # nonzero_prob = np.array(n[x_ind, y_ind].toarray()/np.sum(n)).squeeze()
            num_choice = len(x_ind)

            for i in range(num_to_add):
                choice = np.random.choice(num_choice, p=nonzero_prob)
                n_to_add[x_ind[choice], y_ind[choice]] +=1

            nh_new = nh + n_to_add

    if ndim == 1:
        nh_new = np.rint(nh_new).astype(int)
    else:
        x_ind, y_ind = nh_new.nonzero()

        nonzero_items = nh_new[x_ind, y_ind]

        if issparse(nonzero_items):
            nonzero_items = nonzero_items.toarray().squeeze()
        else:
            nonzero_items = np.array(nonzero_items).squeeze()
        nh_new[x_ind, y_ind] = np.rint(nonzero_items).astype(int)

    new_tt_number = np.sum(nh_new)
    error = int(int(np.sum(nh)) + num_to_add - int(new_tt_number))

    if ndim == 1:
        x_inds = np.nonzero(nh_new)[0]

        if error > 0:
            for _ in range(error):
                x_ind = np.random.choice(x_inds)
                nh_new[x_ind] += 1

        if error < 0:
            while(error < 0):
                # print(x_inds)
                x_ind = np.random.choice(x_inds)
                if nh_new[x_ind] > 0:
                    nh_new[x_ind] -= 1
                    error += 1
        min_val = np.min(nh_new)

    else:
        nh_new = scipy.sparse.dok_matrix(nh_new)
        # print(error)
        x_ind, y_ind = nh_new.nonzero()
        support_size = len(x_ind)

        if error > 0:
            for _ in range(error):
                choice = np.random.choice(support_size)
                nh_new[x_ind[choice], y_ind[choice]] += 1

        if error < 0:
            while(error < 0):
                # print(x_inds)
                choice = np.random.choice(support_size)

                if nh_new[x_ind[choice], y_ind[choice]] > 0:
                    nh_new[x_ind[choice], y_ind[choice]] -= 1
                    error += 1

        min_val = np.min(nh_new.tocoo()) if (scipy.sparse.issparse(nh_new)) else np.min(nh_new)

    if np.sum(nh_new) != np.ceil(np.sum(nh) + num_to_add):
        raise ValueError("bacteria died/reproduced at immunity gain, Nh = ", np.sum(nh_new))
    
    if min_val < 0:
        print(params)
        raise ValueError("bacteria population is negative")

    return nh_new

def immunity_mean_field_remove(nh_integrated, n, params, sim_params, num_to_remove = None):
    Nh = params["Nh"]
    M = params["M"]
    A = params["A"]
    ndim = sim_params["ndim"]

    prob = (nh_integrated/np.sum(nh_integrated))

    if num_to_remove is None:
        num_to_remove = int(np.sum(nh_integrated)) - int(Nh*M)

    nh_new = nh_integrated - num_to_remove*prob
    # nh_new = np.rint(nh_new).astype(int)

    if ndim == 1:
        # print(ndim)
        nh_new = np.rint(nh_new).astype(int)
    else:
        x_ind, y_ind = nh_new.nonzero()
        nonzero_items = nh_new[x_ind, y_ind]
        if issparse(nonzero_items):
            nonzero_items = nonzero_items.toarray().squeeze()
        else:
            nonzero_items = np.array(nonzero_items).squeeze()
        nh_new[x_ind, y_ind] = np.rint(nonzero_items).astype(int)

    new_tt_number = np.sum(nh_new)
    error = int(int(np.sum(nh_integrated)) - num_to_remove - int(new_tt_number))
    # print(error)
    
    if ndim == 1:
        x_inds = np.nonzero(nh_new)[0]

        if error > 0:
            for _ in range(error):
                x_ind = np.random.choice(x_inds)
                nh_new[x_ind] += 1

        if error < 0:
            while(error < 0):
                # print(x_inds)
                x_ind = np.random.choice(x_inds)
                if nh_new[x_ind] > 0:
                    nh_new[x_ind] -= 1
                    error += 1
        min_val = np.min(nh_new)

    else:
        nh_new = scipy.sparse.dok_matrix(nh_new)
        # print(error)
        x_ind, y_ind = nh_new.nonzero()
        support_size = len(x_ind)

        if error > 0:
            for _ in range(error):
                choice = np.random.choice(support_size)
                nh_new[x_ind[choice], y_ind[choice]] += 1

        if error < 0:
            while(error < 0):
                # print(x_inds)
                choice = np.random.choice(support_size)

                if nh_new[x_ind[choice], y_ind[choice]] > 0:
                    nh_new[x_ind[choice], y_ind[choice]] -= 1
                    error += 1

        min_val = np.min(nh_new.tocoo()) if (scipy.sparse.issparse(nh_new)) else np.min(nh_new)

    if np.sum(nh_new) != np.ceil(np.sum(nh_integrated) - num_to_remove):
        ceiling = np.ceil(np.sum(nh_integrated) - num_to_remove) - np.sum(nh_new)
        int_error = int(int(np.sum(nh_integrated)) - num_to_remove - int(nh_new))
        raise ValueError("bacteria died/reproduced at immunity gain, Nh = ", np.sum(nh_new), "ceiling error:    ", ceiling, "int error:  ", int_error)
    
    if min_val < 0:
        raise ValueError("bacteria population is negative")
    return nh_new

# @timeit
def immunity_update_SerialChoice(nh, n, params, sim_params):
    Nh = params["Nh"]
    M = params["M"]
    num_threads = sim_params["num_threads"]
    nh = nh + n
    num_to_remove = int(np.sum(nh) - Nh*M)

    nonzero_indices = np.transpose(nh.nonzero())
    nonzero_indices_subset = np.array_split(nonzero_indices, num_threads, axis=0)
    nonzero_values = nh[nonzero_indices[:, 0], nonzero_indices[:, 1]].toarray().squeeze()
    nonzero_values_subset = np.array_split(nonzero_values, num_threads, axis=0)

    def process_value(values, indexes):
        index_nonzero_w_repeats = []
        for value, index in zip(values, indexes):
            index_nonzero_w_repeats.extend([index for _ in range(int(value))])
        return index_nonzero_w_repeats

    results = Parallel(n_jobs=num_threads)(delayed(process_value)(values, indexes)
                                for values, indexes in zip(nonzero_values_subset, nonzero_indices_subset))

    index_nonzero_w_repeats = []
    for sublist in results:
        index_nonzero_w_repeats.extend(sublist)

    sample_flat_ind = np.random.choice(len(index_nonzero_w_repeats), num_to_remove,replace = False)
    ind_per_thread_list = np.array_split(sample_flat_ind, num_threads)

    def remove_points(flat_index):
        array = scipy.sparse.dok_matrix(nh.shape, dtype=int)
        sample_ind = [index_nonzero_w_repeats[i] for i in flat_index]
        for x,y in sample_ind:
            array[x, y] -= 1

        return array

    results = Parallel(n_jobs=num_threads)(
        delayed(remove_points)(flat_index) for flat_index in ind_per_thread_list)
    nh = nh + np.sum(results, axis=0)

    if np.abs(np.sum(nh) - Nh*M) >= 1:
        raise ValueError("bacteria died/reproduced at immunity gain, Nh = ", np.sum(nh))
    
    min_val = np.min(nh.tocoo()) if (scipy.sparse.issparse(nh)) else np.min(nh)

    if min_val < 0:
        raise ValueError("bacteria population is negative")

    return nh