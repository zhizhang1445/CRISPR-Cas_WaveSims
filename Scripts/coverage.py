import numpy as np
import numpy.ma as ma
from pysam import coverage
import scipy
# from scipy.ndimage import convolve
from scipy.signal import convolve
from scipy import signal
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from concurrent.futures import as_completed
from supMethods import timeit
from scipy.spatial.distance import cdist

def square_split(array, split_size): #Find smallest matrix possible and make it I'm thinking 100x100
    if np.ndim(array) == 1:
        return np.split(array, len(array)//split_size)
    elif np.ndim(array) > 2:
        raise IndexError("2D or 1D plz")

    def getIndexes(subarrays, axis):
        sizes = [subarray.shape[axis] for subarray 
                        in subarrays]
        index = 0
        indexes = [0]
        for size in sizes[:-1]:
            index = index+size
            indexes.append(index)
        return indexes
    
    subarrays = []
    start_indices = []

    col_subarrays = np.array_split(array, array.shape[0]//split_size, axis=0)
    col_indexes = getIndexes(col_subarrays, 0)

    for subarray, col_index in zip(col_subarrays, col_indexes):
        row_subarrays = np.array_split(subarray, array.shape[1]//split_size, axis=1)
        subarrays.extend(row_subarrays)
        row_indexes = getIndexes(row_subarrays, 1)
        for row_index in row_indexes:
            start_indices.append((col_index, row_index))

    # Create a list of starting indices for each subarray
    return subarrays, start_indices

def split_coverage(nh, n, kernel, params, sim_params): #TODO This is already parallel, SPARSE THIS
    if scipy.sparse.issparse(nh):
        nh = nh.toarray()
        n = n.toarray()
    num_cores = sim_params["num_threads"]
    input_data = nh/params["Nh"]

    def masked_array(subarray, loc_i): 
        subarray_shape = subarray.shape
        array_masked = np.zeros_like(input_data)

        x_loc, y_loc = loc_i
        array_masked[x_loc:x_loc+subarray_shape[0], 
                     y_loc:y_loc+subarray_shape[1]] = subarray

        return array_masked
    
    def convolve_subset(input_data_subset, start_index):
        if np.sum(input_data_subset) == 0:
            return np.array([])
        
        else:
            masked_input = masked_array(input_data_subset, start_index)
            return scipy.signal.convolve2d(masked_input, kernel, mode='same')

    input_data_subsets, start_indexes = square_split(input_data, 32)

    # results = Parallel(n_jobs=num_cores, backend="multiprocessing")
    results = Parallel(n_jobs=num_cores)(delayed(convolve_subset)(subset, index) for subset, index 
                in zip(input_data_subsets, start_indexes))

    # output = np.sum(results, axis=0)
    x_ind, y_ind = np.nonzero(n)
    output = scipy.sparse.dok_matrix(input_data.shape, dtype=float)
    for result in results:
        if result.size == 0:
            continue
        else:
            output[x_ind, y_ind] += result[x_ind, y_ind]

    return output/params["M"]

def elementwise_coverage(nh, n, kernel_quarter, params, sim_params, print_progress = False):
    conv_size = sim_params["conv_size"]
    Nh = params["Nh"]
    M = params["M"]
    num_threads = sim_params["num_threads"]

    x_ind_nh, y_ind_nh = nh.nonzero()
    x_ind_n, y_ind_n = n.nonzero()

    x_nh_sets = np.array_split(x_ind_nh, num_threads)
    y_nh_sets = np.array_split(y_ind_nh, num_threads)

    input_h = np.divide(nh, Nh)
    tt_num_of_ind = len(x_ind_nh)

    def convolve_subset(x_ind_nh, y_ind_nh):
        res = scipy.sparse.dok_matrix(nh.shape, dtype=float)
        if print_progress:
            ind_nh_left = tt_num_of_ind

        for x_nh, y_nh in zip(x_ind_nh, y_ind_nh):
            value = input_h[x_nh, y_nh]

            for x_n, y_n in zip(x_ind_n, y_ind_n):

                x_kernel = np.abs(x_nh-x_n)
                y_kernel = np.abs(y_nh-y_n)

                if np.any((x_kernel >= conv_size, y_kernel >= conv_size)):
                    continue
                
                try:
                    interaction = kernel_quarter[x_kernel, y_kernel]
                except(IndexError):
                    print("wtf? Convolution out of Bounds??", x_kernel, y_kernel)
                    break

                res[x_n, y_n] += value*interaction

            if print_progress:
                print(f"Number of nh index left: {ind_nh_left}")
                ind_nh_left -= 1
        return res

    results = Parallel(n_jobs=num_threads)(delayed(convolve_subset)
        (x_ind_nh, y_ind_nh) 
            for x_ind_nh, y_ind_nh
                in zip(x_nh_sets, y_nh_sets))
    
    out = np.sum(results, axis=0)
    return out/M

def elementwise_coverage_vectorized(nh, n, kernel_dict:dict, params, sim_params, print_progress = False):
    def lookup_value(val):
        val = float(val)
        return kernel_dict.get(val, 0.)

    def convolve_subset(A, nonzero_values):
        res = np.zeros(len_ind_n)

        # dist = cdist(A, B)
        for i in range(len_ind_n): #go through indexes of n
            dist = cdist(A, B[i, :].reshape(1,2))
            res[i] = np.dot(np.vectorize(lookup_value)(dist).squeeze(), nonzero_values)
            # res[i] = np.dot(np.vectorize(lookup_value)(dist[:, 0]).squeeze(), nonzero_values)
            # res[i] = np.dot(dist[:, 0], nonzero_values)
            # dist = dist[:, 1:]
        return res

    Nh = params["Nh"]
    M = params["M"]

    x_ind_nh, y_ind_nh = nh.nonzero()
    x_ind_n, y_ind_n = n.nonzero()

    A = np.array([x_ind_nh, y_ind_nh]).transpose()
    B = np.array([x_ind_n, y_ind_n]).transpose()
    len_ind_n = len(x_ind_n)

    input_h = np.divide(nh, Nh*M)
    if scipy.sparse.issparse(input_h):
        input_h = input_h[x_ind_nh, y_ind_nh].toarray()
        nonzero_values = np.array(input_h).squeeze()
    else:
        input_h = input_h[x_ind_nh, y_ind_nh]
        nonzero_values = np.array(input_h).squeeze()

    result_values = convolve_subset(A, nonzero_values)
    res = scipy.sparse.dok_matrix(n.shape, dtype=float)
    res[x_ind_n, y_ind_n] = result_values
    return res

def double_vectorized_coverage(nh, n, coverage_kernel_dict, acquisition_kernel_dict, params, sim_params, print_progress = False):
    def lookup_value_coverage(val):
        val = float(val)
        return coverage_kernel_dict.get(val, 0.)
    def lookup_value_acquisition(val):
        val = float(val)
        return acquisition_kernel_dict.get(val, 0.)

    def double_convolve_subset(A, nonzero_values):
        coverage_res = np.zeros(len_ind_n)
        acquisition_res = np.zeros(len_ind_n)

        for i in range(len_ind_n): #go through indexes of n
            dist = cdist(A, B[i, :].reshape(1,2))
            coverage_res[i] = np.dot(np.vectorize(lookup_value_coverage)(dist).squeeze(), nonzero_values)
            acquisition_res[i] = np.dot(np.vectorize(lookup_value_acquisition)(dist).squeeze(), nonzero_values)
        return coverage_res, acquisition_res

    Nh = params["Nh"]
    M = params["M"]

    x_ind_nh, y_ind_nh = nh.nonzero()
    x_ind_n, y_ind_n = n.nonzero()

    A = np.array([x_ind_nh, y_ind_nh]).transpose()
    B = np.array([x_ind_n, y_ind_n]).transpose()
    len_ind_n = len(x_ind_n)

    input_h = np.divide(nh, Nh*M)
    if scipy.sparse.issparse(input_h):
        input_h = input_h[x_ind_nh, y_ind_nh].toarray()
        nonzero_values = np.array(input_h).squeeze()
    else:
        input_h = input_h[x_ind_nh, y_ind_nh]
        nonzero_values = np.array(input_h).squeeze()

    coverage_res, acquisition_res = double_convolve_subset(A, nonzero_values)
    out_coverage = scipy.sparse.dok_matrix(n.shape, dtype=float)
    out_coverage[x_ind_n, y_ind_n] = coverage_res

    out_aquisition = scipy.sparse.dok_matrix(n.shape, dtype=float)
    out_aquisition[x_ind_n, y_ind_n] = acquisition_res

    return out_coverage, out_aquisition

def coverage_1D(nh, kernel1D, params, sim_params):
    M = params["M"]
    Nh = params["Nh"]
    input_h = nh/(M*Nh)
    convolution = convolve(input_h, kernel1D, mode="same")
    return convolution