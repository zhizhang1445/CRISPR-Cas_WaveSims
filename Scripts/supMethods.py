from matplotlib.pyplot import xlim
import numpy as np
import scipy
import json
import os
import time
from functools import wraps
import matplotlib.colors as mcolors
from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, a, x0, sigma):
    """Gaussian function."""
    normalization = 1/(sigma*np.sqrt(2*np.pi))
    return a *normalization*np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def fit_gaussian(xdomain, data):

    x = xdomain
    a_guess = np.max(data)
    x0_guess = np.sum(x * data) / np.sum(data)
    sigma_guess = np.sqrt(np.sum(data * (x - x0_guess) ** 2) / np.sum(data))
    
    popt, _ = curve_fit(gaussian, x, data, p0=[a_guess, x0_guess, sigma_guess])
    return popt

def crop_matrix_and_ranges(matrix, mean, window_size, x_range):
    # Determine the half-window size
    half_window = window_size // 2
    
    # Calculate the bounds of the cropping window
    row_start = max(mean[0] - half_window, 0)
    row_end = min(mean[0] + half_window + 1, matrix.shape[0])
    col_start = max(mean[1] - half_window, 0)
    col_end = min(mean[1] + half_window + 1, matrix.shape[1])
    
    # Crop the sparse matrix within the bounds and convert to a dense array
    cropped_matrix = matrix[row_start:row_end, col_start:col_end].toarray()
    
    # Crop x_range and y_range based on the same bounds
    x_range_cropped = x_range[row_start:row_end]
    y_range_cropped = x_range[col_start:col_end]  # y labels are the same as x labels
    
    # Pad the matrix with zeros if the cropped section is smaller than window_size
    row_pad = max(0, window_size - (row_end - row_start))
    col_pad = max(0, window_size - (col_end - col_start))
    
    if row_pad > 0 or col_pad > 0:
        padded_matrix = np.zeros((window_size, window_size))
        padded_matrix[:cropped_matrix.shape[0], :cropped_matrix.shape[1]] = cropped_matrix
        
        # Update the cropped matrix
        cropped_matrix = padded_matrix
        
        # Adjust x_range_cropped and y_range_cropped to match the padded window_size with NaNs
        x_range_cropped = np.pad(x_range_cropped, (0, row_pad), constant_values=np.nan)
        y_range_cropped = np.pad(y_range_cropped, (0, col_pad), constant_values=np.nan)
    
    return cropped_matrix, x_range_cropped, y_range_cropped


def sum_parallel(results_list, num_threads):
    def sum_pair(array1, array2):
        return array1 + array2

    if len(results_list) < num_threads:
        num_threads = 0

    if num_threads >= 32:
        #First set with 32 elements
        partial_sums = results_list[:32]

        # Second Set with all elements more than 32
        remainder_to_sum = results_list[32:]

        # Sum the first level with 16 threads
        partial_sums = Parallel(n_jobs=16)(
            delayed(sum_pair)(partial_sums[i], partial_sums[i + 1]) for i in range(0, 32, 2)
        )

        partial_sums.extend(remainder_to_sum)
        results_list = partial_sums

    if num_threads >= 16:
        partial_sums = results_list[:16]
        remainder_to_sum = results_list[16:]

        # Sum the second level with 8 threads
        partial_sums = Parallel(n_jobs=8)(
            delayed(sum_pair)(partial_sums[i], partial_sums[i + 1]) for i in range(0, 16, 2)
        )

        partial_sums.extend(remainder_to_sum)   
        results_list = partial_sums

    final_sum = np.sum(results_list, axis=0)
    return final_sum

def write2json(name, params, sim_params, results = None):
    with open(name + '/params.json', 'w') as fp:
        json.dump(params, fp)

    with open(name + '/sim_params.json', 'w') as fp:
        json.dump(sim_params, fp)

    if results is not None:
        with open(name + '/results.json', 'w') as fp:
            json.dump(results, fp)

def read_json(foldername, results_flag = False):
    params = {}
    sim_params = {}
    results = {}
    
        # Read params.json
    try:
        with open(foldername+'params.json', 'r') as params_file:
            params = json.load(params_file)
        
            # Read sim_params.json
        with open(foldername+'sim_params.json', 'r') as sim_params_file:
            sim_params = json.load(sim_params_file)
    except FileNotFoundError:
        with open(foldername+'/params.json', 'r') as params_file:
            params = json.load(params_file)
        
            # Read sim_params.json
        with open(foldername+'/sim_params.json', 'r') as sim_params_file:
            sim_params = json.load(sim_params_file)
    
    if results_flag:
        try:
            with open(foldername+'results.json', 'r') as results_file:
                    results = json.load(results_file)
        except FileNotFoundError:
            raise FileNotFoundError("results.json not found")
        return params, sim_params, results

    return params, sim_params

def load_last_output(foldername, dim = 2):
    if dim == 2:
        prefix = "sp_frame_nh"
        sufix = ".npz"
    else:
        prefix = "frame_nh"
        sufix = ".npy"

    files_in_directory = [f for f in os.listdir(foldername) if f.startswith(prefix) and f.endswith(sufix)]

    if not files_in_directory:
        raise FileNotFoundError("No " + prefix + "*" + sufix + " files found in the current directory.")

    highest_numeric_value = float('-inf')

    for filename in files_in_directory:
        numeric_value = int(filename.split(prefix)[1].split(sufix)[0]) #USE Nh instead of N cuz N will give you h0
        if numeric_value > highest_numeric_value:
            highest_numeric_value = numeric_value
    
    if dim == 2:
        n = scipy.sparse.load_npz(foldername + f"/sp_frame_n{highest_numeric_value}.npz").todok()
        nh = scipy.sparse.load_npz(foldername + f"/sp_frame_nh{highest_numeric_value}.npz").todok()
    else:
        n = np.load(foldername + f"/frame_n{highest_numeric_value}.npy")
        nh = np.load(foldername + f"/frame_nh{highest_numeric_value}.npy")
    return highest_numeric_value, n, nh

def load_outputs(foldername, t, add_fitness = False, dim = 2):
    if dim == 2:
        try:
            n = scipy.sparse.load_npz(foldername + f"/sp_frame_n{t}.npz").todok()
            nh = scipy.sparse.load_npz(foldername + f"/sp_frame_nh{t}.npz").todok()
            if add_fitness:
                f = scipy.sparse.load_npz(foldername + f"/sp_frame_f{t}.npz").todok()
        except FileNotFoundError:
            raise FileNotFoundError(f"The output 2D was not found at {t}")
    
    else:
        try:
            n = np.load(foldername + f"/frame_n{t}.npy")
            nh = np.load(foldername + f"/frame_nh{t}.npy")

            if add_fitness:
                f = np.load(foldername + f"/frame_f{t}.npy")
        except FileNotFoundError:
            raise FileNotFoundError(f"The output 1D was not found at {t}")
        
    if add_fitness:
        return n, nh, f

    return n, nh

def time_conv(st):
    return time.strftime("%H:%M:%S", time.gmtime(st))

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{func.__name__} took {time_conv(total_time)}')
        return result
    return timeit_wrapper

def minmax_norm(array):
    max_val = array.max()
    min_val = array.min()
    res = (array - min_val)/np.ptp(array)
    return res

def extract_xy(list) : #if you have a list [[x0,y0], [x1, y1], ...] and you want [x0, x1, ...] and [y0, y1, ...]
    try:
        if len(list) == 0:
            x_val = y_val = []

        elif len(list) == 1:
            A = np.array(list).squeeze()
            x_val = A[0]
            y_val = A[1]
            
        else:
            A = np.array(list).squeeze()
            x_val = A[:,0]
            y_val = A[:,1]

    except TypeError:
        x_val = np.zeros(len(list))
        y_val = np.zeros(len(list))
        for x, y, item in zip(x_val, y_val, list):
            x = item[0]
            y = item[1]

    return x_val, y_val

def normalize_Array(array_input, norm = 1):
    try:
        output = array_input/np.linalg.norm(array_input)
        output = output*norm
    except TypeError:
        output = []
        for a in array_input:
            output_single = a/np.linalg.norm(a)
            output.append(output_single*norm)
    
    return output

def average_of_pairs(arr):
    averages = []
    for i in range(0, len(arr)-1, 1):
        average = (arr[i] + arr[i+1]) / 2.0
        averages.append(average)
    return np.array(averages)

def find_max_value_location(matrix):
    max_value = float('-inf')
    max_row, max_col = None, None

    for (row, col), value in matrix.items():
        if value > max_value:
            max_value = value
            max_row = row
            max_col = col

    return max_row, max_col

def find_mean_location(matrix):
    mean_row = 0
    mean_col = 0
    tt_value = 0

    for (row, col), value in matrix.items():
        mean_row += row*value
        mean_col += col*value
        tt_value += value
    
    mean_row = mean_row/tt_value
    mean_col = mean_col/tt_value

    return mean_row, mean_col

def calc_diff_const(params, sim_params):
    dx = sim_params["dx"]
    shape = params["gamma_shape"]
    mu = params["mu"]
    ndim = sim_params["ndim"]

    mean = 2*dx
    scale = mean/shape
    gamma_var = shape*(scale**2)
    
    if ndim == 2:
        cos_uni_var = 1/2
    elif ndim == 1:
        cos_uni_var = 1

    prod_var = (mean**2 + gamma_var)*(cos_uni_var)

    diff_const = mu*prod_var/2
    return diff_const

def get_xdomain(params, sim_params):
    x_lim = sim_params["xdomain"]
    dx = sim_params["dx"]
    x_range = np.arange(-x_lim, x_lim, dx)
    return x_range

def compute_shift(nh, nh_prev, type = "max"):
    if type == "max":
        x_old, y_old= find_max_value_location(nh_prev)
        x_new, y_new= find_max_value_location(nh)
    elif type == "mean":
        x_old, y_old = find_mean_location(nh_prev)
        x_new, y_new = find_mean_location(nh)

    shift_x = np.rint(x_new - x_old).astype(int)
    shift_y = np.rint(y_new - y_old).astype(int)
    return shift_x, shift_y

def calculate_velocity(N, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    r = params["r"]

    D = calc_diff_const(params, sim_params)
    inv_v_tau = (np.power(R0, 1/M)-1)/r
    s = M*inv_v_tau

    common_log = 24*np.log(N*np.power(D*np.power(s, 2), 1/3))
    v = np.power(s, 1/3)*np.power(D, 2/3)*np.power(common_log, 1/3)
    return v

def calculate_DlnND(N, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    r = params["r"]

    D = calc_diff_const(params, sim_params)
    inv_v_tau = (np.power(R0, 1/M)-1)/r
    s = M*inv_v_tau

    common_log = np.log(N*np.power(D*np.power(s, 2), 1/3))
    v = np.power(D, 2/3)*np.power(common_log, 1/3)
    return v

def calculate_var(N, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    r = params["r"]

    D = calc_diff_const(params, sim_params)
    inv_v_tau = (np.power(R0, 1/M)-1)/r
    s = M*inv_v_tau

    common_log = np.log(N*np.power(D*np.power(s, 2), 1/3))
    var = np.power(D/s, 1/3)*np.power(common_log, 1/6)
    return var

def calculate_FisherVelocity(N, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    r = params["r"]

    D = calc_diff_const(params, sim_params)
    inv_v_tau = (np.power(R0, 1/M)-1)/r
    s = M*inv_v_tau

    common_log = np.log(N*np.power(D*np.power(s, 2), 1/3))
    uc = (1/4)*np.power(D/s, 1/3)*(np.power(common_log, 2/3))
    v = 2*np.sqrt(s*uc*D)
    return v

def running_median_filter(signal, window_size, padding = 'symmetric'):
    pad_width = window_size // 2
    padded_signal = np.pad(signal, pad_width, mode=padding)
    filtered_signal = scipy.signal.medfilt(padded_signal, kernel_size=window_size)
    trimmed_signal = filtered_signal[pad_width:-pad_width]
    return trimmed_signal