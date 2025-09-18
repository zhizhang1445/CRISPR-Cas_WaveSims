from operator import mul
from re import T
import numpy as np
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy.stats import multivariate_normal
from supMethods import normalize_Array
from trajectoryVisual import make_ellipse
from formulas import calc_diff_const

def get_nonzero_w_repeats(n_i):
    x_ind, y_ind = np.nonzero(n_i)
    nonzero_values = [n_i[index] for index in zip(x_ind, y_ind)]
    index_nonzero_w_repeats = []
    for value, index in zip(nonzero_values, zip(x_ind, y_ind)):
        for i in range(int(value)):
            index_nonzero_w_repeats.append(index)
    return index_nonzero_w_repeats

def get_nonzero_no_repeats(n_i):
    x_ind, y_ind = np.nonzero(n_i)
    nonzero_values = [n_i[index] for index in zip(x_ind, y_ind)]
    index_nonzero_w_no_repeats = []
    for index in zip(x_ind, y_ind):
        index_nonzero_w_no_repeats.append(index)
    return nonzero_values, index_nonzero_w_no_repeats

def checkIfInEllipse(mean1, mean2, cov1, scale = 1) -> bool:
    eigval, eigvec = np.linalg.eigh(cov1)
    # dist = np.linalg.norm(mean1-mean2)
    diff_in_eigev = np.matmul(eigvec, mean1-mean2)

    norm_dist = np.linalg.norm(diff_in_eigev/eigval)
    
    if norm_dist <= np.power(scale, 2):
        return True

def issquare(m):
    try:
        m.shape
        return m.shape[0] == m.shape[1]
    except IndexError or AttributeError or TypeError:
        return False

def get_Variances(cov, vectors = None):
    if vectors is None or np.linalg.norm(vectors) < 0.01:
        eigval, eigvec = np.linalg.eigh(cov)
        return eigval

    else:
        if np.abs(np.linalg.norm(vectors)-1) > 0.01:
            vectors = normalize_Array(vectors)

        if issquare(vectors):
            projected = np.matmul(cov, vectors)
            eigval, eigvec = np.linalg.eigh(projected)
            return eigval
        else:
            try:
                projected = np.matmul(cov, vectors)
            except IndexError:
                projected = np.matmul(cov, vectors.transpose())
            eigval_inline  = np.sum(np.abs(projected))

            vector_perpendicular = np.zeros_like(vectors)
            vector_perpendicular[1] = vectors[0]
            vector_perpendicular[0] = -1*vectors[1]

            try:
                projected = np.matmul(cov, vector_perpendicular)
            except IndexError:
                projected = np.matmul(cov, vector_perpendicular.transpose())

            eigval_transverse = np.sum(np.abs(projected))
            return np.array([eigval_transverse, eigval_inline]) #largest, smallest

def fit_unknown_GMM(index_nonzero_w_repeats,
                     n_components = 20, w = 10, reg_covar = 0):

    gaussian_estimator =  BayesianGaussianMixture(
                weight_concentration_prior_type="dirichlet_distribution",
                n_components = n_components,
                reg_covar = 0,
                init_params="kmeans",
                max_iter = 2000,
                mean_precision_prior = 0.8,
                covariance_type = "full",
                weight_concentration_prior = w,
                warm_start = True
            )
    gaussian_estimator.fit(index_nonzero_w_repeats)

    covariances = gaussian_estimator.covariances_
    means = gaussian_estimator.means_
    clusters = gaussian_estimator.predict(index_nonzero_w_repeats)

    means_red = []
    covs_red = []
    for i in range(len(np.unique(clusters))):
        means_red.append(means[i])
        covs_red.append(covariances[i])

    _ , counts = np.unique(clusters, return_counts= True)
    return means_red, covs_red, counts

def find_redudant(means, covs, counts, scale = np.sqrt(2)):
    true_count = np.copy(counts)
    to_join = [[i] for i in range(len(counts))]

    for i in range(len(means)):
        for j in range(len(means)):
            if i == j: continue
            
            avg_cov = (covs[i]*counts[i]+covs[j]*counts[j])
            avg_cov = avg_cov/(counts[i]+counts[j])

            if checkIfInEllipse(means[i], means[j], avg_cov, 
                                scale = scale):
                true_count[i] += true_count[j]
                true_count[j] = 0
                to_join[i].extend(to_join[j])
                to_join[j] = []
    return to_join, true_count

def fit_GMM_unknown_components(n, params, sim_params, index_nonzero_w_repeats = [], 
                               num_components_max = 9, scale = np.sqrt(2), plot_chi_sq = False):
    chi_sq_list = []

    if len(index_nonzero_w_repeats) == 0:
        index_nonzero_w_repeats = get_nonzero_w_repeats(n)
    if len(index_nonzero_w_repeats) == 0:
        raise ValueError("Fit GMM Failed")
    
    if num_components_max > 0:
        for n_component in range(1, num_components_max):
            means, covs, counts, chi_sq = fit_GMM(n, params, 
                                                  sim_params, index_nonzero_w_repeats=index_nonzero_w_repeats, 
                                                  n_components=n_component, return_chi_sq=True)
            chi_sq_list.append(chi_sq)

    means, covs, counts, chi_sq = fit_GMM(n, params, 
                                          sim_params, index_nonzero_w_repeats = index_nonzero_w_repeats, 
                                          cov_type = "full", n_components = num_components_max, return_chi_sq = True)
    chi_sq_list.append(chi_sq)

    index_closest_to_1 = np.abs(np.array(chi_sq_list).squeeze() - 1).argmin()    
    n_component = index_closest_to_1+1

    if plot_chi_sq:
        comp_range = np.arange(1, num_components_max+1)
        plt.figure()
        plt.plot(comp_range, chi_sq_list)
        plt.title("Chi Square vs Num of Clusters")
        plt.xlabel("Num of Components")
        plt.ylabel("Reduced Chi Squared")

    means, covs, counts= fit_GMM(n, params, sim_params, 
                                index_nonzero_w_repeats=index_nonzero_w_repeats, 
                                n_components=n_component, return_chi_sq=False)
    
    if scale > 0:
        _, true_count = find_redudant(means, covs, counts, scale)
        n_component = np.count_nonzero(true_count)

        means, covs, counts= fit_GMM(n, params, sim_params, 
                                    index_nonzero_w_repeats=index_nonzero_w_repeats, 
                                    n_components=n_component, return_chi_sq=False)
    return means, covs, counts


def fit_GMM(n, params, sim_params, index_nonzero_w_repeats = [], cov_type = "full",
                     n_components = 1, return_chi_sq = False):
    
    if len(index_nonzero_w_repeats) == 0:
        index_nonzero_w_repeats = get_nonzero_w_repeats(n)
    if len(index_nonzero_w_repeats) == 0:
        raise ValueError("Fit GMM Failed")

    gaussian_estimator =  GaussianMixture(
                n_components= n_components,
                init_params="k-means++",
                max_iter=4000,
                covariance_type = cov_type,
            )
    gaussian_estimator.fit(index_nonzero_w_repeats)
    
    covs = gaussian_estimator.covariances_
    means = gaussian_estimator.means_
    
    clusters = gaussian_estimator.predict(index_nonzero_w_repeats)
    _ , counts = np.unique(clusters, return_counts= True)

    if not return_chi_sq:
        return means, covs, counts

    nonzero_val, nonzero_ind = get_nonzero_no_repeats(n)
    indexes_cluster_assigned = gaussian_estimator.predict(nonzero_ind)
    tt_number = np.sum(n)

    chi_sq = 0
    for val, ind, index_of_cluster in zip(nonzero_val, nonzero_ind, indexes_cluster_assigned):
        mean = means[index_of_cluster]
        cov = covs[index_of_cluster]
        try:
            count_ratio = counts[index_of_cluster]/tt_number
        except IndexError:
            count_ratio = 0

        if count_ratio > 0 and np.linalg.det(cov) > 0:
            rv = multivariate_normal(mean, cov)
            pred = count_ratio*rv.pdf(ind)
            diff = np.power(val - pred, 2)/np.linalg.det(cov)
            chi_sq += np.sum(diff)

    n_deg_freedom = len(nonzero_val) - (n_components*5)
    chi_sq_red = chi_sq/n_deg_freedom

    return means, covs, counts, chi_sq_red


