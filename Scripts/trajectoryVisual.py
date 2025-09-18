import numpy as np
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio
import os


def plot_sparse(n, title = None, margins = (-0.45, -0.45)):
    plt.figure()
    plt.margins(margins[0], margins[1])
    plt.contourf(n.toarray().transpose(), cmap = "Reds")
    if title is not None:
        plt.title(title)

def make_frame(foldername, i, save = True, margins = (-0.45, -0.45)):
    n_i = scipy.sparse.load_npz(foldername+f"/sp_frame_n{i}.npz")
    nh_i = scipy.sparse.load_npz(foldername+f"/sp_frame_nh{i}.npz")

    fig = plt.figure()
    plt.contour(n_i.toarray().transpose(), cmap = "Reds")
    plt.contour(nh_i.toarray().transpose(), cmap = "Blues")
    plt.margins(margins[0], margins[1])
    
    plt.title(f"N and Nh distribution at timestep {i}")
    if save:
        try:
            plt.savefig(f'{foldername}/time_plots/img_{i}.png', 
                        transparent = False, facecolor = 'white')
        except FileNotFoundError:
            os.mkdir(foldername+"/time_plots")
            plt.savefig(f'{foldername}/time_plots/img_{i}.png', 
                        transparent = False, facecolor = 'white')
        plt.close("all")

def generate_colors(num_colors):
    all_colors = list(mcolors.CSS4_COLORS.keys())
    return all_colors[:num_colors]

def make_ellipse(means, covariances, color = "teal"):
    if isinstance(covariances,float):
        covariances = np.eye(means.size)*covariances

    v, w = np.linalg.eigh(covariances)
    v = v.squeeze(); w = w.squeeze()
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(
        means, v[0], v[1], angle = 180 + angle, color = color
    )
    return ell

def plot_Ellipses(n, i, means, covs, margins = (-0.45, -0.45), 
            save = False, foldername = None, input_color = "random"):
    if scipy.sparse.issparse(n):
        n = n.toarray()

    num_clusters = len(means)

    fig, ax = plt.subplots()
    plt.title(f"{num_clusters} GMMs at time {i}")
    ax.contour(n.transpose(), cmap = "Reds") #Note that contour is trasnposed
    ax.margins(margins[0], margins[1])
    
    if input_color == "random":
        colors = generate_colors(num_clusters)
    elif mcolors.is_color_like(input_color):
        colors = [input_color for i in range(num_clusters)]
    else:
        colors = ["teal" for i in range(num_clusters)]

    for mean , cov, color in zip(means, covs, colors):
        ell1 = make_ellipse(mean, cov, color)
        ax.add_patch(ell1)

    if save:
        try:
            plt.savefig(f'./{foldername}/GMM_plots/img_{i}.png', 
                        transparent = False, facecolor = 'white')
        except FileNotFoundError:
            os.mkdir(foldername+"/GMM_plots")
            plt.savefig(f'./{foldername}/GMM_plots/img_{i}.png', 
                        transparent = False, facecolor = 'white')
        plt.close()

def make_Gif(foldername, tdomain, typename = "time_plots"):
    frames = []
    if not (typename == "GMM_plots" or typename == "time_plots"):
        raise ValueError("Input should be Either GMM_plots or time_plots")
    for i in tdomain:
        image = imageio.v2.imread(f'./{foldername}/{typename}/img_{i}.png')
        frames.append(image)

    imageio.mimsave(f'./{foldername}/{typename}/Animation.gif', 
                    frames)

# def make_timeGif(foldername, t_domain):
#     for t in t_domain:
#         make_frame(foldername, t, save = True)

#     make_Gif(foldername, t_domain, typename = "time_plots")

# def make_GMMGif(foldername, t_domain):
#     for t in t_domain:
#         n_i = scipy.sparse.load_npz(foldername+f"/sp_frame_n{t}.npz").todok()
#         indexes = get_nonzero_w_repeats(n_i)
#         means_gmm, covs_gmm, counts_gmm = fit_unknown_GMM(indexes,n_components=1, w = 1000, reg_covar=1e4)
#         means, covs, counts = reduce_GMM(means_gmm, covs_gmm, counts_gmm)
#         plot_Ellipses(n_i, t, means, covs, save = True,
#                     foldername = foldername, input_color = "teal")
        
#     make_Gif(foldername, t_domain, typename = "GMM_plots")
