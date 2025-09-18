from ast import Raise
import numpy as np
import matplotlib.pyplot as plt
import scipy
import json
import matplotlib.pyplot as plt
import sys
import os

from analysisMethods import get_tdomain, create_both_Gifs, create_Tree, create_results

def analysis(foldername, dt = 0, plot_flag = True, margin = -0.4, GMM_flag = 0):
    margins = (margin, margin)

    t_domain = get_tdomain(foldername,to_plot=False, t0 = 0, margins = margins, dt = dt)
    if len(t_domain) <= 1:
        return 0
    
    if plot_flag:
        root_list = create_both_Gifs(t_domain, foldername, margins, GMM_flag)
    else: 
        root_list = create_Tree(t_domain, foldername, GMM_flag)
    
    try:
        create_results(t_domain, foldername, root_list)
    except FileNotFoundError:
        print("No Analysis Made at ", foldername)
    return 1

if __name__ == "__main__":
    dt = 0
    GMM_flag = 0
    margin = 0.0
    input_flag = False
    subfolder_flag = False
    to_plot = False

    if len(sys.argv) > 1:
        foldername = sys.argv[1] #foldername is read out first
    
    if len(sys.argv) > 2:
        GMM_flag = int(sys.argv[2]) #analysis can be restricted to less timesteps

    if len(sys.argv) > 3:
        subfolder_flag = sys.argv[3] #analysis can be restricted to less timesteps
        if subfolder_flag in ["Superfolder", "1", "superfolder", "subfolder", "Subfolder", "true", "True"]:
            subfolder_flag = True
        else: subfolder_flag = False

    if len(sys.argv) > 4:
        to_plot = sys.argv[4]
        if to_plot in ["Plot", "to_plot", "1", "True", "plot", "true"]:
            to_plot = True

    if len(sys.argv) > 5:
        margin = float(sys.argv[5]) #analysis can be restricted to less timesteps

    if len(sys.argv) > 6:
        dt = int(sys.argv[6]) #analysis can be restricted to less timesteps

    if subfolder_flag:
        margin = float(sys.argv[4])
        subfolders = [f.path for f in os.scandir(foldername) if f.is_dir()]
        for folder in subfolders:
            analysis(folder, dt, plot_flag=to_plot, margin=margin, GMM_flag = GMM_flag)

    else:
        if to_plot:
            input_flag = True

        while input_flag:
            t_domain= get_tdomain(foldername,to_plot=True, t0 = 0, margins = (margin, margin), dt = dt)

            user_input = input("Please enter 'Yes[Y]', a float to resize margins (default -0.4) or 'No[N]' to exit \n").lower()
            if user_input in {'yes', 'y', 'Yes'}:
                input_flag = False

            elif user_input in {'no', 'n', 'No'}:
                print("Command To Exit")
                raise KeyboardInterrupt
            
            else:
                margin = float(user_input)
        analysis(foldername, dt, to_plot, margin, GMM_flag = GMM_flag)
        pass