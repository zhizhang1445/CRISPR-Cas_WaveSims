import numpy as np
import numpy.ma as ma
import scipy

def get_time_next_HGT(current_t, params, sim_params, sampling = "random"):
    rate_event = params["rate_HGT"]
    if rate_event == 0:
        return sim_params["tf"]
    
    scale_param = 1/rate_event
    if sampling == "random":
        next_time_reaction = current_t + np.random.exponential(scale_param, 1)[0]
    elif sampling == "constant":
        next_time_reaction = current_t + scale_param
    else:
        next_time_reaction = current_t + np.inf
    return next_time_reaction

def HGT_logistic_event(t, n, params, sim_params, toprint = False, sampling = "random", no_punishment = False):
    rate_recovery = params["rate_recovery"]
    time2nextevent = sim_params["time_next_event"]
    flag_acquisition = params["HGT_type"]
    A = params["A"]

    if t > time2nextevent:
        if toprint: print("HGT Occurs at time:", t)
        num_to_add = np.sum(n)*(A + params["HGT_bonus_acq_ratio"])
        num_to_remove = A*np.sum(n)
        params["M"] = params["M"] + (params["HGT_bonus_acq_ratio"]*np.sum(n))/params["Nh"]

        time2nextevent = get_time_next_HGT(t, params, sim_params, sampling)
        sim_params["time_next_event"] = time2nextevent

    #Calculate return to normal amount (num to remove)
    elif (params["M"] > params["M0"]) and (not no_punishment): #Punishment

        if toprint: print("Lower Acquisition:", t)
        tt_spacers_current = int(params["M"]*params["Nh"])
        tt_spacers_lost = np.rint(params["Nh"]*(1-(params["M"]/params["M0"]))*params["M"]*rate_recovery)
        params["M"] = params["M"] + tt_spacers_lost/int(params["Nh"])

        if flag_acquisition == 0: #Acquisition surpressed
            num_to_add = A*np.sum(n) + tt_spacers_lost
            if num_to_add < 0:
                num_to_add = 0

            num_to_remove = A*np.sum(n)
        
        if flag_acquisition == 1:  #Loss Enhanced
            num_to_remove = A*np.sum(n) - tt_spacers_lost
            if num_to_remove < 0:
                num_to_remove = 0
            
            num_to_add = A*np.sum(n)

    else:
        if toprint: print("Nothing happens at time:", t)

        num_to_add = A*np.sum(n)
        num_to_remove = A*np.sum(n)
        return params, sim_params, np.rint(num_to_add).astype(int), np.rint(num_to_remove).astype(int)

    
    return params, sim_params, np.rint(num_to_add).astype(int), np.rint(num_to_remove).astype(int)

def HGT_discrete_event(t, n, params, sim_params, toprint = False):
    timesteps_to_recovery = np.ceil(1/(params["rate_recovery"]*sim_params["dt"]))
    counter_to_recovery = sim_params["counter_to_recovery"]
    time2nextevent = sim_params["time_next_event"]

    if t > time2nextevent:
        if toprint: print("HGT Occurs at time:", t)
        num_to_add = np.sum(n)*(1 + params["HGT_bonus_acq_ratio"])
        num_to_remove = np.sum(n)
        params["M"] = params["M"] + (params["HGT_bonus_acq_ratio"]*np.sum(n))/params["Nh"]

        counter_to_recovery = counter_to_recovery + timesteps_to_recovery
        time2nextevent = get_time_next_HGT(t, params, sim_params)

    elif counter_to_recovery > 0:
        if toprint: print("Lower Acquisition:", t)
        num_to_add = np.sum(n) - (params["HGT_bonus_acq_ratio"]*np.sum(n))*(1/timesteps_to_recovery)
        num_to_remove = np.sum(n)
        params["M"] = params["M"] - (params["HGT_bonus_acq_ratio"]*np.sum(n)/params["Nh"])*(1/timesteps_to_recovery)
        counter_to_recovery -= 1

    else:
        if toprint: print("Nothing happens at time:", t)

        num_to_add = np.sum(n)
        num_to_remove = np.sum(n)

    sim_params["counter_to_recovery"] = counter_to_recovery
    sim_params["time_next_event"] = time2nextevent
    return params, sim_params, num_to_add, num_to_remove