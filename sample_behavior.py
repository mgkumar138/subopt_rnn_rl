'''
Sample RNN and behavior data using trained model params.
save RNN behavior data to separate folders.

'''

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import glob
import os
import pickle

import sys; sys.path.append('src/')
import utils_funcs, utils_fp, model_rnn, tasks


analysis = 'all'
epochs = 1 #must adjust format of saved variables if you increase from 1
seeds = 50 # used for confidence intervals 
contexts = ['CP', 'OB'] # contexts to analyze

data_dir = "./model_params_101000/"
save_dir_behav = "data/rnn_behav/model_params_101000/"
save_dir_rnn = "data/rnn_activity/model_params_101000/"
save_dir_fp = "data/fixed_points/model_params_101000/"
os.makedirs(save_dir_behav, exist_ok=True)
os.makedirs(save_dir_rnn, exist_ok=True)
os.makedirs(save_dir_fp, exist_ok=True)


#rnn model inputs
input_dim = 6 + 3  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.
hidden_dim = 64  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.
#setup the rnn model 
rnn_model = model_rnn.ActorCritic(input_dim, hidden_dim, action_dim)

if analysis == 'gamma' or analysis == "all":
    # influence of gamma

    gammas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'gammas':gammas,'states':[]}
    fps = np.zeros([len(gammas),seeds, len(contexts), 3])


    gamma_dict = {}
    gamma_cp_list = [] 
    gamma_ob_list = []
    gamma_models = []
    gamma_rnn_activity = []
    gamma_fp = []
    
    for g, gamma in enumerate(gammas):
        
        file_names= data_dir+f"*_V3_{gamma}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        models = [f for f in models if float(f.split("\\")[-1].split("_")[0]) > 5] # remove items in file_names that begin with "-"

        print(gamma, len(models))

        for m, model in enumerate(models):
            all_states, rnn_activity, loaded_model = model_rnn.rnn_predict(
                rnn_model = rnn_model,
                model_path = model,
                epochs=epochs)
            fps[g, m] = utils_fp.find_fixed_points(loaded_model, rnn_activity)

            all_param_states['states'].append(all_states)

            gamma_dict[m, g] = {"gamma", gamma}
            gamma_cp_list.append(all_states[0,0])
            gamma_ob_list.append(all_states[0,1])
            gamma_rnn_activity.append(rnn_activity)
            gamma_fp.append(fps[g, m])
            gamma_models.append(model)

        with open(os.path.join(save_dir_behav, "gamma_all_param_states.pkl"), "wb") as f:
            pickle.dump(all_param_states, f)
        with open(os.path.join(save_dir_behav, "gamma_dict.pkl"), "wb") as f:
            pickle.dump(gamma_dict, f)
        with open(os.path.join(save_dir_behav, "gamma_cp_list.pkl"), "wb") as f:
            pickle.dump(gamma_cp_list, f)
        with open(os.path.join(save_dir_behav, "gamma_ob_list.pkl"), "wb") as f:
            pickle.dump(gamma_ob_list, f)

        #add saving of rnn_activity
        with open(os.path.join(save_dir_rnn, f"rnn_activity_gamma.pkl"), "wb") as f:
            pickle.dump(gamma_rnn_activity, f)
        with open(os.path.join(save_dir_fp, f"fps_gamma_model.pkl"), "wb") as f:
            pickle.dump(gamma_fp, f)


if analysis == 'rollout' or analysis == 'all':
    # influence of rollout
    rollouts = [5, 10,20, 30, 40, 50, 75, 100, 150, 200] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'rollouts':rollouts,'states':[]}

    rollout_dict = {}
    rollout_cp_list = [] 
    rollout_ob_list = []
    rollout_models = []
    rollout_rnn_activity = []
    rollout_fp = []
    
    for g, rollout in enumerate(rollouts):

        file_names= data_dir+f"*_V3_0.95g_0.0rm_{rollout}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        models = [f for f in models if float(f.split("\\")[-1].split("_")[0]) > 5] # remove items in file_names that begin with "-"
        print(rollout, len(models))

        for m,model in enumerate(models):

            all_states, rnn_activity, loaded_model = model_rnn.rnn_predict(rnn_model = rnn_model,
                model_path = model,
                epochs=epochs)
            fp = utils_fp.find_fixed_points(loaded_model, rnn_activity)
            all_param_states['states'].append(all_states)

            rollout_dict[m, g] = {"rollout", rollout}
            rollout_cp_list.append(all_states[0,0])
            rollout_ob_list.append(all_states[0,1]) 
            rollout_models.append(model)  
            rollout_rnn_activity.append(rnn_activity)
            rollout_fp.append(fp)

        with open(os.path.join(save_dir_behav, "rollout_all_param_states.pkl"), "wb") as f:
            pickle.dump(all_param_states, f)
        with open(os.path.join(save_dir_behav, "rollout_dict.pkl"), "wb") as f:
            pickle.dump(rollout_dict, f)
        with open(os.path.join(save_dir_behav, "rollout_cp_list.pkl"), "wb") as f:
            pickle.dump(rollout_cp_list, f)
        with open(os.path.join(save_dir_behav, "rollout_ob_list.pkl"), "wb") as f:
            pickle.dump(rollout_ob_list, f)
        with open(os.path.join(save_dir_rnn, f"rnn_activity_rollout.pkl"), "wb") as f:
            pickle.dump(rollout_rnn_activity, f)
        with open(os.path.join(save_dir_fp, f"fps_rollout_model.pkl"), "wb") as f:
            pickle.dump(rollout_fp, f)

# introduce variables into sampling
if analysis == 'preset' or analysis == 'all':
    # influence of preset

    presets = [0.0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'presets':presets,'states':[]}

    preset_dict = {}
    preset_cp_list = [] 
    preset_ob_list = []
    preset_models = []
    preset_rnn_activity = []
    preset_fp = []
    
    for g, preset in enumerate(presets):

        file_names = data_dir+f"*_V3_0.95g_{preset}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        models = [f for f in models if float(f.split("\\")[-1].split("_")[0]) > 5] # remove items in file_names that begin with "-"
        print(preset, len(models))

        for m,model in enumerate(models):
            all_states, rnn_activity, loaded_model = model_rnn.rnn_predict(rnn_model = rnn_model,
                model_path = model,
                epochs=epochs)
            fp = utils_fp.find_fixed_points(loaded_model, rnn_activity)
            all_param_states['states'].append(all_states)

            preset_dict[m, g] = {"preset", preset}
            preset_cp_list.append(all_states[0,0])
            preset_ob_list.append(all_states[0,1])   
            preset_models.append(model)
            preset_rnn_activity.append(rnn_activity)
            preset_fp.append(fp)

        with open(os.path.join(save_dir_behav, "preset_all_param_states.pkl"), "wb") as f:
            pickle.dump(all_param_states, f)
        with open(os.path.join(save_dir_behav, "preset_dict.pkl"), "wb") as f:
            pickle.dump(preset_dict, f)
        with open(os.path.join(save_dir_behav, "preset_cp_list.pkl"), "wb") as f:
            pickle.dump(preset_cp_list, f)
        with open(os.path.join(save_dir_behav, "preset_ob_list.pkl"), "wb") as f:
            pickle.dump(preset_ob_list, f)
        with open(os.path.join(save_dir_rnn, f"rnn_activity_preset.pkl"), "wb") as f:
            pickle.dump(preset_rnn_activity, f)
        with open(os.path.join(save_dir_fp, f"fps_preset_model.pkl"), "wb") as f:
            pickle.dump(preset_fp, f)

if analysis == 'scale' or analysis == 'all':
    # influence of scale

    scales =  [0.25, 0.5,0.75, 1.0, 1.25, 1.5] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'scales':scales,'states':[]}

    scale_dict = {}
    scale_cp_list = [] 
    scale_ob_list = []
    scale_models = []
    scale_rnn_activity = []
    scale_fp = []
    
    for g, scale in enumerate(scales):

        # file_names= data_dir+f"*_V5_0.95g_0.0rm_50bz_0.0td_{scale}tds_64n_50000e_10md_5.0rz_*s.pth"
        file_names = data_dir+f"*_V3_0.95g_0.0rm_100bz_0.0td_{scale}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth" 
        models = glob.glob(file_names)
        models = [f for f in models if float(f.split("\\")[-1].split("_")[0]) > 5] # remove items in file_names that begin with "-"
        print(scale, len(models))


        for m,model in enumerate(models):
            all_states, rnn_activity, loaded_model = model_rnn.rnn_predict(rnn_model = rnn_model,
                model_path = model,
                epochs=epochs)
            fp = utils_fp.find_fixed_points(loaded_model, rnn_activity)
            all_param_states['states'].append(all_states)

            scale_dict[m, g] = {"scale", scale}
            scale_cp_list.append(all_states[0,0])
            scale_ob_list.append(all_states[0,1])   
            scale_models.append(model)
            scale_rnn_activity.append(rnn_activity)
            scale_fp.append(fp)

        with open(os.path.join(save_dir_behav, "scale_all_param_states.pkl"), "wb") as f:
            pickle.dump(all_param_states, f)
        with open(os.path.join(save_dir_behav, "scale_dict.pkl"), "wb") as f:
            pickle.dump(scale_dict, f)
        with open(os.path.join(save_dir_behav, "scale_cp_list.pkl"), "wb") as f:
            pickle.dump(scale_cp_list, f)
        with open(os.path.join(save_dir_behav, "scale_ob_list.pkl"), "wb") as f:
            pickle.dump(scale_ob_list, f)
        with open(os.path.join(save_dir_rnn, f"rnn_activity_scale.pkl"), "wb") as f:
            pickle.dump(scale_rnn_activity, f)
        with open(os.path.join(save_dir_fp, f"fps_scale_model.pkl"), "wb") as f:
            pickle.dump(scale_fp, f)

#combined_dict doesn't work because of overlapping keys

if analysis == 'all': 
    cp_array  = [gamma_cp_list, rollout_cp_list, preset_cp_list, scale_cp_list]
    ob_array  = [gamma_ob_list, rollout_ob_list, preset_ob_list, scale_ob_list]
    mod_array = [gamma_models, rollout_models, preset_models, scale_models]
    rnn_activity_array = [gamma_rnn_activity, rollout_rnn_activity, preset_rnn_activity, scale_rnn_activity]
    fp_array = [gamma_fp, rollout_fp, preset_fp, scale_fp]
    dict_array = [gamma_dict, rollout_dict, preset_dict, scale_dict]

    with open(os.path.join(save_dir_behav, "combined_cp_array_filtered.pkl"), "wb") as f:
        pickle.dump(cp_array, f)
    with open(os.path.join(save_dir_behav, "combined_ob_array_filtered.pkl"), "wb") as f:
        pickle.dump(ob_array, f)
    with open(os.path.join(save_dir_behav, "combined_mod_array_filtered.pkl"), "wb") as f:
        pickle.dump(mod_array, f)
    with open(os.path.join(save_dir_rnn, "combined_rnn_activity_array.pkl"), "wb") as f:
        pickle.dump(rnn_activity_array, f)
    with open(os.path.join(save_dir_fp, "combined_fp_array.pkl"), "wb") as f:
        pickle.dump(fp_array, f)
    with open(os.path.join(save_dir_behav, "combined_dict_array.pkl"), "wb") as f:
        pickle.dump(dict_array, f)

    # combined_all = {
    #     'cp': cp_array,
    #     'ob': ob_array,
    #     'models': mod_array,
    #     'rnn_activity': rnn_activity_array,
    #     'fp': fp_array,
    #     'dicts': dict_array
    # }
    # with open(os.path.join(save_dir_behav, "combined_all.pkl"), "wb") as f:
    #     pickle.dump(combined_all, f)
