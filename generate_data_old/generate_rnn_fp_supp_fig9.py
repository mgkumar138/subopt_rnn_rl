#%%
'''
Script to find fixed points in RNN for each hyperparameter setting for Fig. 9
'''

from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from tasks import PIE_CP_OB_v2
import matplotlib.pyplot as plt
from torch.nn import init
from utils_funcs import get_lrs_v2, saveload, plot_behavior, ActorCritic
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from copy import deepcopy
import os
import glob
from fixed_point_finder.FixedPointFinderTorch import FixedPointFinderTorch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

contexts = ["change-point", "oddball"]  # "change-point","oddball"
num_contexts = len(contexts)
train_cond = False
reward_size = 5
max_displacement = 10
max_time = 300
n_trials = 200
epochs = 6
N_INITS = 1024  # The number of initial states to provide

input_dim = 6 + 3  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.
hidden_dim = 64  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.

seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)


def compute_unstable_modes(fps):
    '''

    Args:
        fps: pass in unique_fps output from fpf.find_fixed_points

    Returns:
        unstable_fp_locs: (n_unstable_fps, hidden_state_size)
        unstable_fp_modes: (n_unstable_fps, hidden_state_size)
    '''
    unstable_fp_locs, unstable_fp_modes = [], []
    for i, fp in enumerate(fps):
        e_vals = fp.eigval_J_xstar[0]
        is_stable = np.all(np.abs(e_vals) < 1.0)
        if not is_stable:
            leading_unstable_mode = np.real(fp.eigvec_J_xstar[0][:, 0])
            unstable_fp_locs.append(fp.xstar[0])
            unstable_fp_modes.append(leading_unstable_mode)
    return np.stack(unstable_fp_locs), np.stack(unstable_fp_modes)

def cosine_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_dynamics_corr_near_unstable_fps(fps, hidden_states, tol = 10, max_corr = 1000):
    '''

    Args:
        fps: pass in unique_fps output from fpf.find_fixed_points (n_unstable_fps)
        hidden_states: (n_steps, hidden_state_size)
        tol: distance from an unstable FP to be defined as "close"

    Returns: list of cosine similarities b/t hidden state update near unstable mode and the unstable mode

    '''
    n_steps = hidden_states.shape[0]
    unstable_fp_locs, unstable_fp_modes = compute_unstable_modes(fps)
    distances = cdist(hidden_states, unstable_fp_locs) # (n_steps, n_unstable_fps)
    within_tol_idxes = np.nonzero(distances < tol)
    n_pairs = within_tol_idxes[0].shape[0]
    cos_sims = []
    if n_pairs > 0:
        for i in range(min(n_pairs, max_corr)):
            j, k = within_tol_idxes[0][i], within_tol_idxes[1][i] #j is hidden state step idx, k is unstable fp idx
            if j < n_steps - 1:
                hs_step_vec = hidden_states[j+1] - hidden_states[j]
                cos_sims.append(cosine_sim(hs_step_vec, unstable_fp_modes[k]))
    # else:
    #     print('nothing found')
    # print(np.mean(cos_sims), 'mean cos sim over {} pairs'.format(min(n_pairs, max_corr)))
    return cos_sims


def find_fixed_points(model, hidden_states, load_from_file = True, model_name=''):
    stable_unstable_fps = np.zeros([len(contexts),3])
    fps_list = []
    for context, context_name in enumerate(contexts):

        NOISE_SCALE = 0.5  # Standard deviation of noise added to initial states

        '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
        descriptions of available hyperparameters.'''
        fpf_hps = {
            'max_iters': 10000,
            'lr_init': 1.,
            'outlier_distance_scale': 10.0,
            'verbose': False,
            'super_verbose': False}

        # Setup the fixed point finder
        fpf = FixedPointFinderTorch(model.rnn, **fpf_hps)

        initial_states = fpf.sample_states(hidden_states[context],
        	n_inits=N_INITS,
        	noise_scale=NOISE_SCALE)

        # Study the system in the absence of input pulses (e.g., all inputs are 0 except the context cue)
        inputs = np.zeros((1, model.input_dim))
        inputs[:, -3+context] = 1.0

        # Run the fixed point finder
        unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

        # fp_fname = './saved_fp/unique_fps_context_{}_model_{}.pk'.format(context, model_name)
        # if os.path.exists(fp_fname) and load_from_file:
        #     with open(fp_fname, 'rb') as f:
        #         unique_fps = pk.load(f)
        # else:
        #     unique_fps, all_fps = fpf.find_fixed_points(noisy_state_traj[0].copy(), inputs)
        #     with open(fp_fname, 'wb') as f:
        #         pk.dump(unique_fps, f)

        # Visualize identified fixed points with overlaid RNN state trajectories
        # All visualized in the 3D PCA space fit the the example RNN states.
        stable_fp_cnt = 0
        unstable_fp_cnt = 0
        for i, fp in enumerate(unique_fps):
            e_vals = fp.eigval_J_xstar[0]
            is_stable = np.all(np.abs(e_vals) < 1.0)
            if is_stable:
                stable_fp_cnt += 1
            else:
                unstable_fp_cnt += 1

        stable_unstable_fps[context] = np.array([stable_fp_cnt, unstable_fp_cnt, stable_fp_cnt+unstable_fp_cnt])
        fps_list.append(unique_fps)
    return stable_unstable_fps, fps_list


def get_states_hs(model_path,reset_memory=0.0):
    model = ActorCritic(input_dim, hidden_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    print('Load Model')
    all_states = np.zeros([epochs, num_contexts, 5, n_trials])
    Hs_all = [[],[]]
    Os_all = [[],[]]
    # get rnn, actor, critic activity
    for epoch in range(epochs):
        Hs = []
        As = []
        Cs = []
        Rs = []
        Os = []
        for tt, context in enumerate(contexts):
            env = PIE_CP_OB_v2(condition=context, max_time=max_time, total_trials=n_trials,
                            train_cond=train_cond, max_displacement=max_displacement, reward_size=reward_size)

            h, a, c, r, o = [], [], [], [], []
            hx = torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5 
            for trial in range(n_trials):

                next_obs, done = env.reset()
                norm_next_obs = env.normalize_states(next_obs)
                next_state = np.concatenate([norm_next_obs, env.context, np.array([0.0])])
                next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

                while not done:
                    if np.random.random_sample()< reset_memory:
                        hx = (torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5)
                
                    actor_logits, critic_value, hx = model(next_state, hx)
                    probs = Categorical(logits=actor_logits)
                    action = probs.sample()

                    # Take action and observe reward
                    next_obs, reward, done = env.step(action.item())

                    h.append(hx[0, 0]), a.append(actor_logits[0]), c.append(critic_value[0]), r.append(reward), o.append(
                        env.hazard_trigger)

                    # Prep next state
                    norm_next_obs = env.normalize_states(next_obs)
                    next_state = np.concatenate([norm_next_obs, env.context, np.array([reward])])
                    next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

            Hs.append(h), As.append(a), Cs.append(c), Rs.append(r), Os.append(o)
            Hs_all[tt].append(torch.stack(h))
            Os_all[tt].append(torch.tensor(o))

            all_states[epoch, tt] = np.array(
                [env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
    hidden_states_all = [torch.vstack(h).detach().unsqueeze(0).numpy() for h in Hs_all]
    return hidden_states_all, model

def get_mean_ci(x, valididx):
    m = []
    s = []
    numparams = x.shape[0]
    for p in range(numparams):
        idx = int(valididx[p])
        m.append(np.mean(x[p,:idx],axis=0))
        s.append(np.std(x[p,:idx],axis=0)/np.sqrt(idx))
    m = np.array(m)
    s = np.array(s)
    return m, s

def plot_param_fps(param, fps, xlabel, validms, logx=False, legend=False):

    # saveload(f'./analysis/{xlabel}_fps_{N_INITS}',[param, fps, validms], 'save')

    labels = ['CP', 'OB']
    colors= ['orange', 'brown']
    markers = ['o','x','v']
    markers=[None, None, None]
    titles= ['Stable FPs', 'Unstable FPs', 'All FPs']
    f,axs = plt.subplots(3,1,figsize=(3.5,2*3))

    for i in range(3):

        for c in range(2):
            m,s = get_mean_ci(fps[:,:,c],validms)

            axs[i].plot(param, m[:,i], label=labels[c], color=colors[c], marker=markers[i])
            axs[i].fill_between(x=param, y1=m[:,i]-s[:,i], y2=m[:,i]+s[:,i], alpha=0.2, color=colors[c])
        
        axs[i].set_ylabel(titles[i])

        ax2 = axs[i].twinx()
        dffps = fps[:,:,0,i] - fps[:,:,1,i]
        m,s = get_mean_ci(dffps,validms)
        axs[i].plot([], [], label='CP-OB', color='k', marker=markers[i])
        # axs[i].plot(param, m, label='CP-OB', color='k', marker=markers[i])
        # axs[i].fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color='k')


        ax2.plot(param, m, label='CP-OB', color='k', marker=markers[i])
        ax2.fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color='k')
        ax2.set_ylabel('$\Delta$ FPs')

    axs[-1].legend(fontsize=8)
    axs[-1].set_xlabel(xlabel)
    f.tight_layout()
    # f.savefig(f'./analysis/{xlabel}_fp_{n_trials*epochs}.png')
    # f.savefig(f'./analysis/{xlabel}_fp_{n_trials*epochs}.svg')


#%%
analysis = 'gamma'
data_dir = "./model_params_101000/"
seeds = 50
max_corr = 1000
tol = 10

if analysis == 'gamma':

    gammas = [0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1] 
    fps = np.zeros([len(gammas),seeds, len(contexts), 3])
    cos_sims = np.zeros([len(gammas),seeds, len(contexts), max_corr])
    validms = np.zeros(len(gammas), dtype=int)

    for g, gamma in enumerate(gammas):
        file_names= data_dir+f"*_V3_{gamma}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        model_paths = glob.glob(file_names)
        validms[g] = len(model_paths)
        print(gamma, validms[g])

        for m,model_path in enumerate(model_paths):
                
            hidden_states_all, model = get_states_hs(model_path)
            fps[g, m], fps_list = find_fixed_points(model, hidden_states_all)

            for c,unique_fps in enumerate(fps_list):
                cos_sims[g,m,c] = compute_dynamics_corr_near_unstable_fps(unique_fps, hidden_states_all[c][0][:N_INITS], tol=tol, max_corr=max_corr)

        print(gamma, np.mean(fps[g,:m],axis=0))

    plot_param_fps(gammas, fps, '$\gamma$', validms)

if analysis == 'scale':

    scales = [0.25, 0.5,0.75, 1.0, 1.25, 1.5]  # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    fps = np.zeros([len(scales),seeds, len(contexts), 3])
    validms = np.zeros(len(scales), dtype=int)
    cos_sims = np.zeros([len(scales),seeds, len(contexts), max_corr])
    for g, scale in enumerate(scales):
        
        file_names = data_dir+f"*_V3_0.95g_0.0rm_100bz_0.0td_{scale}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth" 
        model_paths = glob.glob(file_names)
        validms[g] = len(model_paths)
        print(scale, validms[g])

        for m,model_path in enumerate(model_paths):
                
            hidden_states_all, model = get_states_hs(model_path)
            fps[g, m], fps_list = find_fixed_points(model, hidden_states_all)

            for c,unique_fps in enumerate(fps_list):
                cos_sims[g,m,c] = compute_dynamics_corr_near_unstable_fps(unique_fps, hidden_states_all[c][0][:N_INITS], tol=tol, max_corr=max_corr)


        print(scale, np.mean(fps[g,:m],axis=0))

    plot_param_fps(scales, fps,  '$\\beta_{\delta}$', validms)


if analysis == 'rollout':

    rollouts = [5, 10,20, 30, 40, 50, 75, 100, 150, 200] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    fps = np.zeros([len(rollouts),seeds, len(contexts), 3])
    validms = np.zeros(len(rollouts), dtype=int)
    cos_sims = np.zeros([len(rollouts),seeds, len(contexts), max_corr])
    for g, rollout in enumerate(rollouts):
        
        file_names= data_dir+f"*_V3_0.95g_0.0rm_{rollout}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        model_paths = glob.glob(file_names)
        validms[g] = len(model_paths)
        print(rollout, validms[g])

        for m,model_path in enumerate(model_paths):
                
            hidden_states_all, model = get_states_hs(model_path)
            fps[g, m], fps_list = find_fixed_points(model, hidden_states_all)

            for c,unique_fps in enumerate(fps_list):
                cos_sims[g,m,c] = compute_dynamics_corr_near_unstable_fps(unique_fps, hidden_states_all[c][0][:N_INITS], tol=tol, max_corr=max_corr)


        print(rollout, np.mean(fps[g,:m],axis=0))

    plot_param_fps(rollouts, fps,  '$\\tau$', validms)

if analysis == 'preset':

    presets = [0.0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]  # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    fps = np.zeros([len(presets),seeds, len(contexts), 3])
    validms = np.zeros(len(presets), dtype=int)
    cos_sims = np.zeros([len(presets),seeds, len(contexts), max_corr])
    for g, preset in enumerate(presets):
        
        file_names = data_dir+f"*_V3_0.95g_{preset}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        model_paths = glob.glob(file_names)
        validms[g] = len(model_paths)
        print(preset, validms[g])

        for m,model_path in enumerate(model_paths):
                
            hidden_states_all, model = get_states_hs(model_path, reset_memory=preset)
            fps[g, m], fps_list = find_fixed_points(model, hidden_states_all)
            for c,unique_fps in enumerate(fps_list):
                cos_sims[g,m,c] = compute_dynamics_corr_near_unstable_fps(unique_fps, hidden_states_all[c][0][:N_INITS], tol=tol, max_corr=max_corr)


        print(preset, np.mean(fps[g,:m],axis=0))

    plot_param_fps(presets, fps,  '$p_{reset}$', validms)
