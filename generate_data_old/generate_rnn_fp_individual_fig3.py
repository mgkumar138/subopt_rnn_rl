# %%

'''
Generate figure 3
'''
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import pickle as pk
from fixed_point_finder.plot_utils import plot_fps
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
reward_size = 2
max_displacement = 10
max_time = 300
n_trials = 200
epochs = 1

input_dim = 6 + 3  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.
hidden_dim = 64  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.

seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)

# weaker top, better bot
#model_path = "./model_params/12.0_V3_0.0ns_Nonelb_Noneub_0.7g_64n_40000e_2s.pth"
# noinspection PyPackageRequirements
# model_path = "./model_params/36.0_V3_0.0ns_Nonelb_Noneub_0.95g_64n_40000e_2s.pth"

gamma = 0.95
rollout = 100
preset = 0.0
beta = 1.0

model_dir = f"./model_params_101000/*_V3_{gamma}g_{preset}rm_{rollout}bz_0.0td_{beta}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
files = glob.glob(model_dir)
sorted_file_paths = sorted(files, key=lambda x: float(x.split('/')[2].split('_')[0]))
model_path = sorted_file_paths[-1]
print(model_path)


#moved to utils_fp
# def make_eigenvalue_plot(W):
#     plt.figure(dpi=150, figsize=(2.5, 2.5))
#     t = np.linspace(0, np.pi * 2, 100)
#     plt.plot(np.cos(t), np.sin(t), color='gray', linewidth=1, alpha=0.5)
#     plt.xlabel(r'$\Re(\lambda)$', fontsize=16)
#     plt.ylabel(r'$\Im(\lambda)$', fontsize=16)
#     evs = np.linalg.eigvals(W)
#     evs_real = np.real(evs)
#     evs_imag = np.imag(evs)
#     arg = np.argsort(evs_real)
#     np.random.shuffle(arg)
#     evs_real = evs_real[arg]
#     evs_imag = evs_imag[arg]
#     plt.scatter(evs_real, evs_imag, s=20, edgecolor='black', linewidth=0.1, alpha=0.5)
#     ax = plt.gca()
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.set_aspect('equal')
#     ax.xaxis.set_tick_params(labelsize=16)
#     ax.yaxis.set_tick_params(labelsize=16)
#     ax.set_xticks([-1, 0, 1])
#     ax.set_yticks([-1, 0, 1])
#     plt.tight_layout()

#labeled as v2 and placed into utils_fp
# def find_fixed_points(model, hidden_states, context=0, separate_traj = None, pca_dim=2, load_from_file = True, show_plots = True,model_name='', plot_params={}):
#     context_names = ['CP','OB']
#     state_traj = hidden_states[context]
#     # NOISE_SCALE = 1.0  # Standard deviation of noise added to initial states
#     # N_INITS = state_traj.shape[1]  # The number of initial states to provide
#     # noise = np.random.randn(*state_traj.shape)
#     # noisy_state_traj = state_traj + noise

#     # cut the number of states
#     N_INITS = n_trials
#     state_traj = state_traj[:,:N_INITS,:]
#     noisy_state_traj = state_traj + np.random.randn(*state_traj.shape)



#     #state_traj += noise
#     '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
# 	descriptions of available hyperparameters.'''
#     fpf_hps = {
#         'max_iters': 1000,
#         'lr_init': 1.,
#         'outlier_distance_scale': 100.0,
#         'verbose': True,
#         'super_verbose': True}

#     # Setup the fixed point finder
#     fpf = FixedPointFinderTorch(model.rnn, **fpf_hps)

#     # initial_states = fpf.sample_states(hidden_states,
#     # 	n_inits=N_INITS,
#     # 	noise_scale=NOISE_SCALE)
#     initial_states = state_traj[0]

#     # Study the system in the absence of input pulses (e.g., all inputs are 0)
#     #inputs = np.zeros([1, model.input_dim])
#     inputs = np.zeros((N_INITS, model.input_dim))
#     inputs[:, -3+context] = 1.0
#     # Run the fixed point finder
#     unique_fps, all_fps = fpf.find_fixed_points(noisy_state_traj[0].copy(), inputs)
#     # fp_fname = './saved_fp/unique_fps_context_{}_model_{}.pk'.format(context,model_name)
#     # if os.path.exists(fp_fname) and load_from_file:
#     #     with open(fp_fname, 'rb') as f:
#     #         unique_fps = pk.load(f)
#     # else:
#     #     unique_fps, all_fps = fpf.find_fixed_points(noisy_state_traj[0].copy(), inputs)
#     #     with open(fp_fname, 'wb') as f:
#     #         pk.dump(unique_fps, f)

#     # Visualize identified fixed points with overlaid RNN state trajectories
#     # All visualized in the 3D PCA space fit the the example RNN states.
#     stable_fp_cnt = 0
#     unstable_fp_cnt = 0
#     for i, fp in enumerate(unique_fps):
#         e_vals = fp.eigval_J_xstar[0]
#         is_stable = np.all(np.abs(e_vals) < 1.0)
#         if is_stable:
#             stable_fp_cnt += 1
#         else:
#             unstable_fp_cnt += 1
#     if show_plots:
#         fig = plt.figure(figsize=(4,3))
#         fig.suptitle(f'{context_names[context]}: $\gamma={gamma}$')
#         fig = plot_fps(unique_fps, state_traj,fig=fig,
#                        plot_batch_idx=None,
#                        plot_start_time=0, context=context,model=model, separate_traj=separate_traj[context],pca_dim=pca_dim, hazards=None, 
#                        model_name=model_name, nfps=[stable_fp_cnt, unstable_fp_cnt, stable_fp_cnt+unstable_fp_cnt])
        
#     return stable_fp_cnt, unstable_fp_cnt, unique_fps, state_traj, fig


#already setup in sample_behavior.py

# model = ActorCritic(input_dim, hidden_dim, action_dim)
# if model_path is not None:
#     model.load_state_dict(torch.load(model_path))
#     print('Load Model')


# all_states = np.zeros([epochs, num_contexts, 5, n_trials])
# Hs_all = [[],[]]
# Os_all = [[],[]]
# # get rnn, actor, critic activity
# for epoch in range(epochs):
#     Hs = []
#     As = []
#     Cs = []
#     Rs = []
#     Os = []
#     for tt, context in enumerate(contexts):
#         env = PIE_CP_OB_v2(condition=context, max_time=max_time, total_trials=n_trials,
#                            train_cond=train_cond, max_displacement=max_displacement, reward_size=reward_size)

#         h, a, c, r, o = [], [], [], [], []
#         hx = torch.randn(1, 1, hidden_dim) * 1 / hidden_dim
#         for trial in range(n_trials):

#             next_obs, done = env.reset()
#             norm_next_obs = env.normalize_states(next_obs)
#             next_state = np.concatenate([norm_next_obs, env.context, np.array([0.0])])
#             next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

#             while not done:
#                 actor_logits, critic_value, hx = model(next_state, hx)
#                 probs = Categorical(logits=actor_logits)
#                 action = probs.sample()

#                 # Take action and observe reward
#                 next_obs, reward, done = env.step(action.item())

#                 h.append(hx[0, 0]), a.append(actor_logits[0]), c.append(critic_value[0]), r.append(reward), o.append(
#                     env.hazard_trigger)

#                 # Prep next state
#                 norm_next_obs = env.normalize_states(next_obs)
#                 next_state = np.concatenate([norm_next_obs, env.context, np.array([reward])])
#                 next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

#         Hs.append(h), As.append(a), Cs.append(c), Rs.append(r), Os.append(o)
#         Hs_all[tt].append(torch.stack(h))
#         Os_all[tt].append(torch.tensor(o))

#         all_states[epoch, tt] = np.array(
#             [env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])

# saveload('state_hs',[all_states, Hs_all], 'save')

context=0
pca_dim = 2

hidden_states_all = [torch.vstack(h).detach().unsqueeze(0).numpy() for h in Hs_all]

#unused:
hazards = [torch.tensor(o).detach().numpy() for o in Os]
hazards_all = [torch.hstack(o).detach().numpy() for o in Os_all]

# hidden_states_all: concatenated hidden states across all epochs, used for FP search.
# Hs_all: hidden states separated by epoch. Used for plotting individual epoch trajectories. Set to None if just want to plot everything in Hs_all together.

#moved to main_fp.ipynb

# for context in range(2):
#     stable_fp_cnt, unstable_fp_cnt, unique_fps, state_traj, fig = find_fixed_points(model, hidden_states_all, context=context, separate_traj = Hs_all, pca_dim=pca_dim, show_plots=True,model_name='strong')
#     # fig.savefig(f'./analysis/fp_eg_{pca_dim}D_{gamma}_{context}c.png', dpi=300)
#     # fig.savefig(f'./analysis/fp_eg_{pca_dim}D_{gamma}_{context}c.svg')
#     print(stable_fp_cnt, unstable_fp_cnt)
