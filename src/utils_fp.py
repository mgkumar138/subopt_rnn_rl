'''
Utility functions for fixed point analysis of RNNs.
'''




def find_fixed_points(model, hidden_states, load_from_file = True, model_name=''):
    stable_unstable_fps = np.zeros([len(contexts),3])
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
    return stable_unstable_fps

def find_fixed_points_v2(model, hidden_states, context=0, separate_traj = None, pca_dim=2, load_from_file = True, show_plots = True,model_name='', plot_params={}):
    context_names = ['CP','OB']
    state_traj = hidden_states[context]
    # NOISE_SCALE = 1.0  # Standard deviation of noise added to initial states
    # N_INITS = state_traj.shape[1]  # The number of initial states to provide
    # noise = np.random.randn(*state_traj.shape)
    # noisy_state_traj = state_traj + noise

    # cut the number of states
    N_INITS = n_trials
    state_traj = state_traj[:,:N_INITS,:]
    noisy_state_traj = state_traj + np.random.randn(*state_traj.shape)



    #state_traj += noise
    '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
	descriptions of available hyperparameters.'''
    fpf_hps = {
        'max_iters': 1000,
        'lr_init': 1.,
        'outlier_distance_scale': 100.0,
        'verbose': True,
        'super_verbose': True}

    # Setup the fixed point finder
    fpf = FixedPointFinderTorch(model.rnn, **fpf_hps)

    # initial_states = fpf.sample_states(hidden_states,
    # 	n_inits=N_INITS,
    # 	noise_scale=NOISE_SCALE)
    initial_states = state_traj[0]

    # Study the system in the absence of input pulses (e.g., all inputs are 0)
    #inputs = np.zeros([1, model.input_dim])
    inputs = np.zeros((N_INITS, model.input_dim))
    inputs[:, -3+context] = 1.0
    # Run the fixed point finder
    unique_fps, all_fps = fpf.find_fixed_points(noisy_state_traj[0].copy(), inputs)
    # fp_fname = './saved_fp/unique_fps_context_{}_model_{}.pk'.format(context,model_name)
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
    if show_plots:
        fig = plt.figure(figsize=(4,3))
        fig.suptitle(f'{context_names[context]}: $\gamma={gamma}$')
        fig = plot_fps(unique_fps, state_traj,fig=fig,
                       plot_batch_idx=None,
                       plot_start_time=0, context=context,model=model, separate_traj=separate_traj[context],pca_dim=pca_dim, hazards=None, 
                       model_name=model_name, nfps=[stable_fp_cnt, unstable_fp_cnt, stable_fp_cnt+unstable_fp_cnt])
        
    return stable_fp_cnt, unstable_fp_cnt, unique_fps, state_traj, fig








def make_eigenvalue_plot(W):
    plt.figure(dpi=150, figsize=(2.5, 2.5))
    t = np.linspace(0, np.pi * 2, 100)
    plt.plot(np.cos(t), np.sin(t), color='gray', linewidth=1, alpha=0.5)
    plt.xlabel(r'$\Re(\lambda)$', fontsize=16)
    plt.ylabel(r'$\Im(\lambda)$', fontsize=16)
    evs = np.linalg.eigvals(W)
    evs_real = np.real(evs)
    evs_imag = np.imag(evs)
    arg = np.argsort(evs_real)
    np.random.shuffle(arg)
    evs_real = evs_real[arg]
    evs_imag = evs_imag[arg]
    plt.scatter(evs_real, evs_imag, s=20, edgecolor='black', linewidth=0.1, alpha=0.5)
    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_aspect('equal')
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    plt.tight_layout()
