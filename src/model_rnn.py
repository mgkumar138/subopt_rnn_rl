'''
Holds functions for the RNN model itself, and running it to get behav and rnn activity
'''
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Categorical
import numpy as np
from tasks import PIE_CP_OB_v2
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    """
    Actor-Critic model with RNN for sequential decision making tasks.
    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layer in the RNN.
        action_dim (int): Number of actions in the action space.
        gain (float): Gain for weight initialization.
        noise (float): Variance of noise added to the model.
        bias (bool): Whether to include bias in linear layers.
    """
    def __init__(self, input_dim, hidden_dim, action_dim, gain=1.5, noise=0.0, bias=False):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gain = gain
        self.noise = noise  # Include the noise variance as an argument
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh',bias=bias)
        self.actor = nn.Linear(hidden_dim, action_dim,bias=bias)
        self.critic = nn.Linear(hidden_dim, 1,bias=bias)
        self.init_weights()

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                init.normal_(param, mean=0, std=1/(self.input_dim**0.5))
            elif 'weight_hh' in name:
                init.normal_(param, mean=0, std=self.gain / self.hidden_dim**0.5)
            elif 'bias_ih' in name or 'bias_hh' in name:
                init.constant_(param, 0)

        for layer in [self.actor, self.critic]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.normal_(param, mean=0, std=1/self.hidden_dim)
                elif 'bias' in name:
                    init.constant_(param, 0)

    def forward(self, x, hx):
        r, h = self.rnn(x, hx)
        r = r.squeeze(1)
        critic_value = self.critic(r)

        return self.actor(r), critic_value, h #this return in analyze_rnn.py
    
        #return self.actor(r), self.critic(r), h #written like this in analyze_rnn_fp

def rnn_predict(
        rnn_model, 
        model_path, 
        hidden_dim = 64, 
        trials = 200,
        contexts = ["change-point", "oddball"],
        epochs=100, 
        reset_memory=0.0):
    '''
    Runs the RNN model given a model_path to pretrained weights
    -returns all_states, rnn_activity
    '''
    
    model = rnn_model
    model.load_state_dict(torch.load(model_path))

    Hs, As, Cs, Rs, Os = [], [], [], [], []
    Hs_all, Os_all = [], []

    all_states = np.zeros([epochs, len(contexts), 5, trials])
    rnn_activity = np.zeros([epochs, len(contexts)], dtype=object)

    for epoch in range(epochs):
        Hs = []; As = []; Cs = []; Rs = []; Os = []
        for tt, context in enumerate(contexts):
            env = PIE_CP_OB_v2(condition=context, 
                               max_time=300, 
                               total_trials=trials, 
                               train_cond=False, max_displacement=10, 
                               reward_size=5)

            h, a, c, r, o = [],[],[], [], []
            hx = torch.randn(1, 1, hidden_dim) * 1 / hidden_dim**0.5
            for trial in range(trials):

                next_obs, done = env.reset()
                norm_next_obs = env.normalize_states(next_obs)
                next_state = np.concatenate([norm_next_obs, env.context, np.array([0.0])])
                next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

                hx = hx.detach()

                while not done:
                    
                    # memory check not in analyze_rnn_fp 
                    if np.random.random_sample() < reset_memory:
                        hx = (torch.randn(1, 1, hidden_dim) * 1 / hidden_dim**0.5)

                    actor_logits, critic_value, hx = model(next_state, hx)
                    probs = Categorical(logits=actor_logits)
                    action = probs.sample()

                    # Take action and observe reward
                    next_obs, reward, done = env.step(action.item())

                    h.append(hx[0,0]), a.append(actor_logits[0]), c.append(critic_value[0]), r.append(reward), o.append(env.hazard_trigger)

                    # Prep next state
                    norm_next_obs = env.normalize_states(next_obs)
                    next_state = np.concatenate([norm_next_obs, env.context, np.array([reward])])
                    next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

            
            Hs.append(h), As.append(a), Cs.append(c), Rs.append(r), Os.append(o)
            Hs_all.append(torch.stack(h))
            Os_all.append(torch.tensor(o))

            rnn_activity[epoch, tt] = (h, a, c, r, o)
            all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
        
    hidden_states_all = [torch.vstack(h).detach().unsqueeze(0).numpy() for h in Hs_all]

    return all_states, hidden_states_all, model