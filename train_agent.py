#%%
'''
Idea is to pretrain a vanilla RNN using RL on several epochs of the helicopter task so that it sort of knows what to do. 
The model weights are saved to sample epochs of 200 trials of each condition, similar to Nassar et al. 2021 and for additional analyses
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=False, help='total epochs', default=1000)
parser.add_argument('--trials', type=int, required=False, help='number of trials in each epoch', default=200)
parser.add_argument('--maxt', type=int, required=False, help='maxt', default=300)
parser.add_argument('--maxdisp', type=int, required=False, help='maxdisp', default=10)
parser.add_argument('--rewardsize', type=float, required=False, help='rewardsize', default=5)

parser.add_argument('--nrnn', type=int, required=False, help='number rnn units', default=64)
parser.add_argument('--seed', type=int, required=False, help='seed', default=0)
parser.add_argument('--ratio', type=float, required=False, help='train_eval_ratio', default=0.5)

# 4 different hyperparameters
parser.add_argument('--gamma', type=float, required=False, help='reward discount factor', default=0.95)
parser.add_argument('--tau', type=int, required=False, help='rollout buffer size', default=100)
parser.add_argument('--p_reset', type=float, required=False, help='probability of resetting memory with noise', default=0.0)
parser.add_argument('--beta_delta', type=float, required=False, help='scale TD error magnitude', default=1.0)

args, unknown = parser.parse_known_args()
print(args)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from src.tasks import PIE_CP_OB_v2
import matplotlib.pyplot as plt
from torch.nn import init
from src.utils_funcs import get_lrs_v2, saveload, plot_behavior, get_lrs_v3, plot_lrs
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from copy import deepcopy

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

# Env parameters
n_epochs = args.epochs  # number of epochs to train the model on. Similar to the number of times the agent is trained on the helicopter task. 
n_trials = args.trials  # number of trials per epoch for each condition.
train_epochs = n_epochs*args.ratio #n_epochs*0.5  # number of epochs where the helicopter is shown to the agent. if 0, helicopter is never shown.
contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)

# Task parameters
max_displacement = args.maxdisp # number of units each left or right moves.
max_time = args.maxt # int(5*300//max_displacement)
step_cost = 0 #-1/300  # penalize every additional step that the agent does not confirm. 
reward_size = args.rewardsize # smaller value means a tighter margin to get reward.
alpha = 1  # smoothing factor for action. 1 means no smoothing for discrete action steps, closer to 0 means more smoothing.

# Model Parameters
input_dim = 4+2+1  # observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos] + context vector is length 2 + 1 for reward
hidden_dim = args.nrnn  # size of RNN
action_dim = 3  # action space. 0 is left, 1 is right, 2 is confirm bucket position.
total_params = hidden_dim*(input_dim+hidden_dim + action_dim+1)
beta_ent = 0.0  # can be a small positive. did not significantly change results. 
seed = args.seed

# hyperparameters
gamma = args.gamma
rollout_size = args.tau
reset_memory = args.p_reset  # reset RNN activity after T trials
tdscale = args.beta_delta

# reduce learning rate if smaller rollout size, else model overfits to short trajectories
learning_rate = 0.0001/100 * rollout_size 

np.random.seed(seed)
torch.manual_seed(seed)

exptname = f"V3_{gamma}g_{reset_memory}rm_{rollout_size}bz_{tdscale}tds_{hidden_dim}n_{n_epochs}e_{max_displacement}md_{reward_size}rz_{seed}s"
print(exptname)
model_path = None


# Actor-Critic Network with RNN
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, gain=1.5):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gain = gain
        bias = False
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh', bias=bias)
        self.actor = nn.Linear(hidden_dim, action_dim, bias=bias)
        self.critic = nn.Linear(hidden_dim, 1, bias=bias)
        self.init_weights()

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                init.normal_(param, mean=0, std=1/(self.input_dim**0.5))  # 1/sqrtN scaling
            elif 'weight_hh' in name:
                init.normal_(param, mean=0, std=self.gain / self.hidden_dim**0.5)  # gain/sqrtN scaling
            elif 'bias_ih' in name or 'bias_hh' in name:
                init.constant_(param, 0)

        for layer in [self.actor, self.critic]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.normal_(param, mean=0, std=1/self.hidden_dim)  # 1/N scaling to initialize policy readout weights to be small
                elif 'bias' in name:
                    init.constant_(param, 0)

    def forward(self, x, hx):
        r, h = self.rnn(x, hx)
        r = r.squeeze(1)
        return self.actor(r), self.critic(r), h
    
def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


class RolloutBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropies = []

    def add_experience(self, state, action, reward, value, log_prob, entropy):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.entropies.clear()

def compute_advantages_and_returns(buffer, gamma, device):
    returns = []
    G = 0.0
    advantages = []
    buffer_size = len(buffer.rewards)

    for reward, value in zip(reversed(buffer.rewards), reversed(buffer.values)):
        G = reward + gamma * G
        returns.insert(0, G)

        # Calculate TD error
        td_error = G - value.detach().cpu().numpy()
        td_error *= tdscale

        advantages.insert(0, td_error)

    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    log_probs = torch.stack(buffer.log_probs)
    values = torch.stack(buffer.values).squeeze()

    return returns, advantages, log_probs, values

def train_with_rollouts(env, model, optimizer, epoch, n_trials, gamma, rollout_buffer_size):
    buffer = RolloutBuffer(rollout_buffer_size)

    totG = []
    totloss = []
    tottime = []

    hx = (torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5).to(device)
    trial_counter = 0
    
    while trial_counter < n_trials:
    
        next_obs, done = env.reset()
        norm_next_obs = env.normalize_states(next_obs)
        next_state = np.concatenate([norm_next_obs, env.context, np.array([0.0])])
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)
        hx = hx.detach() # Detach hx to prevent backpropagation through the entire history
        totR = 0
        
        while not done:

            if np.random.random_sample()< reset_memory:
                hx = (torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5).to(device)

            actor_logits, critic_value, hx = model(next_state, hx)
            probs = Categorical(logits=actor_logits)
            action = probs.sample()

            # Take action and observe reward
            next_obs, reward, done = env.step(action.item())
            totR += reward

            # Append experiences to rollout buffer
            buffer.add_experience(next_state, action, reward, critic_value, probs.log_prob(action), probs.entropy())

            # Prep next state
            norm_next_obs = env.normalize_states(next_obs)
            next_state = np.concatenate([norm_next_obs, env.context, np.array([reward])])
            next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)

        trial_counter += 1

        if len(buffer.rewards) >= rollout_buffer_size:
            # Computation for advantages and returns
            returns, advantages, log_probs, values = compute_advantages_and_returns(buffer, gamma, device)

            # Calculate losses
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = ((returns - values)**2).mean()
            entropy_loss = -torch.stack(buffer.entropies).mean()

            loss = actor_loss + 0.5 * critic_loss + beta_ent * entropy_loss

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            buffer.clear()

            totloss.append(to_numpy(loss))

        totG.append(totR)
        tottime.append(env.time)

    return np.array(totG), np.array(totloss), np.array(tottime)

# initialize untrained model
model = ActorCritic(input_dim, hidden_dim, action_dim).to(device)
store_params = []
store_params.append(deepcopy(model.state_dict()))
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # initialize optimizer for training

# load pretrained, if any
if model_path is not None:
    print('Load Model')
    model.load_state_dict(torch.load(model_path))

# store variables
epoch_perf = np.zeros([n_epochs, num_contexts, 4, n_trials])
all_states = np.zeros([n_epochs, num_contexts, 5, n_trials])
all_lrs = np.zeros([n_epochs, num_contexts, n_trials-1])
all_pes = np.zeros([n_epochs, num_contexts, n_trials-1])


# training loop
for epoch in range(n_epochs):

    if epoch < train_epochs:
        train_cond = True  # give helicopter position for these epochs to simulate train condition
    else:
        train_cond = False # dont give helicopter position for these epochs to simulate test condition
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # reinitialize optimizer for test conditon

    idxs = np.random.choice(np.arange(2), size=2, replace=False)  # randomize CP and OB conditions during training

    for tt in idxs:

        task_type = contexts[tt]

        env = PIE_CP_OB_v2(condition=task_type, max_time=max_time, total_trials=n_trials, 
                        train_cond=train_cond, max_displacement=max_displacement, reward_size=reward_size, step_cost=step_cost, alpha=alpha)

        totG, totloss,tottime = train_with_rollouts(env, model, optimizer, epoch=epoch, n_trials=n_trials, gamma=gamma, rollout_buffer_size=rollout_size)
        totloss = np.tile(np.mean(totloss),200)

        # save performance
        all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
        all_pes[epoch,tt], all_lrs[epoch, tt] = get_lrs_v2(all_states[epoch, tt])
        perf = abs(all_states[epoch, tt, 3] - all_states[epoch, tt,1])
        epoch_perf[epoch,tt] = np.array([totG, totloss, tottime, perf])

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Task {task_type}, G {np.mean(totG):.3f}, d {np.mean(epoch_perf[epoch,tt, 3]):.3f}, s {np.mean(all_lrs[epoch, tt]):.3f}")

        if epoch == n_epochs-1 or epoch == train_epochs-1:
            #plot last epochs behav data
            _ = env.render(epoch)


    perf = np.mean(abs(all_states[epoch,:, 3] - all_states[epoch,:,1]))

    # check if model has learned the training task
    if epoch == train_epochs-1 and perf < 32:
        store_params.append(deepcopy(model.state_dict()))

    # store model params after training on test condition IF model had learned the training condition
    if epoch == n_epochs-1 and len(store_params)>1:
        store_params.append(deepcopy(model.state_dict()))



colors = ['orange', 'brown']
labels = ['CP', 'OB']
plt.figure(figsize=(4*2,3*5))

plt.subplot(521)
for i in range(2):
    plt.plot(np.mean(epoch_perf[:,i,3],axis=1), color=colors[i], label=labels[i])
plt.xlabel('Epoch')
plt.ylabel('Heli - Bucket error') # should become more positive.
plt.axhline(32, color='r')
plt.axvline(train_epochs, color='b')

plt.subplot(522)
for i in range(2):
    plt.plot(np.mean(epoch_perf[:,i, 0],axis=1), color=colors[i], label=labels[i])
plt.xlabel('Epoch')
plt.ylabel('G')
plt.axvline(train_epochs, color='b')


plt.subplot(523)
for i in range(2):
    plt.plot(np.mean(epoch_perf[:,i,2],axis=1), color=colors[i], label=labels[i])
plt.xlabel('Epoch')
plt.ylabel('Time to Confirm')
plt.axvline(train_epochs, color='b')

scores = np.mean(all_lrs[:,0],axis=1) - np.mean(all_lrs[:,1],axis=1)
plt.subplot(524)
plt.plot(scores, color='tab:green')
slope, intercept, r_value, p_value, std_err = linregress(np.arange(n_epochs), scores)
plt.plot(slope*np.arange(n_epochs)+intercept, color='k')
plt.xlabel('Epoch')
plt.ylabel('Avg LR Diff') # should become more positive.
plt.title(f'm={slope:.3f}, R={r_value:.3f}, p={p_value:.3f}')

idxs = [int(train_epochs)-1, int(n_epochs)-1]
titles = ['With Heli', 'Without Heli']

gap = 100  # consider number of epochs to aggregate learning rate curves
for i,id in enumerate(idxs):
    plt.subplot(5,2,i+5)
    pess, lrss, area = plot_lrs(all_states[id-gap:id])
    
    for c in range(2):
        window_size = int(len(lrss[c])*0.2)
        smoothed_learning_rate = uniform_filter1d(lrss[c], size=window_size)
        plt.plot(pess[c], smoothed_learning_rate, color=colors[c], linewidth=2,label=labels[c])

    plt.title(f'CB={area[0]:.1f}, OB={area[1]:.1f}, A={(area[0]-area[1]):.1f}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Learning Rate')
    plt.title(titles[i])

j=7
for i,id in enumerate(idxs):
    for c in range(2):
        plot_behavior(all_states[id, c], contexts[c],np.mean(epoch_perf[id,c,3]), ax=plt.subplot(5,2,j))
        j+=1

plt.tight_layout()

df_area = np.round(area[0]-area[1])  # delta area between CP and OB learning rate curves

if len(store_params)==5:
    plt.savefig(f'./figs/{df_area}_{exptname}.png')
    print('Fig saved')
    # saveload('./state_data/'+exptname, [epoch_perf, all_states, all_pes, all_lrs, store_params], 'save')
    # print('Data saved')
    model_path = f'./model_params/{df_area}_{exptname}.pth'
    torch.save(model.state_dict(), model_path)
    print('Model saved')
