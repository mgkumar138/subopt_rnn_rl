# RNN-RL Agent for Predictive Inference Task

This repository implements a recurrent neural network (RNN) based reinforcement learning (RL) agent to model decision-making deficits in psychiatric disorders, using schizophrenia as a case study. The agent is trained on a predictive inference task from Nassar et al. (2021) that assesses cognitive deficits through two conditions: change-point (volatile environment) and oddball (resistance to outliers).

## Key Features

- Implements an RNN-RL agent using Advantage Actor-Critic (A2C) algorithm
- Trains agents on the predictive inference task with both change-point and oddball conditions
- Analyzes agent behavior through learning curves and prediction error metrics
- Identifies and characterizes fixed points in RNN dynamics
- Replicates key findings from the original study:
  - Suboptimal belief updating patterns (Figures 1 & 2)
  - Fixed point analysis of network dynamics (Figures 3 & 4)
  - Hyperparameter effects on behavior and neural dynamics

## Installation

1. Clone this repository:
2. Install required packages:

## Usage

### 1. Training the Agent

Train an RNN-RL agent on the predictive inference task:
```python
python train_agent.py \
    --gamma 0.95 \          # reward discount factor
    --beta_delta 1.0 \      # advantage scaling factor
    --p_reset 0.0 \         # probability of RNN dynamics reset
    --tau 20 \              # rollout buffer size
    --epochs 50000 \        # training epochs
    --hidden_size 64       # RNN hidden units
```

### 2. Sampling Behavior

Sample behavior from a trained agent:
```python
python sample_behavior.py \
    --model_path saved_models/agent_gamma0.95.pt \
    --num_epochs 10 \
    --trials_per_epoch 200 \
    --output_dir results/behavior
```

### 3. Analyzing Learning Curves

Generate learning curves and behavioral metrics:
```python
python analyze_behavior.py \
    --data_dir results/behavior \
    --output_dir figures \
    --plot_type learning_curves  # or prediction_error
```

### 4. Fixed Point Analysis

Find and analyze fixed points in RNN dynamics:
```python
python fixed_point_analysis.py \
    --model_path saved_models/agent_gamma0.95.pt \
    --num_samples 1000 \
    --output_dir results/fixed_points
```


## Citation

If you use this code in your research, please cite the original paper:

```
Kumar, M. G.*, Manoogian, A.*, Qian, B., Pehlevan, C., & Rhoads, S. A. (2025). 
Neurocomputational underpinnings of suboptimal beliefs in recurrent neural network-based agents. 
bioRxiv, 2025.03.13.642273. https://doi.org/10.1101/2025.03.13.642273
```
