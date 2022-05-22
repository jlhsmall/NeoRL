import torch
from offlinerl.utils.exp import select_free_cuda

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42

device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

vae_features = 256#750
vae_layers = 2
jitter_features = 256#400
jitter_layers = 2
value_features = 256#400
value_layers = 2
phi = 0.05
lam = 0.75

batch_size = 100
steps_per_epoch = 1000#0
max_epoch = 100#200

vae_lr = 1e-3
jitter_lr = 3e-4
critic_lr = 3e-4
gamma = 0.99
soft_target_tau = 5e-3

#tune
params_tune = {
    "phi" : {"type" : "discrete", "value": [0.1]},
    "lam" : {"type" : "continuous", "value": [0, 0.25, 0.5, 0.75, 1]},
}

#tune
grid_tune = {
    "lam" : [0, 0.25, 0.5, 0.75, 1]
}
