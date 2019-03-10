import numpy as np

# global vars
n_exp        = 1
k            = 100
# model vars
drop         = 0.5
std          = 0.001
fm1          = 16
fm2          = 32
w_norm       = True
# optim vars
lr           = 0.001
beta2        = 0.99
num_epochs   = 3
batch_size   = 60
# temporal ensembling vars
alpha        = 0.6
data_norm    = 'channelwise'
divide_by_bs = False
# RNG
rng          = np.random.RandomState(42)
seeds        = [rng.randint(200) for _ in range(n_exp)]