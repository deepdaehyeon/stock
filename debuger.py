# Modules 
from src.lgbm import LGBM

random_seed  = 17
n_trials  = 100 
n_batch  = 100 
metric  = 'mae'

gbm = LGBM(
    n_trials  = n_trials,  
    n_batch  = n_batch, 
    metric  = metric, 
    random_seed  = random_seed, 
) 

gbm.tune()
