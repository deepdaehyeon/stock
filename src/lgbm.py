from dataclasses import dataclass, field 
import yaml 
import lightgbm as lgb 
import pandas as pd 
import numpy as np 
import os 

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

class Config: 
    datapath = './data/'
    ckptpath = './bin/ckpt/'
    with open('./config/config.yaml') as f: 
        conf = yaml.load(f)
    X = conf['X'] 
    cat = ['day', 'month']
C = Config 


# Modules 
if True: 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--random_seed', type=int, default= 17) 
    parser.add_argument('--n_trials', type=int, default= 100) 
    parser.add_argument('--n_batch', type=int, default= 100) 
    parser.add_argument('--metric', type=str, default= 'mae') 
    args = parser.parse_args()

def main(args): 
    study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=args.random_seed))
    study.optimize(lambda trial : objective(trial, args), n_trials=args.n_trials)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))


class lgbm_dataset(lgb.Dataset): 
    def __init__(self, types = 'train'):
        df = pd.read_csv(os.path.join(C.datapath, types +'.csv'))
        cols = df.columns()
        call = C.X
        cx = [c for c in call if c not in cy]
        cy = [c for c in call if 'next' in c]
        ccat = C.cat
        
        self.df = df 
        self.X = df[cx]        
        self.y = df[cy]
        super().__init__(self.X, self.y, categorical_feature=ccat) 
    
    @property
    def get_y(self): 
        y = self.y
        return self.df.y

    
def objective(trial: Trial, args): 
    params = {
        'boosting_type' : 'gbdt',
        "n_estimators" : 10000,
        'max_depth':trial.suggest_int('max_depth', 4, 16),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 8, 32),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.8, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 16, 64),
        'learning_rate': 1, 
        'random_state': args.random_seed,
        'metric': args.metric, 
    }
    score = lgbm_params_fold_start(params, args)
    return score


def lgbm_params_fold_start(params, args):
    train_set = lgbm_dataset('train')
    valid_set = lgbm_dataset('valid')

    model = lgb.train(params= params,
                      num_boost_round= args.n_batch, 
                      train_set= train_set, 
                      valid_sets = valid_set,
                      categorical_feature=C.cat, 
                      early_stopping_rounds= args.early_stopping_round,  )
    y_pred = model.predict(valid_set)
    score = mean_absolute_error(y_pred, valid_set.y)
    return score 