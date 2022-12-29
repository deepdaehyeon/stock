import lightgbm as lgb
from env.config import Config 

import pandas as pd 
import os 
import warnings

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass 

warnings.filterwarnings('ignore')

C = Config 
@dataclass
class LGBM: 
    random_seed: int = 17
    n_trials: int = 100 
    n_batch: int = 100 
    metric: str = 'mae'
    early_stopping_round: int = 1000 
    
    def tune(self, target_col): 
        study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=self.random_seed))
        study.optimize(lambda trial : objective(trial,self, target_col), 
                       n_trials=self.n_trials)
        print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))
        
    #TODO: best model을 save/load/predict 하는 기능 

    
def objective(trial: Trial, self, target_col): 
    params = {
        'boosting_type' : 'gbdt',
        "n_estimators" : 10000,
        'max_depth':trial.suggest_int('max_depth', 4, 16),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 8, 32),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 16, 64),
        'random_state': self.random_seed,
        'metric': self.metric, 
        'verbose' : -1, 
    }
    train_set = LgbmDataset('train', target_col)
    valid_set = LgbmDataset('valid', target_col)
    callbacks = [lgb.callback.early_stopping(self.early_stopping_round)]

    model = lgb.train(params= params,
                      num_boost_round= self.n_batch, 
                      train_set= train_set, 
                      valid_sets = valid_set,
                      verbose_eval= False, 
                      categorical_feature=C.cat, 
                      callbacks= callbacks, 
    )
    y_pred = model.predict(valid_set._X)
    score = mean_absolute_error(y_pred, valid_set._y)
    return score


# lgbm dataset wrapper
class LgbmDataset(lgb.Dataset): 
    def __init__(self, types = 'train', target_y = 'QQQ'):
        df = pd.read_csv(os.path.join(C.datapath, types +'.csv'))
        cols = [c for c in df.columns if 'Unnamed' not in c and 'Date' not in c]
        cy = [c for c in cols if 'next' in c and target_y in c]
        cxcat = C.cat
        cxnum = [c for c in cols if 'next' not in c and c not in cxcat]
        
        # cast type 
        df[cxnum] = df[cxnum].astype(float)
        df[cxcat] = df[cxcat].astype(int)
        df[cy] = df[cy].astype(float)
        
        self.df = df 
        self.X = df[cxnum+cxcat]        
        self.y = df[cy]
        super(LgbmDataset, self).__init__(self.X, self.y, categorical_feature=cxcat) 
    
    @property
    def _y(self): 
        return self.y
    
    @property
    def _X(self): 
        return self.X