import lightgbm as lgb
from env.config import Config 

import pandas as pd 
import os 
import warnings
from termcolor import cprint

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass 


warnings.filterwarnings('ignore')

C = Config 

@dataclass
class LgbmTrainInstance: 
    instance_name: str = ""
    datagroup_name: str = ""
    isTuned: bool = False 
    isTrained: bool = False 
    
    @classmethod
    def list_instance(cls): 
        return [c for c in os.listdir(C.instancepath) ] 

    def save_instance(self): 
        pass 
    
    def load_instance(self, instance_name): 
        self.instance_name = instance_name
        path = os.path.join(C.instancepath, instance_name)
    
    def __repr__(self): 
        print(f'Instance {self.instance_name} Spec.')
        print("_____________________")
        cprint(f'datagroup name: {self.datagroup_name}', 'green')
        cprint(f'is trained?: {self.isTrained}', 'green')
        cprint(f'is tuned?: {self.isTuned}', 'green')
        return "______________________"

    def train(self,datagroup, **params): 
        target = [c for c in datagroup['train'].keys()]
        
        model = lgb.train(params = params, 
                          
                          )
    
    def tune(self,datagroup, **param_space): 
        study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=self.random_seed))
        study.optimize(lambda trial : objective(trial, self, param_space, datagroup), n_trials=self.n_trials)
        print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))
    
def objective(trial: Trial, self, param_space, datagroup): 
    train_set = datagroup['train']
    valid_set = datagroup['valid']
    callbacks = [lgb.callback.early_stopping(self.early_stopping_round)]

    model = lgb.train(params= param_space,
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

