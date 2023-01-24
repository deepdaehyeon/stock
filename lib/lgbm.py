import lightgbm as lgb
from env.config import Config 

from typing import List
import pandas as pd 
import os 
import warnings
from termcolor import cprint
import pickle as pkl
import json

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass       

from lib.dataframe import StockDataFrame 
from datetime import datetime 

warnings.filterwarnings('ignore')

C = Config 

@dataclass
class LgbmTrainInstance: 
    instance_name: str = ""
    datagroup_name: str = ""
    isTuned: bool = False 
    isTrained: bool = False 
    params = None

    @classmethod
    def list_instance(cls): 
        return [c for c in os.listdir(C.instancepath) ] 

    def save_instance(self): 
        path = os.path.join(C.instancepath, self.instance_name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'instance.pickle'), mode = 'wb' ) as f: 
            pkl.dump(self, f )
        print(self)
    
    def load_instance(self, instance_name): 
        path = os.path.join(C.instancepath, instance_name)
        with open(os.path.join(path, 'instance.pickle'), mode = 'rb' ) as f: 
            self = pkl.load(f)
        print(self)
    
    def __repr__(self): 
        print(f'Instance {self.instance_name} Spec.')
        print("_____________________")
        cprint(f'is trained?: {self.isTrained}', 'green')
        cprint(f'is tuned?: {self.isTuned}', 'green')
        cprint(f'datagroup name: {self.datagroup_name}', 'green')
        cprint(f'params: {self.params}', 'blue')
        return "______________________"

    def train(self, sdf: StockDataFrame , **params): 
        self.models = {}
        self.params = params
        self.datagroup_name = sdf.datagroup_name
        self.isTrained = True
        
        self.meta = {
            'param' : params, 
            'datagroup_name' : self.datagroup_name, 
            'instance_name' : self.instance_name
        }

        dg = sdf.lgbm_datagroup
        targets = [c for c in dg['train'].keys()]
        callbacks = [lgb.callback.early_stopping(300)]
        for target in targets: 
            model = lgb.train(
                params= params, 
                train_set= dg['train'][target], 
                valid_sets=[dg['valid'][target]],  
                num_boost_round= 10000, 
                verbose_eval= 100, 
                callbacks= callbacks,
                categorical_feature= sdf.meta['catcol']
                )
            self.models[target] = model 
        
    
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
                      callbacks= callbacks, 
    )
    y_pred = model.predict(valid_set._X)
    score = mean_absolute_error(y_pred, valid_set._y)
    return score

