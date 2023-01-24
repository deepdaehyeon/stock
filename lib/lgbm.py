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
from dataclasses import dataclass, field       

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
    models: dict = field(default_factory=dict)

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
        self.params = params
        self.datagroup_name = sdf.datagroup_name
        self.isTrained = True

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
        
    
    def tune(self, sdf: StockDataFrame, _param_space, n_trial, random_seed = 17): 
        targets = [c for c in sdf.datagroup['train'].keys()]
        for target in targets: 
            study = optuna.create_study(direction='minimize',sampler=TPESampler(seed = random_seed))
            study.optimize(
                lambda trial : self.objective(target, trial, _param_space, sdf),
                n_trials= n_trial ,
            )
            cprint(f'{target}"s Best trial: score {study.best_trial.value}, \n params {study.best_trial.params}', 'green')    
            self.models[target] = study.user_attrs['best_booster']

    def objective(self, target, trial, _param_space, sdf): 
        train_set = sdf.datagroup['train'][target]
        valid_set = sdf.datagroup['valid'][target]
        callbacks = [lgb.callback.early_stopping(300), 
                     bestmodel_callback]
    
        model = lgb.train(params= _param_space(trial),
                        train_set= train_set, 
                        valid_sets = valid_set,
                        verbose_eval= 100, 
                        callbacks= callbacks, 
        )
        trial.set_user_attr(key='best_booster', value = model)
        y_pred = model.predict(valid_set[sdf.meta['xcol']])
        score = mean_absolute_error(y_pred, valid_set[sdf.meta['ycol']])
        return score


def bestmodel_callback(study, trial): 
    if study.best_trial.number == trial.number: 
        study.set_user_attr(key= 'bset_booster', value = trial.user_attrs['best_booster'])