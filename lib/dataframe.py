import lightgbm as lgb 
import pandas as pd 
from env.config import Config 
import os 
from dataclasses import dataclass 
from termcolor import cprint 
import json 
C = Config 

class StockDataFrame(): 
    def __init__(self): 
        self.datagroup = None 
        
    def load_datagroup(self,datagroup_name): 
        self.datagroup_name = datagroup_name
        path = os.path.join(C.datapath,datagroup_name)
        
        # load datagroup
        if self.datagroup is not None: 
            print('WARNING: datagroup already exist ')
        self.datagroup = {}
        for type in ['train', 'valid', 'test']: 
            self.datagroup[type] = pd.read_feather(os.path.join(path, type+'.feather'))
        cprint(f'dataset is loaded from {path}','green')

        # load meta 
        with open(os.path.join(path, 'meta.json'), mode = 'r') as js: 
            self.meta = json.load(js)

    def __repr__(self): 
        print(f'Data group "{self.datagroup_name}" Spec.')
        print("___________________")
        for k, v in self.meta.items(): 
            if 'all' not in k: 
                cprint(k, 'green')
                cprint(v, 'blue')
        return '_________________________'
    
    
    def to_lgbm(self): 
        self.lgbm_datagroup = {} 
        if self.datagroup is None: 
            print('There is no loaded data group')
        else: 
            ycol = self.meta['ycol']
            xcol = self.meta['xcol']
            catcol = self.meta['catcol']
            for types, df in self.datagroup.items(): 
                self.lgbm_datagroup[types] = {} 
                for target in ycol: 
                    self.lgbm_datagroup[types][target] = lgb.Dataset(df[xcol], label = df[target], categorical_feature=catcol)

    @classmethod
    def list_datagroup(cls): 
        return [c for c in os.listdir(C.datapath) if 'raw' not in c] 
            