import lightgbm as lgb 
import pandas as pd 
from env.config import Config 
import os 
from dataclasses import dataclass 
from termcolor import cprint 
C = Config 

class StockDataFrame(pd.DataFrame): 
    @classmethod
    def dataset_list(cls): 
        return [c for c in os.listdir(C.datapath) if 'raw' not in c] 
    
    @property 
    def _dataset_name(self): 
        return self.dataset_name
    
    @_dataset_name.setter
    def _dataset_name(self, text): 
        self.dataset_name = text 
        
    def load(self,types ='train'): 
        path = os.path.join(C.datapath, self.dataset_name, types+'.parquet')
        self.from_parquet(path)
        cprint(path,'green')
        
    def to_lgbmdataset(self): 
        pass 
    

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