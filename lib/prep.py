import pandas as pd 
from termcolor import cprint  
import os
from sklearn.model_selection import train_test_split 
from datetime import datetime
from dataclasses import dataclass, field 
from env.config import Config 
import json 


C = Config 

@dataclass
class Prep:
    dataset_name: datetime = field( default_factory= lambda: datetime.now().strftime( '%Y-%m-%d%' ))
    target_period: int = 5 
    scaling_price: str = 'Close'
    scaling_amount: str = 'Lvolume'
    long_period: int = 100
    mid_period: int = 45
    short_period: int = 10
    test_size: float  = 0.1
    valid_size: float = 0.2
    random_seed: int  = 1117
        
    def run(self): 
        ret = None 
        # for all raw data files
        raw_path= os.path.join(C.datapath, 'rawdata')
        for f in os.listdir(raw_path): 
            # Load data
            df = pd.read_csv(os.path.join(raw_path, f))
            df['Date'] = df.Date.transform(lambda x: x.split(' ')[0])
            
            # Feature engineering
            L = self.long_period
            M = self.mid_period
            S = self.short_period
            df['Lhigh'] = df['High'].rolling(L).max()
            df['Llow'] = df['Low'].rolling(L).min()
            df['Lmean'] = df['Open'].rolling(L).mean()
            
            df['Mhigh'] = df['High'].rolling(M).max()
            df['Mlow'] = df['Low'].rolling(M).min()
            df['Mmean'] = df['Open'].rolling(M).mean()
            
            df['Shigh'] = df['High'].rolling(S).max()
            df['Slow'] = df['Low'].rolling(S).min()
            df['Smean'] = df['Open'].rolling(S).mean()
            
            df['Lvolume'] = df['Volume'].rolling(L).mean()
            df['Mvolume'] = df['Volume'].rolling(M).mean()
            df['Svolume'] = df['Volume'].rolling(S).mean()

            ## 향후 5일간의 평균가를 예측하는 문제로 세팅
            df['y'] = df['Close'].rolling(self.target_period).mean().shift(periods=-self.target_period) 
            
            ## Price scaling  
            cols = [c for c in df.columns if 'Date' not in c]  
            cols_price = [c for c in cols if 'amount' not in c]  
            den = df[self.scaling_price]
            mean = 0 
            for col in cols_price: 
                df[col] -= mean 
                df[col] /= den

            ## amount scaling  
            cols_amount = [c for c in cols if c not in cols_price]  
            for col in cols_amount: 
                df[col] = df.transform(lambda x: x[col]/x[self.scaling_amount] )
            
            ## Drop scaling columns 
            df.drop(self.scaling_price, axis=1, inplace= True)
            df.drop(self.scaling_amount, axis=1, inplace= True)

            # rename column 
            D = {c:"num|" + c+"_"+f.split('.')[0] for c in df.columns if 'Date' not in c and 'label' not in c}
            D.update({
                'y' : 'label|'+f.split('.')[0]
                })
            df.rename(D, axis=1, inplace=True)
            
            # Merge
            ret = df if ret is None else pd.merge(left=ret, right=df, on='Date', validate='one_to_one')
        
        # Append day and month 
        ret['cat|day'] = ret['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').weekday()).astype(int)
        ret['cat|month'] = ret['Date'].apply(lambda x: x.split('-')[1]).astype(int)
        
        # Drop NaN 
        ret.dropna(axis = 0, inplace = True )

        # Discrete version dataset split 
        ## random seed를 정해줘야 재현성이 보장됨 
        df_train, df_test= train_test_split(ret, test_size = self.test_size, shuffle=False, random_state = self.random_seed) 
        df_train, df_valid= train_test_split(df_train, test_size = self.valid_size,shuffle=True, random_state = self.random_seed)

        df_train.reset_index(drop = True, inplace= True)
        df_valid.reset_index(drop = True, inplace= True)
        df_test.reset_index(drop = True, inplace= True)

        # Meta data 
        meta = {
            'allcol': [c for c in df_train.columns ],  
            'ycol': [c for c in df_train.columns if 'label|' in c],  
            'xcol': [c for c in df_train.columns if 'label|' not in c and 'Date' not in c],  
            'numcol': [c for c in df_train.columns if 'num|' in c],  
            'catcol': [c for c in df_train.columns if 'cat|' in c],  
            'target_period' : self.target_period,
            'scaling_price ' : self.scaling_price ,
            'scaling_amount' : self.scaling_amount,
            'long_period' : self.long_period,
            'mid_period' : self.mid_period ,
            'short_period' : self.short_period ,
            'test_size' : self.test_size ,
            'valid_size' : self.valid_size ,
            'random_seed' : self.random_seed,
        } 
        # Save 
        save_path = os.path.join(C.datapath, self.dataset_name)      
        os.makedirs(save_path, exist_ok=True) 
        with open(os.path.join(save_path, 'meta.json'), mode= 'w') as js: 
            json.dump(meta, js)
            
        df_train.to_feather(os.path.join(save_path, 'train.feather')) 
        df_valid.to_feather(os.path.join(save_path, 'valid.feather' )) 
        df_test.to_feather(os.path.join( save_path, 'test.feather')) 
        
        cprint(f'datasets saved at {os.path.join(C.datapath, self.dataset_name)}', 'green')
        cprint(f'# of train sample : {len(df_train)}', 'green')
        cprint(f'# of valid sample : {len(df_valid)}', 'green')
        cprint(f'# of test sample : {len(df_test)}', 'green')
    
