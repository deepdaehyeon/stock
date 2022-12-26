import pandas as pd 
import os
from sklearn.model_selection import train_test_split 
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Prep:
    scaling_mode: str = 'plain'
    scaling_price: str = 'Open'
    scaling_amount: str = 'Lvolume'
    long_period: int = 100
    mid_period: int = 45
    short_period: int = 10
    test_size: float  = 0.1
    valid_size: float = 0.2
    random_seed: int  = 1117
    export_dir: str ='/data/rawdata/'
    
    def __post_init__(self):
        self.BASEPATH = os.getcwd().replace('\\', '/')
        
    def run(self): 
        ret = None 
        DATAPATH = self.BASEPATH + self.export_dir
        # for all raw data files
        for f in os.listdir(DATAPATH): 
            # Load data
            df = pd.read_csv(DATAPATH+f)
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

            ## Reward 계산 목적 다음 날 종가 가격 저장 (리밸런싱 시점: 종가 -> 종가에 사고팔고, 다음날은 종가까지 기다림)
            df['next_close_price'] = df['Close'].shift(periods=-1)
            
            ## Price scaling  
            cols = [c for c in df.columns if 'Date' not in c]  
            cols_price = [c for c in cols if 'amount' not in c]  
            if self.scaling_mode == 'plain':
                den = df[self.scaling_price]
                mean = 0 
            elif self.scaling_mode == 'minmax':
                mean = df[self.scaling_price[0]]
                den = df[self.scaling_price[1]] - mean
            else:
                assert NotImplementedError
                
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
            D = {c:c+"_"+f.split('.')[0] for c in df.columns if 'Date' not in c}
            df.rename(D, axis=1, inplace=True)
            
            # Merge
            if ret is None: 
                ret = df 
            else: 
                ret = pd.merge(left=ret, right=df, on='Date', validate='one_to_one')
        
        # Append day and month 
        ret['day'] = ret['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').weekday()) 
        ret['month'] = ret['Date'].apply(lambda x: x.split('-')[1]) 
        
        # Drop NaN 
        ret.dropna(axis = 0, inplace = True )
                    
        # Discrete version dataset split 
        ## random seed를 정해줘야 재현성이 보장됨 
        df_train, df_test= train_test_split(ret, test_size = self.test_size, shuffle=False, random_state = self.random_seed) 
        df_train, df_valid= train_test_split(df_train, test_size = self.valid_size,shuffle=True, random_state = self.random_seed)
        
        # Save result csv
        df_train.to_csv(DATAPATH+f'../train.csv')
        df_valid.to_csv(DATAPATH+f'../valid.csv')
        df_test.to_csv(DATAPATH+f'../test.csv')
