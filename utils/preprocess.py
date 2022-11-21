import pandas as pd 
import os
from sklearn.model_selection import train_test_split 
from datetime import datetime


class Prep:
    def __init__(self, args, export_dir='data/rawdata'):
        self.BASEPATH = os.getcwd().replace('\\', '/')
        self.export_dir = export_dir
        self.args = args 
        
    def prep(self): 
        DATAPATH =self.BASEPATH + self.export_dir
        
        # for all raw data files
        for f in os.listdir(DATAPATH): 
            
            # Load data
            df = pd.read_csv(DATAPATH+f)
            df['Date']= df.Date.transform(lambda x: x.split(' ')[0])
            
            # Feature engineering
            L = self.args.long_period
            M = self.args.mid_period
            S = self.args.short_period
            df['Lhigh'] = df['high'].rolling(L).max()
            df['Llow'] = df['low'].rolling(L).min()
            df['Lmean'] = df['open'].rolling(L).mean()
            
            df['Mhigh'] = df['high'].rolling(M).max()
            df['Mlow'] = df['low'].rolling(M).min()
            df['Mmean'] = df['open'].rolling(M).mean()
            
            df['Shigh'] = df['high'].rolling(S).max()
            df['Smean'] = df['open'].rolling(S).mean()
            df['Slow'] = df['low'].rolling(S).min()
            
            df['Lamount'] = df['amount'].rolling(L).mean()
            df['Mamount'] = df['amount'].rolling(M).mean()
            df['Samount'] = df['amount'].rolling(S).mean()
            
            ## Price scaling  
            cols = [c for c in df.columns if 'Date' not in c]  
            cols_price = [c for c in cols if 'amount' not in c]  
            for col in cols_price: 
                if self.args.scaling_mode =='plain': 
                    df[col] = df.transform(lambda x: x[col]/x[self.args.scaling_price] )
                else: 
                    df[col] = df.transform(lambda x: (x[col]- x[self.args.scaling_price[0]])/x[self.args.scaling_price[1]] )
                    
            ## amount scaling  
            cols_amount = [c for c in cols if c not in cols_price]  
            for col in cols_amount: 
                df[col] = df.transform(lambda x: x[col]/x[self.args.scaling_amount] )
            
            ## Drop scaling columns 
            for col in list(self.args.scailing_price): 
                df.drop(col, axis=1, inplace= True)
            for col in list(self.args.scailing_amount): 
                df.drop(col, axis=1, inplace= True)

            # rename column 
            D = {c:c+"_"+f.split('.')[0] for c in df.columns if 'Date' not in c}
            df.rename(D, axis=1, inplace= True)
            
            # Merge
            if ret is None: 
                ret = df 
            else: 
                ret = pd.merge(left = ret, right = df, on='Date', validate='one_to_one') 
        
        # Append day and month 
        ret['day'] = ret['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').weekday()) 
        ret['month'] = ret['Date'].split('-')[1] 
                    
        # Discrete version dataset split 
        ## random seed를 정해줘야 재현성이 보장됨 
        df_train, df_test= train_test_split(ret, test_size = self.args.test_size, shuffle=False, random_state = self.args.random_seed) 
        df_train, df_valid= train_test_split(df_train, test_size = self.args.valid_size,shuffle=True, random_state = self.args.random_seed)
        
        # Save result csv
        df_train.to_csv(DATAPATH+f'/data/train.csv')
        df_valid.to_csv(DATAPATH+f'/data/valid.csv')
        df_test.to_csv(DATAPATH+f'/data/test.csv')




     
    

 