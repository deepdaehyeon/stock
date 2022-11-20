import pandas as pd 
import os

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
            ''' 
            high, low, close, open, amount, 52주 최고가/최저가, 20일 이평선
            scaling column 지정
            요일
            월
            '''
            
            # rename column 
            D = {c:c+"_"+f.split('.')[0] for c in df.columns if 'Date' not in c}
            df.rename(D, axis=1, inplace= True)
            
            # Merge
            if ret is None: 
                ret = df 
            else: 
                ret = pd.merge(left = ret, right = df, on='Date', validate='one_to_one') 
            
        # Discrete version dataset split 
        
        
        # Save result csv
        ret.to_csv(DATAPATH+f'/data/train.csv')




     
    

 