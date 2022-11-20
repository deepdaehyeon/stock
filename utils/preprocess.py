
# Libraries 
if True: 
    import pandas as pd 
    from datetime import datetime
    import yaml
    import os
    import argparse 

# Argparser & configs 
if True: 
    ## Parsing 
    parser = argparse.ArgumentParser() 
    # parser.add_argument('--year', type=int, default= 2022)
    args = parser.parse_args()
    # Train/test dataset 나누는 arg 필요 
    
    ## Configs 
    BASEPATH =os.getcwd()+'/.' 
    with open(BASEPATH+'/config/config.yaml') as conf:
        config = yaml.load(conf, Loader = yaml.FullLoader) 

# preprocess 
def main(): 
    ret = None 
    for f in os.listdir(BASEPATH +'/data/rawdata/'): 
        # Load data
        df = pd.read_csv(BASEPATH+f'/data/rawdata/{f}')
        df['Date']= df.Date.transform(lambda x: x.split(' ')[0])

        # Feature engineering 
        ''' TBD '''
        
        # rename column 
        D = {c:c+"_"+f.split('.')[0] for c in df.columns if 'Date' not in c}
        df.rename(D, axis=1, inplace= True)
        
        
        # Merge
        if ret is None: 
            ret = df 
        else: 
            ret = pd.merge(left = ret, right = df, on='Date', validate='one_to_one') 
        
    # Concat all rawdata
    ret.to_csv(BASEPATH+'/data/dataset.csv')


if __name__ =="__main__": 
    main() 




     
    

 