import pandas as pd 
import investpy as iv
import investiny as it 
from datetime import datetime
import yaml
import os
import argparse 


if True: 
    ## Parsing 
    parser = argparse.ArgumentParser() 
    # parser.add_argument('--year', type=int, default= 2022)
    args = parser.parse_args()
    
    ## Set path
    BASEPATH =os.getcwd()+'/..' 
    with open(BASEPATH+'/config/config.yaml') as conf:
        config = yaml.load(conf, Loader = yaml.FullLoader) 

# MA

# 52주 최저점 

# 52주 최고점 



     
    

 