'''
Supervised learning trainer

'''

# Modules 
if True: 
    import datetime as dt
    import pandas as pd
    import argparse 
    import os 
    import yaml 

    from src.trainer import main


# Env setting  
if True: 
    # Argparser 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model', type=str, default= 'ppo', 
                        choices=['ppo', 'a2c'] )
    args = parser.parse_args()
    args.D = dt.datetime.strftime('%m%d_%H%M')
    
    # Configs 
    BASEPATH =os.getcwd() +'/..' 
    with open(BASEPATH+'/config/config.yaml') as conf:
        config = yaml.load(conf, Loader = yaml.FullLoader) 



if __name__ =='__main__': 
    main(args) 