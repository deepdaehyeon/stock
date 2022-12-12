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
    parser.add_argument('--mode', type=str, default= 'train', 
                        choices=['train', 'test'] )
    parser.add_argument('--model', type=str, default= 'mlp', 
                        choices=['mlp', 'saint'] )
    
    parser.add_argument('--order', type=str, default= 'min', choices=['max', 'min']) 
    parser.add_argument('--ckpt', type=str, default= None) 
    parser.add_argument('--epochs', type=int, default= 20) 
    parser.add_argument('--lr', type=float, default= 1e-3)
    
    args = parser.parse_args()
    args.D = dt.datetime.strftime('%m%d_%H%M')
    
    # Configs 
    BASEPATH =os.getcwd() +'/..' 
    with open(BASEPATH+'/config/config.yaml') as conf:
        config = yaml.load(conf, Loader = yaml.FullLoader) 


if __name__ =='__main__': 
    main(args) 