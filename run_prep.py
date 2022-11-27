from datetime import datetime
import argparse 
from utils.preprocess import Prep

# Args
parser = argparse.ArgumentParser() 

parser.add_argument('--scaling_mode', type=str, default= 'plain', choices=['plain', 'minmax'])
parser.add_argument('--scaling_price', type=str, default= 'Open')
parser.add_argument('--scaling_amount', type=str, default= 'Lvolume', choices=['Lvolume', 'Mvolume','Svolume'])
parser.add_argument('--long_period', type=int, default= 100)
parser.add_argument('--mid_period', type=int, default= 45)
parser.add_argument('--short_period', type=int, default= 10)
parser.add_argument('--test_size', type=float, default= 0.1)
parser.add_argument('--valid_size', type=float, default= 0.2)
parser.add_argument('--random_seed', type=int, default= 42)
args = parser.parse_args()

if __name__ =="__main__": 
    preprocessor = Prep(args) 
    preprocessor._prep()




     
    

 