from datetime import datetime
import argparse 
from utils.preprocess import Prep

# Args
parser = argparse.ArgumentParser() 

parser.add_argument('--scaling_mode', type=str, default= 'plain', choices=['plain', 'minmax'])
parser.add_argument('--scaling_price', type=str, default= 'open')
parser.add_argument('--scaling_amount', type=str, default= 'Lamount', choices=['Lamount', 'Mamount','Samount'])
parser.add_argument('--long_period', type=int, default= 130)
parser.add_argument('--mid_period', type=int, default= 45)
parser.add_argument('--short_period', type=int, default= 10)
parser.add_argument('--test_size', type=float, default= 0.1)
parser.add_argument('--valid_size', type=float, default= 0.3)
parser.add_argument('--random_seed', type=int, default= 42)
args = parser.parse_args()

if __name__ =="__main__": 
    Prep(args) 




     
    

 