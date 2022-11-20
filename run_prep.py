from datetime import datetime
import argparse 
from utils.preprocess import Prep

# Args
parser = argparse.ArgumentParser() 
parser.add_argument('--scaling_column', type=str, default= 'open', choices=['open', 'close','52highlow'])
parser.add_argument('--test_size', type=float, default= 0.1)
parser.add_argument('--valid_size', type=float, default= 0.3)
parser.add_argument('--random_seed', type=int, default= 42)
args = parser.parse_args()

if __name__ =="__main__": 
    Prep(args) 




     
    

 