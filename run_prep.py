from datetime import datetime
import argparse 
from utils.preprocess import Prep

# Args
parser = argparse.ArgumentParser() 
parser.add_argument('--scaling_column', type=str, default= 'open', choices=['open', 'close','52highlow'])
args = parser.parse_args()

if __name__ =="__main__": 
    Prep(args) 




     
    

 