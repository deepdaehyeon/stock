import gym
import json
import datetime as dt

from stable_baselines3 import a2c, ppo 
from stable_baselines3.common.vec_env import dummy_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

from utils.env import StockTradingEnv
import pandas as pd
import argparse 
import os 
import yaml 


'''
Rl model 학습과정은 
https://github.com/araffin/rl-tutorial-jnrr19/tree/sb3
https://stable-baselines3.readthedocs.io/en/master/guide/examples.html 참고 

확실하진 않지만 학습환경의 데이터셋이 연속적이여야 하는 것 같다. 그러므로 train/test dataset 을 파일을 구분하여 저장해놔야함. 
 


'''

# Config & Args 
if True: 
    # Argparser 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model', type=str, default= 'ppo', 
                        choices=['ppo', 'a2c'] )
    args = parser.parse_args()
    
    # Configs 
    BASEPATH =os.getcwd() +'/..' 
    with open(BASEPATH+'/config/config.yaml') as conf:
        config = yaml.load(conf, Loader = yaml.FullLoader) 

def main(): 
    # Create envs 
    df = pd.read_csv('./data/dataset.csv').sort_values('Date') # load dataset 
    df = df[config['X']] # feature selection 
    env = dummy_vec_env([lambda: StockTradingEnv(df)])

    # Model define
    if args.model =='ppo': 
        model = ppo(ActorCriticPolicy, env, verbose=1)
    elif args.model =='a2c': 
        model = a2c(ActorCriticPolicy, env, verbose=1)

    # Train loop 
    model.learn(total_timesteps=20000) # 
    obs = env.reset() # environment initializing 
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render() # render current environment
        if done: 
            obs = env.reset() 
            
    
if __name__ =='__main__': 
    main() 