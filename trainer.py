import gym
import json
import datetime as dt

from stable_baselines3 import a2c, ppo 
from stable_baselines3.common.vec_env import dummy_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

from utils.env import StockTradingEnv
import pandas as pd



def main(): 
    # Create envs 
    df = pd.read_csv('./data/AAPL.csv').sort_values('Date')
    env = dummy_vec_env([lambda: StockTradingEnv(df)])

    # Model define
    model = a2c(ActorCriticPolicy, env, verbose=1)
    model.learn(total_timesteps=20000)

    # Train loop 
    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done: 
            obs = env.reset() 
            
    
if __name__ =='__main__': 
    main() 