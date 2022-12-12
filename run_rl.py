
'''
TBD 
Reinforcement learning 이전에 
supervised learning 을 먼저 적용하기로 결정
'''

if False:    

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



    # Stable_baseline 3 doc. 을 참고하여 진행!!
    - Tutorial: https://github.com/araffin/rl-tutorial-jnrr19/tree/sb3
    - Model compatibility: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
    - Examples: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html 

    * RL 학습의 본체는 utils/env.py 이다. 

    * train/test dataset 을 파일을 구분하여 저장해놔야함. 
    * discrete/ continuous dataset인지 확인하는게 중요 


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
        D = dt.datetime.strftime('%m%d_%H%M')

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
        
        # Train
        model.learn(total_timesteps=20000) 
        model.save(f'./ckpt/{D}')

        # Test: test dataset으로 env 변경이 필요함 
        obs = env.reset() # environment initializing 
        for i in range(2000):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render() # render current environment
            if done: 
                obs = env.reset() 
                
        
    if __name__ =='__main__': 
        main() 