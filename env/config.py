import yaml 

class Config: 
    datapath = './data/'
    ckptpath = './bin/ckpt/'
    with open('./env/config.yaml') as f: 
        conf = yaml.load(f, Loader=yaml.FullLoader)
    X = conf['X'] 
    cat = ['day', 'month']