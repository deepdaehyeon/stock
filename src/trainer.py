import torch 
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR 
import pandas as pd 
import os 
import yaml 

from lib.models import MLPNET

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config: 
    datapath = './data/'
    ckptpath = './bin/ckpt/'
    with open('./config/config.yaml') as f: 
        conf = yaml.load(f, Loader=yaml.FullLoader)
    X = conf['X']    

C = Config 

# Main functions 
if True: 
    def main(args):
        if args.mode == 'train': 
            _train(args) 
        
        elif args.mode =='test': 
            _test(args) 
        
        else: 
            raise NotImplementedError(f"There is no option like {args.mode} !")

    def _train(args):
        # prepare datasets
        train_ds = CustomDataset('train') 
        valid_ds = CustomDataset('valid')
        train_ds = DataLoader(train_ds, batch_size= args.batch_size, shuffle= True, ) 
        train_ds = DataLoader(valid_ds, batch_size= args.batch_size, shuffle= False , ) 
        
        # Define model
        model = _define_model(args).to(device) 
        _load_model(args, model)        
        
        # Criterion/ optimizer/ scheduler 
        criterion = nn.MSELoss() 
        optimizer = AdamW(model.parameters(), lr= args.lr)
        scheduler = MultiStepLR(optimizer=optimizer, milestones=[args.epochs//4, args.epochs//2, args.epochs//1.5], gamma=0.5) 
        
        # Train loop
        bestperf =  0 if args.order =='max' else 1e4 
        for epoch in range(args.epochs):
            print(f'EPOCH {epoch} th training loop start...!') 
            _train_step(model,train_ds, criterion, optimizer, scheduler) 
            perf = _valid_step(model,valid_ds, criterion) 
            if (bestperf < perf and args.order=='max') or (bestperf >perf and args.order=='min') : 
                bestperf = perf 
                _save_model(model)

    def _test(args): 
        # prepare datasets
        test_ds = CustomDataset('test') 
        test_ds = DataLoader(test_ds, batch_size= args.batch_size, shuffle= False , ) 
        
        # Define model
        model = _define_model(args).to(device) 
        _load_model(args, model)        
        
        # Criterion 
        criterion = nn.MSELoss() 
        
        # Train loop
        _valid_step(model, test_ds, criterion) 


# Functions 
class CustomDataset(Dataset):
    def __init__(self, types= 'train'): 
        self.df = pd.read_csv(C.datapath+f'{types}.csv')
        cols = self.df.columns() 
        self.all = [c for c in cols if  C.X in c]
        self.cy = [c for c in cols if 'next' in cols]
        self.cx = [c for c in self.all if c not in self.cy]
        
        C.cy = self.cy 
        C.cx = self.cx 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[self.cx], self.df[self.cy]


def _train_step(model,dataloader, criterion, optimizer, scheduler):
    epoch_loss =0 
    model.train()
    for X, y in dataloader:
        optimizer.zero_grad() 
        X = X.to(device)
        y = y.to(device)
        out = model(X) 
        loss = criterion(y, out)
        epoch_loss += loss.item()  
        loss.backward() 
        optimizer.step() 
    scheduler.step() 
    print(f'TRAIN LOSS :{epoch_loss:.3f}')

def _valid_step(model,dataloader, criterion):
    epoch_loss =0 
    model.eval()
    with torch.no_grad(): 
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            out = model(X) 
            loss = criterion(y, out)
            epoch_loss += loss.item() 
    print(f'VALID LOSS :{epoch_loss:.3f}')
            
            
def _define_model(args):
    if args.model == 'mlp': 
        model = MLPNET(emb_dim= args.emb_dim, n_cols = C.cx, out_dim = C.cy, n_layers = args.depth, hidden_dim = args.hidden_dim) 
    else: 
        raise NotImplementedError('no ')
    return model 

def _load_model(args, model):
    ckptpath = C.ckptpath + f'{args.ckpt}/bestmodel.pth'
    if os.path.isfile(ckptpath): 
        state_dict = torch.load()
        model.load(state_dict)
        print(f"Model loaded from {ckptpath}")
    else: 
        print("Training start from scratch")
    
    
def _save_model(args, model): 
    ckptpath = C.ckptpath + f'{args.ckpt}/'
    os.makedirs(ckptpath, exist_ok= True)
    ckptpath +='bestmodel.pth'
    torch.save(model.state_dict(),ckptpath )
    print(f"Model saved at {ckptpath}")
 





