import torch 
from torch.utils.data import dataloader, Dataset
from torch import nn, optim 
import pandas as pd 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        train_ds = _load_dataset('train') 
        valid_ds = _load_dataset('valid') 
        
        # Define model
        model = _define_model(args) 
        _load_model()        
        
        # Criterion/ optimizer/ scheduler 
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.AdamW(model.parameters())
        scheduler = None 
        
        bestperf = 0 
        for epoch in range(args.epochs): 
            _train_step(model,train_ds, criterion, optimizer, scheduler) 
            perf = _valid_step(model,valid_ds, criterion, optimizer, scheduler) 
            if bestperf < perf: 
                bestperf = perf 
                _save_model(model)

    def _test(args): 
        pass 


# Functions 
def _define_model(args) -> nn.Module: 
    pass 

def _load_model(model): 
    pass 

def _save_model(model): 
    pass 

def _load_dataset(types = 'train') -> Dataset: 
    pass 


class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        return image, label


def _train_step(model,dataloader, criterion, optimizer, scheduler):
    model.train()
    for X, y in dataloader:
        optimizer.zero_grad() 
        X = X.to(device)
        y = y.to(device, dtype= torch.long)
        
        out = model(X) 
        loss = criterion(y, out)
        loss.backward() 
        optimizer.step() 
    scheduler.step() 

def _valid_step(model,dataloader):
    model.eval()
    outs = [] 
    ys = [] 
    with torch.no_grad(): 
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device, dtype= torch.long)
            out = model(X) 
            # Metrics 
 





