import torch 
from torch import nn



class MLPNET(nn.Module): 
    def __init__(self, emb_dim, n_cols, out_dim, n_layers, hidden_dim):
        self.embedding = None
        input_dim = emb_dim * n_cols 
        b = [input_dim if i ==0 else hidden_dim  for i in range(n_layers) ] 
        h = [hidden_dim, out_dim]
        self.body = MLP(dims = b, act = nn.ReLU()) 
        self.head = MLP(dims = h, act = nn.ReLU())
    
    def forward(self, x): 
        x = self.embedding(x) 
        x = self.body(x) 
        return self.head(x) 
    

class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

