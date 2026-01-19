import torch
import torch.nn as nn

class SAINTLike(nn.Module):
    def __init__(self, num_features, dim=128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(1, dim) for _ in range(num_features)])
        self.cls = nn.Parameter(torch.randn(1,1,dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim*4, batch_first=True),
            num_layers=2
        )
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        toks = [self.proj[i](x[:,i:i+1]).unsqueeze(1) for i in range(x.shape[1])]
        toks = torch.cat(toks, dim=1)
        cls = self.cls.expand(x.size(0), -1, -1)
        out = self.transformer(torch.cat([cls, toks], dim=1))
        return self.head(out[:,0]).squeeze(1)
