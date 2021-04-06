import torch.nn as nn
from torch.nn import functional


class Net2(nn.Module):
    def __init__(self, n_class=3, **kwargs):
        n = kwargs["n_points"]
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(n*2, n*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(n*8, n*2),
            nn.ReLU(inplace=True),
            nn.Linear(n*2, n_class),
        )

    def forward(self, x):
        x = self.clf(x)
        return x
