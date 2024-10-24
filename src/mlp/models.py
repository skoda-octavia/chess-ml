import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, layer_sizes: list[int], dropout: float=0.1):
        super(MLP, self).__init__()
        layers = []
        flat = nn.Flatten(start_dim=1)
        layers.append(flat)
        old_size = input_size
        for layer in layer_sizes:
            layers.append(nn.Linear(old_size, layer))
            layers.append(nn.BatchNorm1d(layer))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            old_size = layer

        layers.append(nn.Linear(old_size, output_size))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class MyDataset(Dataset):
    def __init__(self, x, y, legals):
        self.x = x
        self.y = y
        self.legals = legals

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        src = self.x[idx]
        tar = self.y[idx]
        legals = self.legals[idx]

        return src, tar, legals