import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class Heu(nn.Module):
    def __init__(self, input_size: int, output_size: int, layer_sizes: list[int], dropout: float=0.1):
        super(Heu, self).__init__()
        layers = []
        old_size = input_size
        for layer in layer_sizes:
            layers.append(nn.Linear(old_size, layer))
            layers.append(nn.BatchNorm1d(layer))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            old_size = layer

        layers.append(nn.Linear(old_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        src = self.x[idx]
        tar = self.y[idx]

        return src, tar
    

class GaussianCrossEntropyLoss(nn.Module):
    def __init__(self, num_bins, sigma=1.0):
        super(GaussianCrossEntropyLoss, self).__init__()
        self.num_bins = num_bins
        self.sigma = sigma

    def hl_gauss(self, values):
        bin_centers = torch.arange(self.num_bins, dtype=torch.float32, device=values.device)
        gauss_distrib = torch.exp(-0.5 * ((bin_centers[None, :] - values[:, None]) / self.sigma) ** 2)
        # gauss_distrib = gauss_distrib / gauss_distrib.sum(dim=-1, keepdim=True)
        return gauss_distrib

    def forward(self, predictions, labels):
        target_distribution = self.hl_gauss(labels)
        log_probs = F.log_softmax(predictions, dim=-1)
        loss = -torch.sum(target_distribution * log_probs, dim=-1)
        return loss.mean()