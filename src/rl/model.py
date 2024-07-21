import torch.nn as nn
import torch


class rl(nn.Module):
    def __init__(self, input_size: int, output_size: int, layer_sizes: list[int], dropout: float=0.1):
        super(rl, self).__init__()
        layers = []
        flat = nn.Flatten(start_dim=1)
        layers.append(flat)
        old_size = input_size
        for layer in layer_sizes:
            layers.append(nn.Linear(old_size, layer))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            old_size = layer

        layers.append(nn.Linear(old_size, output_size))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def fit(self, tensor, score, optimizer, lock):
        # print(f"updating model: {id(self)}, pid: {os.getpid()}")
        out = self.forward(tensor)
        out = torch.squeeze(out)
        loss = self.criterion(out, score)
        with lock:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        del out
        del score


    def predict(self, tensor):
        return self.forward(tensor)