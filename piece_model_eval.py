import chess
import torch
import torch.nn as nn

def print_board(tensor):
    # for i in range(tensor.size())
    pass

piece = ""

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(8*8*7, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 8*8),
)



file_path = 'model.pth'
model.load_state_dict(torch.load(file_path))
X = torch.load(f"X_{piece}.pt")
y = torch.load(f"Y_{piece}.pt")

for x_sample, y_sample in zip(X, y):
    with torch.no_grad():
        val_inputs = x_sample.view(-1, 8*8*7)
        val_labels = y_sample.view(-1, 8*8)
        val_outputs = model(val_inputs)
        max_idxs, _ = torch.max(val_outputs, 1)