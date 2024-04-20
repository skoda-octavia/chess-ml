import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model_mlp import Classifier
from torch.utils.data import TensorDataset, DataLoader
from features.get_acc import get_accuracy_class
from sklearn.model_selection import train_test_split


def train(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, eps: int, leraning_rate, device: torch.device):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=leraning_rate)
    epss = []
    train_accs = []
    val_accs = []

    for epoch in range(eps):

        for i, batch in enumerate(train_dl, 0):
            model.train()
            inputs, labels = batch
            inputs, labels=  inputs.to(device), labels.to(device)
            out = model(inputs)           # forward pass
            loss = criterion(out, labels)
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()

        if (epoch+1) % 10 == 0:
            model.eval()
            train_acc = get_accuracy_class(model, train_dl, device)
            val_acc = get_accuracy_class(model, val_dl, device)

            print(f'Epoch [{epoch+1}/{eps}], train_acc: {train_acc:4f}, test_acc: {val_acc:4f}')
            epss.append(epoch)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

    return epss, train_accs, val_accs


if __name__ == "__main__":
    eps = 100000
    batch_size = 4096
    lr = 0.01

    data_path = "data/processes/"
    x_file = data_path + "X_all.pt"
    y_file = data_path + "y_all.pt"
    X = torch.load(x_file)
    y = torch.load(y_file)
    dataset = TensorDataset(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    train_loader = DataLoader(X, batch_size=batch_size, shuffle=True) 
    valid_loader = DataLoader(y, batch_size=batch_size, shuffle=False) 
    model = Classifier(6*8*8, 8*8, [256, 256, 128, 128, 96, 96, 64], 0.05)
    epss, train_acss, val_acss = train(model, train_loader, valid_loader, eps, lr)