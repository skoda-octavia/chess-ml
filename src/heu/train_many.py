import torch
import torch.nn as nn
import torch.optim as optim
from models import Heu, MyDataset, GaussianCrossEntropyLoss 
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F

def print_(str):
    print(str)
    with open("heu_train.txt", "a") as f:
        f.write(str+"\n")


def val_ep(
        model: nn.Module,
        val_dl: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        ep = 0,):

    val_losses = []

    model.eval()
    for i, batch in enumerate(val_dl):
        inputs, labels = batch
        inputs, labels=  inputs.to(device), labels.to(device)
        out = model(inputs)
        out = F.softmax(out, -1)
        loss = criterion(out, labels)
        val_losses.append(loss.item())

    torch.save(model.state_dict(), weights_path + f"stock/model_weights_{ep}.pth")
    torch.save(optimizer.state_dict(), weights_path + f"stock/opti_.pth")


    return sum(val_losses) / len(val_losses)


def train_ep(
        model: nn.Module,
        train_dl: DataLoader,
        criterion: nn.Module,
        optimizer: nn.Module,
        device: torch.device,):
    
    train_losses = []

    model.train()
    for i, batch in enumerate(train_dl, 0):
        
        optimizer.zero_grad()
        inputs, labels = batch
        inputs, labels=  inputs.to(device), labels.to(device)
        out = model(inputs)
        out = F.softmax(out, -1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # if epoch % 10 == 0:

    return sum(train_losses) / len(train_losses)


if __name__ == "__main__":

    load_num = 0
    eps = 1200
    batch_size = 1024
    leraning_rate = 0.002
    dropout = 0
    bins = 128

    data_path = "data/prep/"
    files = [
        ("X_turn_stock_0102.pt", "y_turn_stock_0102.pt"),
        ("X_turn_stock_0304.pt", "y_turn_stock_0304.pt"),
        ("X_turn_stock_0506.pt", "y_turn_stock_0506.pt"),
        ("X_turn_stock_0708.pt", "y_turn_stock_0708.pt"),
        ("X_turn_stock_0910.pt", "y_turn_stock_0910.pt"),
        ("X_turn_stock_1112.pt", "y_turn_stock_1112.pt"),
        ("X_turn_stock_1314.pt", "y_turn_stock_1314.pt"),
        ("X_turn_stock_1516.pt", "y_turn_stock_1516.pt"),
        ]

    # for idx, layers in enumerate(model_weights):
    plt_path = f"plots/train_turn.png"
    weights_path = f"models/heus/"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    else:
        device = torch.device("cpu")
    print_(f"Device: {device}")
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # print_(f"train len: {len(train_loader)}, val len: {len(val_loader)}, data_len: {len(y)}")
    model = Heu(6*8*8 + 1, bins, [384 + 1, 500, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200], dropout)
    model.to(device)
    criterion = GaussianCrossEntropyLoss(num_bins=bins, sigma=1)
    optimizer = optim.Adam(model.parameters(), lr=leraning_rate)
    train_losses = []
    val_losses = []
    steps_list = []

    if load_num != 0:
        model.load_state_dict(torch.load(weights_path + f"stock/model_weights_{load_num}.pth", weights_only=True))        
        optimizer.load_state_dict(torch.load(weights_path + f"stock/opti_.pth", weights_only=True))


    for i in range(eps + 1):
        temp_train_loss = []
        for x_file, y_file in files:
            X = torch.load(data_path + x_file, weights_only=True)
            y = torch.load(data_path + y_file, weights_only=True)
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=32)
            train_loss = train_ep(
                model,
                loader,
                criterion,
                optimizer,
                device)
            temp_train_loss.append(train_loss)
            del X, y, dataset, loader
        
        
        train_losses.append(sum(temp_train_loss)/len(temp_train_loss))
        
        X = torch.load(data_path + "X_turn_stock_17.pt", weights_only=True)
        y = torch.load(data_path + "y_turn_stock_17.pt", weights_only=True)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loss = val_ep(
            model,
            loader,
            criterion,
            device,
            i,)
        val_losses.append(val_loss)
        del X, y, dataset, loader
        
        print_(f'Epoch [{i}/{eps}], train_loss: {train_losses[-1]:4f}, test_loss: {val_losses[-1]:4f}')
        
    plt.plot(steps_list, train_losses, label='train_loss')
    plt.plot(steps_list, val_losses, label='val_loss')
    plt.legend()
    plt.title(f"model train")
    plt.xlabel("eps")
    plt.ylabel("loss")
    plt.savefig(plt_path)
    plt.cla()