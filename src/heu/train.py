import torch
import torch.nn as nn
import torch.optim as optim
from src.heu.models import Heu, MyDataset, GaussianCrossEntropyLoss 
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

def print_(str: str):
    print(str)
    with open("heu_train.txt", "a") as f:
        f.write(str+"\n")


def val_ep(
        model: nn.Module,
        val_dl: DataLoader,
        criterion: nn.Module,
        device: torch.device,):

    val_losses = []
    correct_predictions = 0

    model.eval()
    for i, batch in enumerate(val_dl):
        inputs, labels = batch
        inputs, labels=  inputs.to(device), labels.to(device)
        out = model(inputs)
        # out = torch.squeeze(out)
        out = F.softmax(out, -1)
        loss = criterion(out, labels)
        val_losses.append(loss.item())

        predicted_classes = torch.argmax(out, dim=-1)
        correct_predictions += (predicted_classes == labels).sum().item()

    return sum(val_losses) / len(val_losses), correct_predictions / (len(val_losses) * inputs.shape[0])


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
        # out = torch.squeeze(out)
        out = F.softmax(out, -1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # if epoch % 10 == 0:

    return sum(train_losses) / len(train_losses)


if __name__ == "__main__":

    load_num = 0
    eps = 300
    batch_size = 4096
    leraning_rate = 0.001
    dropout = 0
    bin_num = 64
    files_num = 8
    eval_dataset_num = 9

    plt_path = "plots/"
    data_path = "data/prep/"
    files = [(f"X_{i}.pt", f"y_{i}.pt") for i in range(files_num+1)]

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

    model = Heu(6*8*8 + 1, bin_num, [384 + 1, 500, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200, 128], dropout)
    model.to(device)
    criterion = GaussianCrossEntropyLoss(bin_num, 0.6)
    optimizer = optim.Adam(model.parameters(), lr=leraning_rate)
    train_losses = []
    val_losses = []
    steps_list = []

    if load_num != 0:
        model.load_state_dict(torch.load(weights_path + f"stock/model_weights_{load_num}.pth", weights_only=True))        
        optimizer.load_state_dict(torch.load(weights_path + f"stock/opti_.pth", weights_only=True))

    for i in range(load_num + 1, eps + 1, 1):
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
            print_(str(train_loss))
            del X, y, dataset, loader
        
        
        train_losses.append(sum(temp_train_loss)/len(temp_train_loss))
        
        X = torch.load(data_path + f"X_{eval_dataset_num}.pt", weights_only=True)
        y = torch.load(data_path + f"y_{eval_dataset_num}.pt", weights_only=True)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=32)
        val_loss, val_acc = val_ep(
            model,
            loader,
            criterion,
            device,)
        val_losses.append(val_loss)
        del X, y, dataset, loader
        
        torch.save(model.state_dict(), weights_path + f"stock/model_weights_{i}.pth")
        torch.save(optimizer.state_dict(), weights_path + f"stock/opti_.pth")
        
        print_(f'Epoch [{i}/{eps}], train_loss: {train_losses[-1]:4f}, test_loss: {val_losses[-1]:4f}, val_acc: {val_acc:4f}')
        
    plt.plot(steps_list, train_losses, label='train_loss')
    plt.plot(steps_list, val_losses, label='val_loss')
    plt.legend()
    plt.title(f"model train")
    plt.xlabel("eps")
    plt.ylabel("loss")
    plt.savefig(plt_path)
    plt.cla()