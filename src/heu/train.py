import torch
import torch.nn as nn
import torch.optim as optim
from models import Heu, MyDataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def print_(str):
    print(str)
    with open("heu_train.txt", "a") as f:
        f.write(str+"\n")


def train(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, eps: int, leraning_rate, device: torch.device, weights_path):

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=leraning_rate)
    epss = []
    train_losses = []
    val_losses = []

    for epoch in range(eps+1):

        train_loss = 0
        val_loss = 0
        model.train()
        for i, batch in enumerate(train_dl, 0):
            
            inputs, labels = batch
            inputs, labels=  inputs.to(device), labels.to(device)
            out = model(inputs)
            out = torch.squeeze(out)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        model.eval()
        for i, batch in enumerate(val_dl):
            inputs, labels = batch
            inputs, labels=  inputs.to(device), labels.to(device)
            out = model(inputs)
            out = torch.squeeze(out)
            loss = criterion(out, labels)
            val_loss += loss.item()


        train_loss = train_loss / len(train_dl)
        val_loss = val_loss / len(val_dl)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epss.append(epoch)

        # if epoch % 10 == 0:
        torch.save(model.state_dict(), weights_path + f"model_weights_{epoch}.pth")

        print_(f'Epoch [{epoch}/{eps}], train_loss: {train_loss:4f}, test_loss: {val_loss:4f}')

    return epss, train_losses, val_losses


if __name__ == "__main__":

    # model_weights = [
    #     [384, 400, 500, 500, 400, 300, 200, 100, 64],
    #     [384, 400, 500, 700, 700, 700, 500, 300, 200, 100, 64],
    #     [384, 500, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200, 100, 64],
    #     [384, 600, 1000, 1600, 2000, 2000, 2000, 1600, 1000, 800, 600, 400, 200, 100, 64],
    #     [384, 600, 1200, 2000, 2800, 3600, 4000, 4000, 4000, 3200, 2400, 1600, 800, 400, 64]
    # ] 

    eps = 500
    batch_size = 4096
    lr = 0.001
    dropout = 0
    test_size = 0.1

    data_path = "data/prep/"
    x_file = data_path + "X_turn.pt"
    y_file = data_path + "y_turn.pt"


    # for idx, layers in enumerate(model_weights):
    plt_path = f"plots/train_turn.png"
    weights_path = f"models/heus/"

    X = torch.load(x_file, weights_only=True)
    y = torch.load(y_file, weights_only=True)
    y = y.float()
    dataset = TensorDataset(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    train_dataset = MyDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=32)
    val_dataset = MyDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=32)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    else:
        device = torch.device("cpu")
    print_(f"Device: {device}")
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print_(f"train len: {len(train_loader)}, val len: {len(val_loader)}, data_len: {len(y)}")
    model = Heu(6*8*8 + 1, 1, [384 + 1, 500, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200, 100, 64], dropout)
    epss, train_loss, val_loss = train(model, train_loader, valid_loader, eps, lr, device, weights_path)

    plt.plot(epss, train_loss, label='train_loss')
    plt.plot(epss, val_loss, label='val_loss')
    plt.legend()
    plt.title(f"model train")
    plt.xlabel("eps")
    plt.ylabel("loss")
    plt.savefig(plt_path)
    plt.cla()