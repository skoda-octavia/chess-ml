import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models import MLP, MyDataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def get_acc(predicted: torch.Tensor, labels: torch.Tensor):
    labels = torch.argmax(labels, dim=1)
    correct = (predicted == labels).float()
    return correct.mean() * 100

def get_legal(predicted: torch.Tensor, legals: torch.Tensor):
    legal_predictions = legals.gather(1, predicted.view(-1, 1)).squeeze().float()
    return legal_predictions.mean() * 100


def train(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, eps: int, leraning_rate, device: torch.device, weights_path):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=leraning_rate)
    epss = []
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_legs, val_legs = [], []

    for epoch in range(eps+1):

        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0
        train_leg, val_leg = 0, 0

        model.train()
        for _, batch in enumerate(train_dl, 0):
            
            inputs, labels, legals = batch
            inputs, labels, legals = inputs.to(device), labels.to(device), legals.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted = torch.argmax(out, dim=1)
            train_acc += get_acc(predicted, labels).item()
            train_leg += get_legal(predicted, legals).item()

        model.eval()
        for _, batch in enumerate(val_dl):
            inputs, labels, legals = batch
            inputs, labels, legals = inputs.to(device), labels.to(device), legals.to(device)
            out = model(inputs)
            loss = criterion(out, labels)
            val_loss += loss.item()

            predicted = torch.argmax(out, dim=1)
            val_acc += get_acc(predicted, labels).item()
            val_leg += get_legal(predicted, legals).item()

        train_loss = train_loss / len(train_dl)
        train_acc = train_acc / len(train_dl)
        train_leg = train_leg / len(train_dl)
        val_loss = val_loss / len(val_dl)
        val_acc = val_acc / len(val_dl)
        val_leg = val_leg / len(val_dl)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_legs.append(train_leg)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_legs.append(val_leg)

        epss.append(epoch)

        # if epoch % 10 == 0:
        torch.save(model.state_dict(), weights_path + f"model_weights_{epoch}.pth")

        print(f'Epoch [{epoch}/{eps}]')
        print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_leg: {train_leg:.4f}")
        print(f"val_loss:   {val_loss:.4f}, val_acc:   {val_acc:.4f}, val_leg:   {val_leg:.4f}")

    return epss, train_losses, train_accs, train_legs, val_losses, val_accs, val_legs




def train_model(piece: str, model_in, batch):
    eps = 1
    batch_size = batch
    lr = 0.001
    dropout = 0
    test_size = 0.1

    data_path = "data/prep/mlp/"
    x_file = data_path + f"X_{piece}.pt"
    y_file = data_path + f"y_{piece}.pt"
    legals_file = data_path + f"legals_{piece}.pt"

    plt_path = f"plots/train_{piece}.png"
    weights_path = f"models/mlp/{piece}/"

    X = torch.load(x_file, weights_only=True)
    y = torch.load(y_file, weights_only=True)
    legals = torch.load(legals_file, weights_only=True)

    X_train, X_val, y_train, y_val, legals_train, legals_val = train_test_split(
        X, y, legals, test_size=test_size, random_state=42
    )

    train_dataset = MyDataset(X_train, y_train, legals_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=32)
    val_dataset = MyDataset(X_val, y_val, legals_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=32)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"train len: {len(train_loader)}, val len: {len(val_loader)}, data_len: {len(y)}")
    model = MLP(model_in, 64, [model_in, 500, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200, 100, 64], dropout)
    epss, train_losses, train_accs, train_legs, val_losses, val_accs, val_legs = train(
        model, train_loader, valid_loader, eps, lr, device, weights_path)

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    axs[0].plot(epss, train_losses, label='train_loss')
    axs[0].plot(epss, val_losses, label='val_loss')
    axs[0].legend()
    axs[0].set_title(f"{piece} learning loss")
    axs[0].set_xlabel("eps")
    axs[0].set_ylabel("loss")

    axs[1].plot(epss, train_accs, label='train_acc')
    axs[1].plot(epss, val_accs, label='val_acc')
    axs[1].plot(epss, train_legs, label='train_legals')
    axs[1].plot(epss, val_legs, label='val_legals')    
    axs[1].legend()
    axs[1].set_title(f"{piece} learning accuracy and legals")
    axs[1].set_xlabel("eps")
    axs[1].set_ylabel("percent")
    
    fig.savefig(plt_path)
    plt.cla()

if __name__ == "__main__":
    piece_types = [chess.PAWN, chess.QUEEN, chess.KING, chess.ROOK, chess.BISHOP, chess.KNIGHT]

    train_model("piece_select", 6*8*8, 1024)
    for piece_type in piece_types:
        piece_name = chess.piece_name(piece_type)
        train_model(piece_name, 7*8*8, 256)