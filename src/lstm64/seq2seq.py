import torch
import torch.nn as nn
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import random
from seqModules import Seq2Seq, SequenceDataset, ChessLoss
from utils import (
    get_samples,
    translate_data,
    get_valid_accs,
    load_data,
    find_max_length
    )

seed = 42

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def print_(str):
     print(str)
     with open("rep.txt", "a") as f:
          f.write(str+"\n")


def validate(model: nn.Module, val_dl: DataLoader, criterion, device):

        loss_sum = 0
        legals = 0
        accs = 0
        model.eval()
        batch_size = val_dl.batch_size

        for idx, batch in enumerate(val_dl):
            seq, tar, legal = batch
            seq, tar, legal = seq.to(device), tar.to(device), legal.to(device)
            seq = seq.permute(1, 0)
            tar_perm = tar.permute(1, 0)

            output = model(seq, tar_perm, 0)

            # get legals ratio 
            output_class = output.permute(1, 0, 2)
            output_moves = torch.argmax(output_class, dim=2)[:, 1:]
            output_fields = output_moves[:, :2]
            predictions_expanded = output_fields.unsqueeze(1).expand(batch_size, 1, 2)
            matches = torch.all(predictions_expanded.eq(legal), dim=2)
            num_legal_moves = torch.sum(matches).item()
            total_moves = batch_size
            fraction_of_legal_moves = num_legal_moves / total_moves
            legals += fraction_of_legal_moves

            # get acc
            output_acc = output_moves[:, :-1]
            matches = torch.eq(output_acc, tar[:, 1:-1])
            sequence_matches = torch.all(matches, dim=1)
            acc = torch.sum(sequence_matches).item() 
            accs += acc / batch_size

            output = output[1:].reshape(-1, output.shape[2])
            tar_perm = tar_perm[1:].reshape(-1)

            opti.zero_grad()
            if isinstance(criterion, ChessLoss): 
                loss = criterion(output, tar_perm, fraction_of_legal_moves)
            else:
                loss = criterion(output, tar_perm)

            loss_sum += loss.item()

        return loss_sum / len(val_dl), legals / len(val_dl), accs / len(val_dl)


def train(model: Seq2Seq, train_dl: DataLoader, opti, criterion, device):

        loss_sum = 0
        legals = 0
        accs = 0

        model.train()
        batch_size = train_dl.batch_size

        for idx, batch in enumerate(train_dl):
            seq, tar, legal = batch
            seq, tar, legal = seq.to(device), tar.to(device), legal.to(device)
            seq = seq.permute(1, 0)
            tar_perm = tar.permute(1, 0)

            output = model(seq, tar_perm, 1)

            # get legals ratio 
            output_class = output.permute(1, 0, 2)
            output_moves = torch.argmax(output_class, dim=2)[:, 1:]
            output_fields = output_moves[:, :2]
            predictions_expanded = output_fields.unsqueeze(1).expand(batch_size, 1, 2)
            matches = torch.all(predictions_expanded.eq(legal), dim=2)
            num_legal_moves = torch.sum(matches).item()
            total_moves = batch_size
            fraction_of_legal_moves = num_legal_moves / total_moves
            legals += fraction_of_legal_moves

            # get acc
            output_acc = output_moves[:, :-1]
            matches = torch.eq(output_acc, tar[:, 1:-1])
            sequence_matches = torch.all(matches, dim=1)
            acc = torch.sum(sequence_matches).item() 
            accs += acc / batch_size


            output = output[1:].reshape(-1, output.shape[2])
            tar_perm = tar_perm[1:].reshape(-1)

            opti.zero_grad()
            if isinstance(criterion, ChessLoss): 
                loss = criterion(output, tar_perm, fraction_of_legal_moves)
            else:
                loss = criterion(output, tar_perm)
            loss_sum += loss.item()
            loss.backward()
            opti.step()
        
        return loss_sum / len(train_dl), legals / len(train_dl), accs / len(train_dl)



def fit(model, train_dl, val_dl, eps, opti, criterion, device):

    train_losses, val_losses = [], []
    for i in range(eps):

        train_loss, tr_leg, tr_acc = train(model, train_dl, opti, criterion, device)
        train_losses.append(train_losses)
        
        val_loss, val_leg, val_acc = validate(model, val_dl, criterion, device)
        val_losses.append(val_loss)
        
        print_(f"eps: {i}")
        print_(f"train loss: {train_loss:.4f}, acc: {tr_acc:.4f}, legal: {tr_leg:.4f}")
        print_(f"valid loss: {val_loss:.4f}, acc: {val_acc:.4f}, legal: {val_leg:.4f}")
        torch.save(model.state_dict(), f"transformer{i}.pth")
        torch.save(opti.state_dict(), f"opti{i}.pth")

vocab_path = "src/lstm64/vocab_src.json"
tar_vocab_path = "src/lstm64/vocab_tar.json"
with open(vocab_path, "r") as f:
    vocab = json.load(f)
with open(tar_vocab_path, "r") as f:
    tar_vocab = json.load(f)

csv_path = 'data/prep/tokenized64.csv'
src_pad = len(vocab)
tar_pad = len(tar_vocab)

src_vocab_len = src_pad + 1
tar_vocab_len = tar_pad + 1

# data params
batch=512
num_workers=32
frac = 0.2
valid_size = 0.2

seq_train, seq_val, tar_train, tar_val, legals_train, legal_valid = load_data(csv_path, frac=frac, test_size=valid_size)

max_src_len = max(find_max_length(seq_train), find_max_length(seq_val))
max_tar_len = max(find_max_length(tar_train), find_max_length(tar_val))

dataset_train = SequenceDataset(seq_train, tar_train, legals_train, src_pad, tar_pad, max_src_len, max_tar_len)
dataset_val = SequenceDataset(seq_val, tar_val, legal_valid, src_pad, tar_pad, max_src_len, max_tar_len)

dataloader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)
dataloader_val = DataLoader(dataset_val, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)

print_(f"train len: {len(dataloader_train)}")
print_(f"val len: {len(dataloader_val)}\n")

# train params
eps= 60
lr = 0.001
device = torch.device("cuda")
embed = 512
hidden = 512
layers = 4
drop = 0.2

# acc validation params
samples = 10240

model = Seq2Seq(src_vocab_len, tar_vocab_len, embed, hidden, layers, drop, src_pad, tar_pad).to(device)

# criterion = ChessLoss(tar_pad, tar_vocab_len, 1.5)
criterion = nn.CrossEntropyLoss(ignore_index=tar_pad)

opti = optim.Adam(model.parameters(), lr=lr)

fit(model, dataloader_train, dataloader_val, eps, opti, criterion, device)