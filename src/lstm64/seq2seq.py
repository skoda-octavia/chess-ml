import torch
import torch.nn as nn
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import random
from seqModules import Seq2Seq, SequenceDataset
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


def validate(model: nn.Module, val_dl, criterion, device):

        loss_sum = 0
        model.eval()
        for idx, batch in enumerate(val_dl):
            seq, tar, legal = batch
            seq, tar, legal = seq.to(device), tar.to(device), legal.to(device)
            seq = seq.permute(1, 0)
            tar = tar.permute(1, 0)
            output = model(seq, tar)
            output = output[1:].reshape(-1, output.shape[2])
            tar = tar[1:].reshape(-1)

            loss = criterion(output, tar)
            loss_sum += loss.item()

        # # acc
        # seq = merged_data.permute(1, 0)
        # tar = merged_targs.permute(1, 0)
        # output = model(seq, tar, 0)
        # output = output.permute(1, 0, 2)
        # output = torch.argmax(output, -1)
        # acc, corr = get_valid_accs(output, sample_games_data, tar_vocab)
        return loss_sum / len(val_dl), 0, 0


def train(model, train_dl, opti, criterion, device):

        loss_sum = 0
        model.train()
        for idx, batch in enumerate(train_dl):
            seq, tar, legal = batch
            seq, tar, legal = seq.to(device), tar.to(device), legal.to(device)
            seq = seq.to(device)
            tar = tar.to(device)
            seq = seq.permute(1, 0)
            tar = tar.permute(1, 0)

            output = model(seq, tar)
            output = output[1:].reshape(-1, output.shape[2])
            tar = tar[1:].reshape(-1)

            opti.zero_grad()
            loss = criterion(output, tar)
            loss_sum += loss.item()
            loss.backward()
            opti.step()
        return loss_sum / len(train_dl)



def fit(model, train_dl, val_dl, eps, opti, criterion, device):

    train_losses, val_losses = [], []
    for i in range(eps):

        train_loss = train(model, train_dl, opti, criterion, device)
        train_losses.append(train_losses)
        
        val_loss, acc, corr = validate(model, val_dl, criterion, device)
        val_losses.append(val_loss)
        
        print(f"eps: {i}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, acc: {acc:.4f}, legal: {corr:.4f}")
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
frac = 0.002
valid_size = 0.2

seq_train, seq_val, tar_train, tar_val, legals_train, legal_valid = load_data(csv_path, frac=frac, test_size=valid_size)

max_src_len = max(find_max_length(seq_train), find_max_length(seq_val))
max_tar_len = max(find_max_length(tar_train), find_max_length(tar_val))

dataset_train = SequenceDataset(seq_train, tar_train, legals_train, src_pad, tar_pad, max_src_len, max_tar_len)
dataset_val = SequenceDataset(seq_val, tar_val, legal_valid, src_pad, tar_pad, max_src_len, max_tar_len)

dataloader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)
dataloader_val = DataLoader(dataset_val, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)

print(f"train len: {len(dataloader_train)}")
print(f"val len: {len(dataloader_val)}\n")

# train params
eps= 30
lr = 0.001
device = torch.device("cuda")
embed = 512
hidden = 512
layers = 4
drop = 0.2

# acc validation params
samples = 10240

# merged_data, merged_targs = get_samples(dataloader=dataloader_val,samples=10240)
# sample_games_data = translate_data(merged_data, merged_targs, vocab, tar_vocab)
# merged_data, merged_targs = merged_data.to(device), merged_targs.to(device)

# print(f"accuracy check data len: {len(merged_data)}\n")

model = Seq2Seq(src_vocab_len, tar_vocab_len, embed, hidden, layers, drop, src_pad, tar_pad).to(device)

criterion = nn.CrossEntropyLoss()
opti = optim.Adam(model.parameters(), lr=lr)

fit(model, dataloader_train, dataloader_val, eps, opti, criterion, device)