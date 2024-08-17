import torch
import torch.nn as nn
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import random
from seqModules import Seq2Seq

seed = 42

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class SequenceDataset(Dataset):
    def __init__(self, src_sequences, tar_sequences, src_padd_idx, tar_padd_idx, max_src_len, max_tar_len):
        self.src_sequences = [torch.tensor(seq) for seq in src_sequences]
        self.tar_sequences = [torch.tensor(seq) for seq in tar_sequences]
        # self.src_padd_idx = src_padd_idx
        # self.tar_padd_idx = tar_padd_idx
        self.max_src_len = max_src_len
        self.max_tar_len = max_tar_len

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        src_seq = self.src_sequences[idx]
        tar_seq = self.tar_sequences[idx]

        # src_seq = torch.nn.functional.pad(src_seq, (0, self.max_src_len - len(src_seq)), value=self.src_padd_idx)
        # tar_seq = torch.nn.functional.pad(tar_seq, (0, self.max_tar_len - len(tar_seq)), value=self.tar_padd_idx)

        return src_seq, tar_seq


def load_data(csv_path, test_size=0.1, random_state=42):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=0.4, random_state=random_state).reset_index(drop=True)

    sequences = df['sequence'].apply(json.loads).tolist()
    targets = df['target'].apply(json.loads).tolist()

    seq_train, seq_val, tar_train, tar_val = train_test_split(
        sequences, targets, test_size=test_size, random_state=random_state
    )

    return seq_train, seq_val, tar_train, tar_val

def find_max_length(sequences):
    return max(len(seq) for seq in sequences)


def validate(model: nn.Module, val_dl, criterion, device):

        loss_sum = 0
        model.eval()
        for idx, batch in enumerate(val_dl):
            seq, tar = batch
            seq = seq.to(device)
            tar = tar.to(device)
            seq = seq.permute(1, 0)
            tar = tar.permute(1, 0)
            output = model(seq, tar)
            output = output[1:].reshape(-1, output.shape[2])
            tar = tar[1:].reshape(-1)

            loss = criterion(output, tar)
            loss_sum += loss.item()

        return loss_sum / len(val_dl)


def train(model, train_dl, opti, criterion, device):

        loss_sum = 0
        model.train()
        for idx, batch in enumerate(train_dl):
            seq, tar = batch
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
        
        val_loss = validate(model, val_dl, criterion, device)
        val_losses.append(val_loss)
        
        print(f"eps: {i}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

vocab_path = "src/lstm/fen_vocab_src.json"
tar_vocab_path = "src/lstm/vocab_tar.json"
with open(vocab_path, "r") as f:
    vocab = json.load(f)
with open(tar_vocab_path, "r") as f:
    tar_vocab = json.load(f)

csv_path = 'data/prep/tokenized_ones.csv'
src_vocab_len = len(vocab)
tar_vocab_len = len(tar_vocab)
batch=512
num_workers=32

seq_train, seq_val, tar_train, tar_val = load_data(csv_path)

max_src_len = max(find_max_length(seq_train), find_max_length(seq_val))
max_tar_len = max(find_max_length(tar_train), find_max_length(tar_val))

dataset_train = SequenceDataset(seq_train, tar_train, src_vocab_len, tar_vocab_len, max_src_len, max_tar_len)
dataset_val = SequenceDataset(seq_val, tar_val, src_vocab_len, tar_vocab_len, max_src_len, max_tar_len)

dataloader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)
dataloader_val = DataLoader(dataset_val, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)

eps= 30
lr = 0.001
device = torch.device("cuda")
embed = 512
hidden = 1024
layers = 6
drop = 0.1

print(f"train len: {len(dataloader_train)}")
print(f"val len: {len(dataloader_val)}\n")

model = Seq2Seq(src_vocab_len, tar_vocab_len, embed, hidden, layers, drop).to(device)

criterion = nn.CrossEntropyLoss()
opti = optim.Adam(model.parameters(), lr=lr)

fit(model, dataloader_train, dataloader_val, eps, opti, criterion, device)