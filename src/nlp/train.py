import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from Transformer import Transformer
import torch.optim as optim
import torch.nn as nn
import json


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets, src_padd_idx=102, tar_padd_idx=102):
        self.sequences = sequences
        self.targets = targets
        self.src_padd_idx = src_padd_idx
        self.tar_padd_idx = tar_padd_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        src_seq = self.sequences[idx]
        tar_seq = self.targets[idx]
        return torch.tensor(src_seq, dtype=torch.long), torch.tensor(tar_seq, dtype=torch.long)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    sequences = df['sequence'].apply(ast.literal_eval).tolist()
    targets = df['target'].apply(ast.literal_eval).tolist()
    return sequences, targets

csv_path = 'data/prep/tokenized.csv'

src_padd_idx = 102
tar_padd_idx = 102

sequences, targets = load_data(csv_path)
dataset = SequenceDataset(sequences, targets, src_padd_idx, tar_padd_idx)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

def train_model(model, dataloader, num_epochs, learning_rate, device):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tar_padd_idx)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, data in enumerate(dataloader, 0):
            src_batch, tar_batch = data
            src_batch = src_batch.to(device)
            tar_batch = tar_batch.to(device)

            outputs = model(src_batch, tar_batch[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tar_batch[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")


num_epochs = 20
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_pad_idx = 102
trg_pad_idx = 102
vocab_path = "src/nlp/vocab.json"
with open(vocab_path, "r") as f:
    vocab = json.load(f)

src_vocab_size = len(vocab.items())
trg_vocab_size = src_vocab_size
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
    device
)
model.to(device)


train_model(model, dataloader, num_epochs, learning_rate, device)