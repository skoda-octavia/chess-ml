import torch
from torch.utils.data import DataLoader
from SequenceDataset import SequenceDataset
import pandas as pd
import ast
from Transformer import Transformer
import torch.optim as optim
import torch.nn as nn
import json
from sklearn.model_selection import train_test_split
import chess
from torch.nn.utils.rnn import pad_sequence


with open("src/nlp/vocab.json", "r") as f:
    vocab = json.load(f)
reverse_vocab = {value: key for key, value in vocab.items()}

def create_board(game_tokens):
    board = chess.Board()
    move = ""
    new_tensor = torch.tensor([70, 72]).to("cuda")
    extended_tensor = torch.cat((game_tokens, new_tensor), dim=0)
    for token in extended_tensor:
        if token.item() == 72:
            return board
        if token.item() == 70:
            board.push(chess.Move.from_uci(move))
            move = ""
        elif token < 64:
            move += reverse_vocab[token.item()]
        elif token < 68:
            move += reverse_vocab[token.item()][1].lower()

def evaluate_prediction(output, targets, src_batch):
    predictions = torch.argmax(output, dim=2)
    comparison = (predictions == targets)
    accuracy = comparison.all(dim=1).sum().item()
    correct = 0
    for game, pred in zip(src_batch, predictions):
        board = create_board(game)
        first = reverse_vocab[pred[0].item()]
        sec = reverse_vocab[pred[1].item()]
        move = first+sec
        third = pred[2].item()
        if 63 < third < 68:
            move += reverse_vocab[third][1].lower()
        for leg_move in board.legal_moves:
            if leg_move == move:
                correct += 1

    return accuracy, correct


def load_data(csv_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    sequences = df['sequence'].apply(ast.literal_eval).tolist()
    targets = df['target'].apply(ast.literal_eval).tolist()
    seq_train, seq_val, tar_train, tar_val = train_test_split(
        sequences, targets, test_size=test_size, random_state=random_state
    )
    return seq_train, seq_val, tar_train, tar_val

def collate_fn(self, batch):
    src_batch, tar_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.src_padd_idx)
    tar_batch = pad_sequence(tar_batch, batch_first=True, padding_value=self.tar_padd_idx)
    return src_batch, tar_batch

csv_path = 'data/prep/tokenized.csv'
src_padd_idx = -1
tar_padd_idx = -1

seq_train, seq_val, tar_train, tar_val = load_data(csv_path)

dataset_train = SequenceDataset(seq_train, tar_train, src_padd_idx, tar_padd_idx)
dataset_val = SequenceDataset(seq_val, tar_val, src_padd_idx, tar_padd_idx)

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_fn)

def train_model(model, dataloader_train, dataloader_val, num_epochs, learning_rate, device):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tar_padd_idx)

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for i, data in enumerate(dataloader_train, 0):
            src_batch, tar_batch = data
            src_batch = src_batch.to(device)
            tar_batch = tar_batch.to(device)

            outputs = model(src_batch, tar_batch[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tar_batch[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        eps_loss_val = 0
        acc = 0
        corr = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader_val, 0):
                src_batch, tar_batch = data
                src_batch = src_batch.to(device)
                tar_batch = tar_batch.to(device)

                outputs = model(src_batch, tar_batch[:, :-1])
                # temp_acc, temp_corr = evaluate_prediction(outputs, tar_batch[:, 1:], src_batch)
                # acc += temp_acc
                # corr += temp_corr
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tar_batch[:, 1:].reshape(-1))
                eps_loss_val += loss.item()

        avg_accurate = acc / len(dataloader_val)
        avg_correct = corr / len(dataloader_val)
        avg_loss_eval = eps_loss_val / len(dataloader_val)
        avg_loss = epoch_loss / len(dataloader_train)
        print(f"Epoch {epoch+1} loss train: {avg_loss:.4f}, loss val: {avg_loss_eval:.4f}, acc val: {avg_accurate}, correct val: {avg_correct}")
        torch.save(model.state_dict(), f"models/nlp/transformer{epoch}.pth")


num_epochs = 150
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_pad_idx = -1
trg_pad_idx = -1
vocab_path = "src/nlp/fen_vocab.json"
tar_vocab_path = "src/nlp/vocab.json"
with open(vocab_path, "r") as f:
    vocab = json.load(f)
with open(tar_vocab_path, "r") as f:
    tar_vocab = json.load(f)

src_vocab_size = len(vocab.items())
trg_vocab_size = len(tar_vocab.items())
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device, embed_size=512, num_layers=5).to(
    device
)
# model.load_state_dict(torch.load("models/nlp/transformer26.pth"))
model.to(device)


train_model(model, dataloader_train, dataloader_val, num_epochs, learning_rate, device)