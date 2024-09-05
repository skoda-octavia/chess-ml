import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class SequenceDataset(Dataset):
    def __init__(self, src_sequences, tar_sequences, legals, src_padd_idx, tar_padd_idx, max_src_len, max_tar_len):
        self.src_sequences = [torch.tensor(seq) for seq in src_sequences]
        self.tar_sequences = [torch.tensor(seq) for seq in tar_sequences]
        self.legals = [torch.tensor(legals[i]) for i in range(len(legals))]
        self.src_padd_idx = src_padd_idx
        self.tar_padd_idx = tar_padd_idx
        self.max_src_len = max_src_len
        self.max_tar_len = max_tar_len

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        src_seq = self.src_sequences[idx]
        tar_seq = self.tar_sequences[idx]
        legals = self.legals[idx]

        src_seq = torch.nn.functional.pad(src_seq, (0, self.max_src_len - len(src_seq)), value=self.src_padd_idx)
        tar_seq = torch.nn.functional.pad(tar_seq, (0, self.max_tar_len - len(tar_seq)), value=self.tar_padd_idx)

        return src_seq, tar_seq, legals

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, drop, src_pad):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.embed = nn.Embedding(input_size, embed_size, src_pad)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop)

    def forward(self, x):
        embed = self.dropout(self.embed(x))
        out, (hidden, cell) = self.rnn(embed)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers, drop, tar_pad):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop)
        self.embed = nn.Embedding(input_size, embed_size, tar_pad)
        self.acc = nn.ReLU()


    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embed = self.dropout(self.embed(x))

        out, (hidden, cell) = self.rnn(embed, (hidden, cell))
        preds = self.fc(out)
        return preds.squeeze(0), hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_len, tar_vocab_len, embed, hidden, layers, drop, src_pad, tar_pad):
        super().__init__()
        self.encoder = Encoder(src_vocab_len, embed, hidden, layers, drop, src_pad)
        self.decoder = Decoder(tar_vocab_len, embed, hidden, tar_vocab_len, layers, drop, tar_pad)
        self.tar_vocab_len = tar_vocab_len

    def forward(self, seq, tar, teach_force=0.5, mask=None):
        batch_size = seq.shape[1]
        tar_len = tar.shape[0]
        
        if mask is not None and batch_size != 1:
            raise ValueError("Mask was given but batch size was not size 1")

        hidden, cell = self.encoder(seq)
        outputs = torch.zeros(tar_len, batch_size, self.tar_vocab_len).to("cuda")
        x = tar[0]
        for i in range(1, tar_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            
            if mask is not None:
                output = F.softmax(output, -1)
                output = self.filter_output(output, mask, i-1)
                best = output.argmax(1)
                mask = self.adjust_mask(mask, best, i-1)
            else:
                best = output.argmax(1)

            outputs[i] = output
            x = tar[i] if random.random() < teach_force else best
        return outputs
    

    def filter_output(self, output, mask, idx):
        output = torch.squeeze(output)
        move_mask = mask[:, idx, :]
        max_values, _ = torch.max(move_mask, dim=0)
        output *= max_values
        return torch.unsqueeze(output, 0)
    
    def adjust_mask(self, mask, best, idx):
        selected = best[0].item()
        check = mask[:, idx, selected] == 1
        mask[~check] = 0

        return mask
