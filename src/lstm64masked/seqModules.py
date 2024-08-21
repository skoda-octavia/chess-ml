import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class SequenceDataset(Dataset):
    def __init__(self, src_sequences, tar_sequences, legals, src_padd_idx, tar_padd_idx, max_src_len, max_tar_len):
        
        self.max_legal_len = len(legals[0])
        self.tar_move_len = len(legals[0][0])
        self.src_padd_idx = src_padd_idx
        self.tar_padd_idx = tar_padd_idx       
        self.max_src_len = max_src_len
        self.max_tar_len = max_tar_len
        self.src_sequences = [torch.tensor(seq) for seq in src_sequences]
        self.tar_sequences = [torch.tensor(seq) for seq in tar_sequences]
        self.legals = [self.generate_legal_mask(legal).cpu() for legal in legals]


    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        src_seq = self.src_sequences[idx]
        tar_seq = self.tar_sequences[idx]
        legals = self.legals[idx]

        src_seq = torch.nn.functional.pad(src_seq, (0, self.max_src_len - len(src_seq)), value=self.src_padd_idx)
        tar_seq = torch.nn.functional.pad(tar_seq, (0, self.max_tar_len - len(tar_seq)), value=self.tar_padd_idx)

        return src_seq, tar_seq, legals


    def generate_legal_mask(self, legals: list):
        legals = torch.tensor(legals).to("cuda")
        legals = F.relu(legals)
        one_hot_encoded = torch.nn.functional.one_hot(legals, num_classes=self.tar_padd_idx+1).to("cuda")
        one_hot_encoded = one_hot_encoded.float()
        ones_tensor = torch.ones((self.max_legal_len, self.max_tar_len-1 - self.tar_move_len, self.tar_padd_idx+1), dtype=torch.float).to("cuda")
        final_tensor = torch.cat((one_hot_encoded, ones_tensor), dim=1).to("cuda")
        return final_tensor


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
        return self.acc(preds.squeeze(0)), hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_len, tar_vocab_len, embed, hidden, layers, drop, src_pad, tar_pad):
        super().__init__()
        self.encoder = Encoder(src_vocab_len, embed, hidden, layers, drop, src_pad)
        self.decoder = Decoder(tar_vocab_len, embed, hidden, tar_vocab_len, layers, drop, tar_pad)
        self.tar_vocab_len = tar_vocab_len

    def forward(self, seq, tar, mask):
        batch_size = seq.shape[1]
        tar_len = tar.shape[0]
        
        hidden, cell = self.encoder(seq)

        outputs = torch.zeros(tar_len, batch_size).to(seq.device)
        x = tar[0]

        mask_cl = mask.clone()
        for i in range(1, tar_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            
            mask_next_step = self.get_mask_next_step(mask_cl, i-1).max(dim=1)[0]   
            output = nn.functional.relu(output)
            output = output * mask_next_step
            
            best = output.argmax(1)
            x = best

            outputs[i] = best

            mask_cl = self.update_mask(mask_cl, output, best, i)
        
        return outputs

    def get_mask_next_step(self, mask, step):
        if step < mask.shape[2]:
            return mask[:, :, step, :]
        else:
            return torch.ones_like(mask[:, :, 0, :])

    def update_mask(self, mask, output, best, step):
        batch_size, max_legal_moves, target_seq_len, tokens_num = mask.shape
        output_expanded = torch.zeros(batch_size, max_legal_moves, target_seq_len, tokens_num, device=mask.device)
        best_expanded = best.view(batch_size, 1, 1, 1).expand(batch_size, max_legal_moves, target_seq_len, 1)
        output_expanded.scatter_(3, best_expanded, 1)
        
        if step < mask.shape[2]:
            mask_to_keep = output_expanded[:, :, step, :]
            mask[:, :, step, :] *= mask_to_keep

        return mask