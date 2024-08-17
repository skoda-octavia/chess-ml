import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, drop):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop)

    def forward(self, x):
        embed = self.dropout(self.embed(x))
        out, (hidden, cell) = self.rnn(embed)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers, drop):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop)
        self.embed = nn.Embedding(input_size, embed_size)


    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embed = self.dropout(self.embed(x))

        out, (hidden, cell) = self.rnn(embed, (hidden, cell))
        preds = self.fc(out)
        return preds.squeeze(0), hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_len, tar_vocab_len, embed, hidden, layers, drop):
        super().__init__()
        self.encoder = Encoder(src_vocab_len, embed, hidden, layers, drop)
        self.decoder = Decoder(tar_vocab_len, embed, hidden, tar_vocab_len, layers, drop)
        self.tar_vocab_len = tar_vocab_len

    def forward(self, seq, tar, teach_force=0.5):
        batch_size = seq.shape[1]
        tar_len = tar.shape[0]
        
        hidden, cell = self.encoder(seq)

        outputs = torch.zeros(tar_len, batch_size, self.tar_vocab_len).to("cuda")
        x = tar[0]
        for i in range(1, tar_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[i] = output
            best = output.argmax(1)
            x = tar[i] if random.random() < teach_force else best
        return outputs