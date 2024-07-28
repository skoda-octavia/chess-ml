import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock
from Attention import SelfAtt


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, layers, heads, device, forward_expansion, dropout, max_len) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_len, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                ) for _ in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        size, seq_len = x.shape
        position = torch.arange(0, seq_len).expand(size, seq_len).to(self.device)
        out = self.dropout(self.word_embed(x) + self.pos_embed(position))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out