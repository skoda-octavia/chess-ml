import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, key_padding_mask=mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out