import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock

class DecoderBlock(nn.Module):

    def __init__(self, embed_size, heads, forward_expansion, drop, device) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, drop, batch_first=True)
        self.norm = nn.LayerNorm(embed_size)
        self.trans_block = TransformerBlock(embed_size, heads, drop, forward_expansion)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, value, key, src_mask, tar_mask):
        attention = self.attention(x, x, x, attn_mask=tar_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.trans_block(value, key, query, src_mask)
        return out
