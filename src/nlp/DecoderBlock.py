import torch
import torch.nn as nn
from Attention import SelfAtt
from TransformerBlock import TransformerBlock

class DecoderBlock(nn.Module):

    def __init__(self, embed_size, heads, forward_expansion, drop, device) -> None:
        super().__init__()
        self.attention = SelfAtt(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.trans_block = TransformerBlock(embed_size, heads, drop, forward_expansion)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, value, key, src_mask, tar_mask):
        attention = self.attention(x, x, x, tar_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.trans_block(value, key, query)
        return out
