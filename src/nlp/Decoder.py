import torch
import torch.nn as nn
from DecoderBlock import DecoderBlock


class Decoder(nn.Module):

    def __init__(self, tar_vocab_size, embed_size, layers, heads, forw_exp, drop, device, max_len) -> None:
        super().__init__()
        self.device = device
        self.word_embed = nn.Embedding(tar_vocab_size, embed_size)
        self.position_emb = nn.Embedding(max_len, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forw_exp, drop, device) for _ in range(layers)]
        )
        self.fc_out = nn.Linear(embed_size, tar_vocab_size)
        self.dropout = nn.Dropout(drop)


    def forward(self, x, enc_out, src_mask, tar_mask):
        size, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(size, seq_len).to(self.device)
        x = self.dropout(self.word_embed(x) + self.position_emb(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tar_mask)

        out = self.fc_out(x)
        return out