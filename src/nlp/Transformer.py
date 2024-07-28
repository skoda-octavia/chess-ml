import torch
import torch.nn as nn
from Decoder import Decoder
from Encoder import Encoder

class Transformer(nn.Module):

    def __init__(
            self,
            src_voc_size,
            tar_voc_size,
            src_padd_idx,
            tar_padd_idx,
            embed_size=256,
            num_layers=6,
            for_exp=4,
            heads=8,
            drop=0.1,
            device='cuda',
            max_len=100
            ) -> None:
        super().__init__()

        self.encoder = Encoder(
            src_voc_size,
            embed_size,
            num_layers,
            heads,
            device,
            for_exp,
            drop,
            max_len
            )

        self.decoder = Decoder(
            tar_voc_size,
            embed_size,
            num_layers,
            heads,
            for_exp,
            drop,
            device,
            max_len,
        )
        self.src_padd_idx = src_padd_idx
        self.tar_padd_idx = tar_padd_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_padd_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_target_mask(self, tar):
        size, tar_len = tar.shape
        tar_mask = torch.tril(torch.ones((tar_len, tar_len))).expand(size, 1, tar_len, tar_len)
        return tar_mask.to(self.device)

    def forward(self, src, tar):
        src_mask = self.make_src_mask(src)
        tar_mask = self.make_target_mask(tar)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tar, enc_src, src_mask, tar_mask)

        return out
