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
            batch_size=32,
            tar_len=5,
            embed_size=256,
            num_layers=6,
            for_exp=4,
            heads=8,
            drop=0.1,
            device='cuda',
            max_in_len=200,
            max_out_len=5,
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
            max_in_len
            )

        self.decoder = Decoder(
            tar_voc_size,
            embed_size,
            num_layers,
            heads,
            for_exp,
            drop,
            device,
            max_out_len,
        )
        self.src_padd_idx = src_padd_idx
        self.tar_padd_idx = tar_padd_idx
        self.device = device
        self.batch_size = batch_size
        self.fc_out = nn.Linear(embed_size, tar_voc_size)
        self.soft = nn.Softmax(dim=-1)
        self.att_tar_mask = self.create_tar_attention_mask(batch_size, heads, tar_len)

    def create_src_attention_mask(self, batch_size, tar_seq_len):
        base_mask = torch.tril(torch.ones(tar_seq_len, tar_seq_len))
        mask = base_mask.masked_fill(base_mask == 0, float('-inf')).masked_fill(base_mask == 1, float(0.0))
        mask = mask.unsqueeze(0)
        mask = mask.repeat(batch_size, 1, 1)
        return mask

    def create_tar_attention_mask(self, batch_size, heads_num, tar_seq_len):
        base_mask = torch.tril(torch.ones(tar_seq_len, tar_seq_len))
        mask = base_mask.masked_fill(base_mask == 0, float('-inf')).masked_fill(base_mask == 1, float(0.0))
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(batch_size, heads_num, 1, 1)
        mask = mask.view(batch_size * heads_num, tar_seq_len, tar_seq_len)
        return mask

    def forward(self, src, tar):
        src_mask = self.create_src_attention_mask(src.shape[0], src.shape[1])
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tar, enc_src, src_mask, self.att_tar_mask)
        out = self.fc_out(out)
        log_probs = self.soft(out)

        return log_probs
