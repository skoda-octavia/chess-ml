import torch
import torch.nn as nn

class SelfAtt(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(heads)])
        self.keys = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(heads)])
        self.queries = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(heads)])
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask, size):
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(size, key_len, self.heads, self.head_dim)
        query = query.reshape(size, query_len, self.heads, self.head_dim)

        values = torch.stack([self.values[i](values[:, :, i, :]) for i in range(self.heads)], dim=2)
        keys = torch.stack([self.keys[i](keys[:, :, i, :]) for i in range(self.heads)], dim=2)
        query = torch.stack([self.queries[i](query[:, :, i, :]) for i in range(self.heads)], dim=2)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            size, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
