import torch
import torch.nn as nn

class SelfAtt(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out =  nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask, size):
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(size, key_len, self.heads, self.head_dim)
        query = query.reshape(size, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            size, query_len, self.heads*self.head_dim
        )

        out = self.fc_out(out)
        return out

