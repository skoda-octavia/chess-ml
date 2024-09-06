import torch

# Przykładowa maska o wymiarach batch_size x max_moves_len x tokens_num
batch_size = 3
max_moves_len = 4
tokens_num = 11

# Tworzymy przykładową maskę
mask = torch.tensor([[
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
],
[
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]]

) # sample in batch ->     [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],

# Zsumuj maskę wzdłuż wymiaru max_moves_len (drugi wymiar)
print(mask.shape)
summed_mask = mask.max(dim=1)

print("Zsumowana maska:")
print(summed_mask)