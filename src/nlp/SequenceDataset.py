from torch.utils.data import Dataset
import torch


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets, src_padd_idx=102, tar_padd_idx=102):
        self.sequences = sequences
        self.targets = targets
        self.src_padd_idx = src_padd_idx
        self.tar_padd_idx = tar_padd_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        src_seq = self.sequences[idx]
        tar_seq = self.targets[idx]
        return torch.tensor(src_seq, dtype=torch.long), torch.tensor(tar_seq, dtype=torch.long)