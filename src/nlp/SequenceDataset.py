from torch.utils.data import Dataset
import torch


class SequenceDataset(Dataset):
    def __init__(self, src_sequences, tar_sequences, src_padd_idx, tar_padd_idx, max_src_len, max_tar_len):
        self.src_sequences = [torch.tensor(seq) for seq in src_sequences]
        self.tar_sequences = [torch.tensor(seq) for seq in tar_sequences]
        self.src_padd_idx = src_padd_idx
        self.tar_padd_idx = tar_padd_idx
        self.max_src_len = max_src_len
        self.max_tar_len = max_tar_len

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        src_seq = self.src_sequences[idx]
        tar_seq = self.tar_sequences[idx]

        # Padujemy do maksymalnej długości
        src_seq = torch.nn.functional.pad(src_seq, (0, self.max_src_len - len(src_seq)), value=self.src_padd_idx)
        tar_seq = torch.nn.functional.pad(tar_seq, (0, self.max_tar_len - len(tar_seq)), value=self.tar_padd_idx)

        return src_seq, tar_seq