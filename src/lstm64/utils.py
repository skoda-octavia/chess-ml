import torch
import torch.nn as nn
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import random
from seqModules import Seq2Seq, SequenceDataset
import chess


def load_data(csv_path, test_size=0.1, random_state=42, frac=0.4):
    df = pd.read_csv(csv_path)
    
    df = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    
    sequences = df['sequence'].apply(json.loads).tolist()
    targets = df['target'].apply(json.loads).tolist()
    acceptables = df['legal'].apply(json.loads).tolist()

    seq_train, seq_val, tar_train, tar_val, acc_train, acc_val = train_test_split(
        sequences, targets, acceptables, test_size=test_size, random_state=random_state
    )

    return seq_train, seq_val, tar_train, tar_val, acc_train, acc_val

def find_max_length(sequences):
    return max(len(seq) for seq in sequences)

seed = 42

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
