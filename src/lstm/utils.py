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

    seq_train, seq_val, tar_train, tar_val = train_test_split(
        sequences, targets, test_size=test_size, random_state=random_state
    )

    return seq_train, seq_val, tar_train, tar_val

def find_max_length(sequences):
    return max(len(seq) for seq in sequences)

seed = 42

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


enpassant_moves = [
    (chess.A5, chess.B6), (chess.B5, chess.A6),
    (chess.B5, chess.C6), (chess.C5, chess.B6),
    (chess.C5, chess.D6), (chess.D5, chess.C6),
    (chess.D5, chess.E6), (chess.E5, chess.D6),
    (chess.E5, chess.F6), (chess.F5, chess.E6),
    (chess.F5, chess.G6), (chess.G5, chess.F6),
    (chess.G5, chess.H6), (chess.H5, chess.G6),
    
    (chess.A4, chess.B3), (chess.B4, chess.A3),
    (chess.B4, chess.C3), (chess.C4, chess.B3),
    (chess.C4, chess.D3), (chess.D4, chess.C3),
    (chess.D4, chess.E3), (chess.E4, chess.D3),
    (chess.E4, chess.F3), (chess.F4, chess.E3),
    (chess.F4, chess.G3), (chess.G4, chess.F3),
    (chess.G4, chess.H3), (chess.H4, chess.G3)
]

def translate_move(move_tokens, reverse_tar_vocab):
    move_elems = [reverse_tar_vocab[token.item()] for token in move_tokens]
    move = "".join(move_elems)
    move = chess.Move.from_uci(move)
    return move

def translate_fen(fen_tokens, reverse_src_vocab):
    
    convert_nums = [8,7,6,5,4,3,2]
    fen_elems = [reverse_src_vocab[token.item()] for token in fen_tokens[:-1]]
    fen = "".join(fen_elems)    
    for num in convert_nums:
        old = "1" * num
        fen = fen.replace(old, str(num))
    turn_str = reverse_src_vocab[fen_tokens[-1].item()]
    board = chess.Board(fen)
    board.turn = turn_str == 'True'
    board.castling_rights |= chess.BB_A1 | chess.BB_H1 | chess.BB_A8 | chess.BB_H8
    return board


def translate_data(merged_data, merged_targs, vocab: dict, tar_vocab: dict):

    enpassant_moves = [
        (chess.A5, chess.B6), (chess.B5, chess.A6),
        (chess.B5, chess.C6), (chess.C5, chess.B6),
        (chess.C5, chess.D6), (chess.D5, chess.C6),
        (chess.D5, chess.E6), (chess.E5, chess.D6),
        (chess.E5, chess.F6), (chess.F5, chess.E6),
        (chess.F5, chess.G6), (chess.G5, chess.F6),
        (chess.G5, chess.H6), (chess.H5, chess.G6),
        
        (chess.A4, chess.B3), (chess.B4, chess.A3),
        (chess.B4, chess.C3), (chess.C4, chess.B3),
        (chess.C4, chess.D3), (chess.D4, chess.C3),
        (chess.D4, chess.E3), (chess.E4, chess.D3),
        (chess.E4, chess.F3), (chess.F4, chess.E3),
        (chess.F4, chess.G3), (chess.G4, chess.F3),
        (chess.G4, chess.H3), (chess.H4, chess.G3)
    ]
    moves, legal_moves_data = [], []
    reverse_src = {item: key for key, item in vocab.items()}
    reverse_tar = {item: key for key, item in tar_vocab.items()}
    
    for game, move in zip(merged_data[:, 1:-1], merged_targs[:, 1:-1]):
        move = translate_move(move, reverse_tar)
        board = translate_fen(game, reverse_src)
        legal_moves = list(board.legal_moves)
        ss = board.piece_at(move.from_square)
        if (move.from_square, move.to_square) in enpassant_moves and board.piece_at(move.from_square).piece_type == chess.PAWN:
            legal_moves.append(move)
        else:
            assert move in legal_moves            
        moves.append(move)
        legal_moves_data.append(legal_moves)
    
    return moves, legal_moves_data


def load_data(csv_path, test_size=0.1, random_state=42, frac=0.4):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    sequences = df['sequence'].apply(json.loads).tolist()
    targets = df['target'].apply(json.loads).tolist()

    seq_train, seq_val, tar_train, tar_val = train_test_split(
        sequences, targets, test_size=test_size, random_state=random_state
    )

    return seq_train, seq_val, tar_train, tar_val

def find_max_length(sequences):
    return max(len(seq) for seq in sequences)


def get_samples(dataloader: DataLoader, samples):

    data_list = []
    label_list = []
    batch_size = dataloader.batch_size

    for i, (data, labels) in enumerate(dataloader):
        if i * batch_size >= samples:
            break
        data_list.append(data)
        label_list.append(labels)
    if len(data_list) == 0:
        print("empty samples!!!")

    merged_data = torch.cat(data_list, dim=0)
    merged_labels = torch.cat(label_list, dim=0)

    return merged_data, merged_labels

def get_valid_accs(output, sample_games_data, tar_vocab):
    correct, legals = sample_games_data
    reverse_tar = {item: key for key, item in tar_vocab.items()}
    output = output[:, 1:-1]
    invalids = 0
    output_moves = []
    for pred in output:
        try:
            move = translate_move(pred, reverse_tar)
            output_moves.append(move)
        except Exception:
            invalids += 1
            output_moves.append(None)

    acc_list = [
        pred.from_square == tar.from_square and pred.to_square == tar.to_square
        if pred is not None else False
        for pred, tar in zip(output_moves, correct)
    ]
    legals_acc = [
        pred in legal_list
        if pred is not None else False
        for pred, legal_list in zip(output_moves, legals)
    ]
    return sum(acc_list) / len(acc_list), sum(legals_acc) / len(legals_acc)

if __name__ == "__main__":
    vocab_path = "src/lstm/fen_vocab_src.json"
    tar_vocab_path = "src/lstm/vocab_tar.json"
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    with open(tar_vocab_path, "r") as f:
        tar_vocab = json.load(f)

    csv_path = 'data/prep/tokenized_ones.csv'
    src_vocab_len = len(vocab)
    tar_vocab_len = len(tar_vocab)

    batch=512
    num_workers=32
    frac = 0.01
    valid_size = 0.2

    seq_train, seq_val, tar_train, tar_val = load_data(csv_path, frac=frac, test_size=valid_size)

    max_src_len = max(find_max_length(seq_train), find_max_length(seq_val))
    max_tar_len = max(find_max_length(tar_train), find_max_length(tar_val))

    dataset_train = SequenceDataset(seq_train, tar_train, src_vocab_len, tar_vocab_len, max_src_len, max_tar_len)
    dataset_val = SequenceDataset(seq_val, tar_val, src_vocab_len, tar_vocab_len, max_src_len, max_tar_len)

    dataloader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)

    eps= 30
    lr = 0.001
    device = torch.device("cuda")
    embed = 512
    hidden = 512
    layers = 4
    drop = 0.2

    print(f"train len: {len(dataloader_train)}")
    print(f"val len: {len(dataloader_val)}\n")

    model = Seq2Seq(src_vocab_len, tar_vocab_len, embed, hidden, layers, drop).to(device)


    merged_data, merged_targs = get_samples(dataloader=dataloader_val,samples=10240)
    sample_games_data = translate_data(merged_data, merged_targs, vocab, tar_vocab)



    criterion = nn.CrossEntropyLoss()
    opti = optim.Adam(model.parameters(), lr=lr)

    model.load_state_dict(torch.load("transformer13.pth", weights_only=True))
    model.eval()
    merged_data, merged_targs = merged_data.to("cuda"), merged_targs.to("cuda")


    seq = merged_data.permute(1, 0)
    tar = merged_targs.permute(1, 0)
    output = model(seq, tar, 0)
    print(output.shape)
    output = output.permute(1, 0, 2)
    output = F.softmax(output, -1)
    output = torch.argmax(output, -1)
    print(output.shape)

    acc, corr = get_valid_accs(output, sample_games_data, tar_vocab)

    print(acc, corr)