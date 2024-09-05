import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import random
from seqModules import Seq2Seq
import chess

class LstmService():

    def __init__(self) -> None:
            
        vocab_path = "src/lstm64/vocab_src.json"
        tar_vocab_path = "src/lstm64/vocab_tar.json"
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        with open(tar_vocab_path, "r") as f:
            self.tar_vocab = json.load(f)
            self.reverse_tar_vocab = {v: k for k, v in self.tar_vocab.items()}
        self.src_pad = len(self.vocab)
        self.tar_pad = len(self.tar_vocab)

        self.src_vocab_len = self.src_pad + 1
        self.tar_vocab_len = self.tar_pad + 1

        device = torch.device("cuda")
        embed = 512
        hidden = 512
        layers = 4
        drop = 0

        self.model = Seq2Seq(
            self.src_vocab_len,
            self.tar_vocab_len,
            embed,
            hidden,
            layers,
            drop,
            self.src_pad,
            self.tar_pad
            ).to(device)
        self.model.eval()
        self.model.load_state_dict(torch.load("transformer15.pth", weights_only=True))

    def tokenize_fen(self, fen: str):
        tokens = [self.vocab['SOS']]
        fen_splited = fen.split(" ")
        assert len(fen_splited) == 6
        board, turn_str, cas, en_pass, _, _ = fen_splited 

        nums_to_change = [8, 7, 6, 5, 4, 3, 2]
        for num in nums_to_change:
            board = board.replace(str(num), num * "1")
        for char in board:
            tokens.append(self.vocab[char])

        turn = "True" if turn_str == "w" else "False"
        tokens.append(self.vocab[turn])

        if cas != "-":
            cas = cas.replace("K", "Ki")
            cas = cas.replace("Q", "Qu")
            cas = cas.replace("k", "ki")
            cas = cas.replace("q", "qu")
            elems = [cas[i:i+2] for i in range(0, len(cas), 2)]
            tokens.extend([self.vocab[cr] for cr in elems])

        if en_pass != "-":
            en = en_pass[0] + "_enpas"
            tokens.append(self.vocab[en])
        tokens.append(self.vocab["EOS"])

        return tokens
    
    def create_tar_mask(self, fen: str):
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        move_tokens = []
        for move in legal_moves:
            from_, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
            temp = [self.tar_vocab[from_], self.tar_vocab[to]]
            prom = self.tar_vocab[move.uci()[4]] if move.promotion is not None else self.tar_vocab["EOS"]
            temp.append(prom)
            move_tokens.append(temp)
        return move_tokens
    

    def __call__(self, fen: str):
        tokenized_fen = self.tokenize_fen(fen)
        tar_mask = self.create_tar_mask(fen)
        tensor = torch.tensor(tar_mask)
        max_idx = self.tar_pad
        one_hot_tensor = torch.zeros(tensor.size(0), tensor.size(1), max_idx + 1)
        one_hot_tensor.scatter_(2, tensor.unsqueeze(-1), 1)

        
        fake_output = torch.tensor([0, 38, 30, 1])
        fake_output = torch.unsqueeze(fake_output, 0)
        fake_output = fake_output.permute(1, 0).to("cuda")
        input = torch.tensor(tokenized_fen)
        input = torch.unsqueeze(input, 0)
        input = input.permute(1, 0).to("cuda")
        one_hot_tensor = one_hot_tensor.to("cuda")
        output = self.model(input, fake_output, 1, one_hot_tensor)
        output_class = output.permute(1, 0, 2)
        output_moves = torch.argmax(output_class, dim=2)[:, 1:]
        tokens = torch.squeeze(output_moves).tolist()

        eos_token = self.tar_vocab["EOS"]
        move_symbols = [self.reverse_tar_vocab[token] for token in tokens if token != eos_token]

        print(move_symbols)


serv = LstmService()
serv("K7/4r3/7R/1r2k3/8/8/8/8 w - - 0 1")