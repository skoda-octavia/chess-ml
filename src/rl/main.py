import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import chess
from game import Game
from monte import monte_carlo_value
import requests

class rl(nn.Module):
    def __init__(self, input_size: int, output_size: int, layer_sizes: list[int], dropout: float=0.1):
        super(rl, self).__init__()
        layers = []
        flat = nn.Flatten(start_dim=1)
        layers.append(flat)
        old_size = input_size
        for layer in layer_sizes:
            layers.append(nn.Linear(old_size, layer))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Tanh())
            old_size = layer

        layers.append(nn.Linear(old_size, output_size))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        return self.model(x)

    def fit(self, tensor, score, optimizer, lock):
        score = torch.tensor([[score]], dtype=torch.float32)
        out = self.forward(tensor)
        loss = self.criterion(out, score)
        with lock:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


    def predict(self, tensor):
        return self.forward(tensor)

def worker(fen_queue, max_pieces, games_played, model, optimizer, results, lock):
    while not fen_queue.empty():
        fen = fen_queue.get()
        transform = False
        try:
            board = chess.Board(fen)
        except ValueError:
            print(f"Invalid fen: {fen}")
            continue


        game = Game.from_board(board, transform)
        res = monte_carlo_value(game, games_played, model, optimizer, lock)
        mates = sum([abs(r) for r in res])
        print(f"finished fen, mates: {mates}, remaining: {len(fen_queue)}")
        results.append(mates)


def main():
    model = rl(6*8*8, 1, [384, 400, 300, 200, 100, 50])
    # model.load_state_dict(torch.load('model_weights3.pth'))
    model.share_memory()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    url = 'https://wtharvey.com/'
    filenames = ['m8n2.txt', 'm8n3.txt', 'm8n4.txt']
    for local_filename in filenames:
        temp_url = url + local_filename
        response = requests.get(temp_url)
        response.raise_for_status()

        with open(local_filename, 'wb') as file:
            file.write(response.content)

    fens = []
    max_pieces = 12
    eps = 1
    games_played = 150

    local_filenames = ["m8n2.txt", "m8n3.txt", "m8n4.txt"]
    for local_filename in local_filenames:
        with open(local_filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if ',' not in line and '-' in line and '/' in line:
                    board = chess.Board(line)
                    if len(board.piece_map()) <= max_pieces:
                        fens.append(line)

    fens_num = len(fens)
    print(f"num of all examples: {fens_num}")

    num_processes = mp.cpu_count()
    print(f"proc num: {num_processes}")

    for i in range(eps):
        with mp.Manager() as manager:
            fen_queue = manager.Queue()
            results = manager.list()
            lock = mp.RLock()

            for fen in fens:
                fen_queue.put(fen)

            processes = []
            for _ in range(num_processes):
                p = mp.Process(target=worker, args=(fen_queue, max_pieces, games_played, model, optimizer, results, lock))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            mean = sum(results) / len(results)
            print(f"eps: {i+4}, res mean: {mean}")

        torch.save(model.state_dict(), f"model_weights{i+4}.pth")

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
