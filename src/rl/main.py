import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import chess
from src.rl.gameEnv import Game
from monte import monte_carlo_value, evaluate_model, monte_carlo_move_function
import requests
from model import rl
from copy import deepcopy


def worker(
        fen_queue,
        games_played,
        model,
        optimizer,
        results,
        lock,
        device,
        game_timeout,
        exploration
    ):
    while not fen_queue.empty():
        fen = fen_queue.get()
        # print(f"{fen_queue.qsize()} remain")
        transform = False
        try:
            board = chess.Board(fen)
        except ValueError:
            print(f"Invalid fen: {fen}")
            continue

            
        game = Game.from_board(board, transform)
        
        for move in game.valid_moves():
            tempTensor = game.simulate_move(move)
            temp_game = Game.from_tensor(tempTensor)
            temp_game.board.turn = not game.board.turn
            res = monte_carlo_value(temp_game, games_played, model, optimizer, lock, device, game_timeout, exploration)
            score = sum([abs(r) for r in res])
        
        # print(f"finished fen, mates: {mates}, remaining: {fen_queue.qsize()}")
        results.append(score)


def main():
    model = rl(6*8*8, 1, [384, 400, 300, 200, 100, 50], 0.3)
    model.load_state_dict(torch.load('model_weights99.pth'))
    model.share_memory()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"device: {device}")
    model = model.to(device)

    url = 'https://wtharvey.com/'
    filenames = ['m8n2.txt', 'm8n3.txt', 'm8n4.txt']
    for local_filename in filenames:
        temp_url = url + local_filename
        response = requests.get(temp_url)
        response.raise_for_status()

        with open(local_filename, 'wb') as file:
            file.write(response.content)

    fens = []
    max_pieces = 20
    eps = 50
    games_played = 10
    game_timeout = 100
    exploration = 1

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

    # num_processes = int(mp.cpu_count() /2)
    num_processes = 24
    print(f"proc num: {num_processes}")

    #debug
    # with mp.Manager() as manager:
    #     fen_queue = manager.Queue()
    #     results = manager.list()
    #     lock = mp.RLock()
    #     for fen in fens:
    #         fen_queue.put(fen)
    #     worker(
    #                                 fen_queue,
    #                     games_played,
    #                     model,
    #                     optimizer,
    #                     results,
    #                     lock,
    #                     device,
    #                     game_timeout,
    #                     exploration
    #     )

    for i in range(eps):
        with mp.Manager() as manager:
            fen_queue = manager.Queue()
            results = manager.list()
            lock = mp.RLock()

            for fen in fens:
                fen_queue.put(fen)

            processes = []

            for _ in range(num_processes):
                p = mp.Process(
                    target=worker,
                    args=(
                        fen_queue,
                        games_played,
                        model,
                        optimizer,
                        results,
                        lock,
                        device,
                        game_timeout,
                        exploration
                        )
                    )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            mean = sum(results) / len(results)
            print(f"eps: {i+100}, res mean: {mean}")

        torch.save(model.state_dict(), f"model_weights{i+100}.pth")

if __name__ == '__main__':
    mp.set_start_method('forkserver', force=True)
    main()
