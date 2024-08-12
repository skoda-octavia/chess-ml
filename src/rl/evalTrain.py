import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import chess
from game import Game
from monte import monte_carlo_value, playout_value
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

        res = []
        for _ in range(games_played):
            res.append(playout_value(game, model, optimizer, lock, device, [], [], game_timeout, exploration, True))
        # res = monte_carlo_value(game, games_played, model, optimizer, lock, device, game_timeout, exploration)
        # mates = sum([abs(r) for r in res])
        # print(f"finished fen, mates: {mates}, remaining: {fen_queue.qsize()}")
        results.append(sum([re == 0 or re == 1 for re in res])/games_played)


def main():
    model = rl(6*8*8, 1, [384, 512, 1024, 2048, 4096, 4096, 2048, 1024, 512, 256, 128, 64])
    load_num = 0
    if load_num != 0:
        model.load_state_dict(torch.load(f'models/rlEval/model_weights{5}.pth', weights_only=True))
    model.share_memory()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"device: {device}")
    model = model.to(device)

    # url = 'https://wtharvey.com/'
    # filenames = ['m8n2.txt', 'm8n3.txt', 'm8n4.txt']
    # for local_filename in filenames:
    #     temp_url = url + local_filename
    #     response = requests.get(temp_url)
    #     response.raise_for_status()

    #     with open(local_filename, 'wb') as file:
    #         file.write(response.content)

    fens = []
    max_pieces = 40
    eps = 500
    games_played = 30
    game_timeout = 100
    exploration = 1
    num_processes = 18

    local_filenames = ["m8n2.txt", "m8n3.txt", "m8n4.txt"]
    white_cnt = 0
    black_cnt = 0
    for local_filename in local_filenames:
        with open(local_filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if ',' not in line and '-' in line and '/' in line:
                    board = chess.Board(line)
                    default_turn = board.turn
                    if len(board.piece_map()) <= max_pieces:
                        next_line = line
                        if white_cnt > black_cnt and board.turn:
                            game = Game.from_board(board, transform=True)
                            next_line = game.board.fen()
                            default_turn = False
                        if default_turn:
                            white_cnt += 1
                        else:
                            black_cnt += 1
                        fens.append(next_line)

                        
    boards = [chess.Board(fen) for fen in fens]
    print(f"white puzzle: {sum([board.turn for board in boards])}")
    print(f"black puzzle: {sum([not board.turn for board in boards])}")
    fens_num = len(fens)
    print(f"num of all examples: {fens_num}")

    print(f"proc num: {num_processes}")

    #debug
    # with mp.Manager() as manager:
    #     fen_queue = manager.Queue()
    #     results = manager.list()
    #     lock = mp.RLock()
    #     for fen in fens:
    #         fen_queue.put(fen)
    #     worker(
    #         fen_queue,
    #         games_played,
    #         model,
    #         optimizer,
    #         results,
    #         lock,
    #         device,
    #         game_timeout,
    #         exploration
    #     )

        # eval_model = deepcopy(model)
        # print(id(eval_model))
        # print(id(model))
        # evaluate_model(
        #     monte_carlo_move_function,
        #     20,
        #     eval_model,
        #     optimizer,
        #     lock,
        #     device,
        #     100,
        #     1,
        #     1,
        #     20      
        # )

    for i in range(eps):
        with mp.Manager() as manager:
            fen_queue = manager.Queue()
            results = manager.list()
            lock = mp.RLock()

            for fen in fens:
                fen_queue.put(fen)

            processes = []

            # eval_model = deepcopy(model)
            
            # eval_proc = mp.Process(
            #     target=evaluate_model,
            #     args=(
            #         monte_carlo_move_function,
            #         10,
            #         eval_model,
            #         optimizer,
            #         lock,
            #         device,
            #         100,
            #         1,
            #         i,
            #         10
            #         )
            #     )
            # eval_proc.start()
            # processes.append(eval_proc)

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
            print(f"eps: {load_num+eps+1}, mates found : {mean*100}%")

        torch.save(model.state_dict(), f"models/rlEval/model_weights{load_num+eps+1}.pth")

if __name__ == '__main__':
    mp.set_start_method('forkserver', force=True)
    main()
