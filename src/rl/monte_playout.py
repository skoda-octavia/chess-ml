from gameEnv import Game
from model import rl
from monte import monte_carlo_value
import chess
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn


def print_(str, filename):
    with open(filename, "a") as f:
        f.write(str+"\n")


def worker(
        game: Game,
        move: chess.Move,
        model: nn.Module,
        optimizer,
        lock,
        device,
        playouts,
        game_timeout,
        res_queue,
        move_queue: mp.Queue
    ):
        while not move_queue.empty():
            try:
                move = move_queue.get(timeout=1)
            except Exception:
                return
            # print(move_queue.qsize(), move)
            next_game = game.simulate_move(move)
            res = monte_carlo_value(
                next_game,
                playouts,
                model,
                optimizer,
                lock,
                device,
                game_timeout,
                1,
                True
            )
            eval = move, res
            res_queue.put(eval)

def get_monte_values(
        game: Game,
        model: nn.Module,
        optimizer,
        device,
        playouts,
        game_timeout,
        proc_num=32
        ) -> tuple[list[chess.Move], list[float]]:
    
    
    legal_moves = list(game.board.legal_moves)
    move_eval = []
    with mp.Manager() as manager:
        res_queue = manager.Queue()
        move_queue = manager.Queue()
        lock = manager.RLock()
        processes = []

        for move in legal_moves:
            move_queue.put(move)

        # print(move_queue.qsize())

        for _ in range(proc_num):
            p = mp.Process(
                target=worker,
                args=(
                    game,
                    move,
                    model,
                    optimizer,
                    lock,
                    device,
                    playouts,
                    game_timeout,
                    res_queue,
                    move_queue
                    )
                )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        while not res_queue.empty():
            move_eval.append(res_queue.get())
    
    return move_eval

        


def main():

    dropout = 0.05
    playouts = 30
    lr = 0.001
    max_pieces = 50
    game_timeout = 100
    eps = 300
    load_num = 8
    puzzle_timeout = 10

    model = rl(6*8*8, 1, [384, 400, 500, 500, 400, 300, 200, 100, 64], dropout)

    if load_num != 0:
        model_name = f'models/rlEval/model_weights{load_num}.pth'
        print(f"loading model: {model_name}")
        model.load_state_dict(torch.load(model_name, weights_only=True))
    else:
        print("new weights")
    model.share_memory()
    
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"device: {device}")
    model = model.to(device)

    fens = []

    white_cnt = 0
    bl_cnt = 0
    local_filename = "fens.txt"
    with open(local_filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if ',' not in line and '-' in line and '/' in line:
                board = chess.Board(line)
                if len(board.piece_map()) <= max_pieces:
                    fens.append(line)
                    if board.turn:
                        white_cnt += 1
                    else:
                        bl_cnt += 1
    
    print(f"fens in ep: {len(fens)}")
    print(f"white puzzles: {white_cnt}")
    print(f"black puzzles: {bl_cnt}")

    for i in range(eps):
        scores = []
        for idx, fen in enumerate(fens):
            # print(fen)
            moves = []
            print_(fen, "playout_rap.txt")
            board = chess.Board(fen)
            game = Game.from_board(board, False)
            print(game.board)
            cnt = 0

            while cnt != puzzle_timeout and not game.board.is_game_over():

                moves_eval = get_monte_values(
                    game,
                    model,
                    optimizer,
                    device,
                    playouts,
                    game_timeout,
                )
                if game.board.turn:
                    next_move = max(moves_eval, key=lambda x: x[1])[0]
                else:
                    next_move = min(moves_eval, key=lambda x: x[1])[0]
                game = game.make_move(next_move)
                board = game.board
                moves.append(next_move)
                print(next_move)
                print(board)
                cnt += 1
            game.over()
            score = game.score()
            print_(score)
            print(moves)
            print_(str(moves), "playout_rap.txt")
            print("---------------------------------------")
            print_("---------------------------------------", "playout_rap.txt")
            scores.append(score)
            torch.save(model.state_dict(), f"models/playoutTrain/model_weights_{i}_{idx}.pth")
        print_(f"eps {i}")
        print(f"eps {i}")

if __name__ == '__main__':
    mp.set_start_method('forkserver', force=True)
    main()