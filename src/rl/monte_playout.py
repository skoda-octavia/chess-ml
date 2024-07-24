from gameEnv import Game
from model import rl
from monte import monte_carlo_value
import chess
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn


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

        try:
            move = move_queue.get(timeout=1)
        except Exception:
            return
        next_tensor = game.simulate_move(move)
        next_game = Game.from_tensor(next_tensor)
        next_game.board.turn = not game.board.turn
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
        proc_num=28
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

    dropout = 0.1
    playouts = 20
    lr = 0.001
    max_pieces = 14
    game_timeout = 50
    eps = 30



    model = rl(6*8*8, 1, [384, 400, 300, 200, 100, 50], dropout)
    model.load_state_dict(torch.load('model_weights99.pth'))
    model.share_memory()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"device: {device}")
    model = model.to(device)

    fens = []

    local_filename = "fens.txt"
    with open(local_filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if ',' not in line and '-' in line and '/' in line:
                board = chess.Board(line)
                if len(board.piece_map()) <= max_pieces:
                    fens.append(line)
    print(f"fens in ep: {len(fens)}")

    for i in range(eps):
        moves = []
        for fen in fens:
            # print(fen)
            moves = []
            board = chess.Board(fen)
            game = Game.from_board(board, False)
            # print(game.board)

            while not game.board.is_game_over():

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
                game.make_move(next_move)
                moves.append(next_move)
                # print(next_move)
                # print(board)
            game.over()
            # print(game.score())
            # print(moves)
            # print("---------------------------------------")
            moves.append(len(moves))
        torch.save(model.state_dict(), f"mwpuzzle/model_weights{i}.pth")
        print(f"eps {i}, moves_to mate: {sum(moves)/len(moves)}")

if __name__ == '__main__':
    mp.set_start_method('forkserver', force=True)
    main()