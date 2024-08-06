import torch.optim as optim
import random
from gameEnv import Game
import torch
import chess
import gc
# from stockfish import Stockfish


def record(score, model, optimalizer, lock, pos_stack: list, device, evals):
    winning_bias = 0.3
    positions = torch.stack(pos_stack)
    score = max(min(score, 1), 0)
    if score == 0:
        evals.append(0.5 - winning_bias)
        starting_val = min(evals)
        scores = torch.linspace(starting_val, 0, len(pos_stack))
    elif score == 1:
        evals.append(0.5 + winning_bias)
        starting_val = max(evals)
        scores = torch.linspace(starting_val, 1, len(pos_stack))
    else:
        scores = torch.full((len(positions),), score)
    scores = scores.float()
    scores, positions = scores.to(device), positions.to(device)
    model.fit(positions, scores, optimalizer, lock)
    del scores
    del positions

def heuristic_value(tensor, model):
    return model.predict(tensor)

def get_move(heus_gpu: torch.Tensor, moves: list[chess.Move], white_moves, to_consider=3):
    heus = heus_gpu.detach().cpu()
    del heus_gpu
    if to_consider == 1 and white_moves:
        best = torch.argmax(heus).item()
        return moves[best], torch.max(heus).item()
    elif to_consider == 1 and not white_moves:
        best = torch.argmin(heus).item()
        return moves[best], torch.min(heus).item()
    const_bias = 0.05
    heus = torch.squeeze(heus)
    if len(moves) == 1:
        return moves[0], heus[0].item()
    if not white_moves:
        heus = heus * -1
    bias = torch.min(heus)
    if bias < 0:
        heus -= bias
    to_consider = min(to_consider, len(heus))
    topk_values, topk_indices = torch.topk(heus, to_consider)
    topk_values += const_bias
    probabilities = topk_values / torch.sum(topk_values)

    chosen_index = torch.multinomial(probabilities, 1)
    selected_index = topk_indices[chosen_index]
    return moves[selected_index.item()], heus[selected_index.item()]


def playout_value(
        game: Game,
        model,
        optimalizer,
        lock,
        device,
        pos_stack,
        evals,
        game_timeout,
        exploration,
        update = True,
    ):
    pos_stack.append(game.tensor)
    if game.over(game_timeout):
        # print(len(game.board.move_stack))
        # print(game.board)
        # print(f"---------------: {game.score()}")
        return game.score()

    next_states = []
    moves = []
    for move in game.valid_moves():
        moves.append(move)
        tempTensor = game.simulate_move(move)
        next_states.append(tempTensor)

    final = torch.stack(next_states).to(device)
    heus = model.predict(final)
    del final
    move, eval = get_move(heus, moves, game.board.turn, exploration)
    evals.append(eval)
    del heus
    next_game = game.copy()
    next_game.make_move(move)

    value = playout_value(next_game, model, optimalizer, lock, device, pos_stack, evals, game_timeout, exploration)
    if len(game.board.move_stack) == 0 and update:
        record(value, model, optimalizer, lock, pos_stack, device, evals)

    gc.collect()
    torch.cuda.empty_cache()
    return value

def monte_carlo_value(
        game,
        N,
        model,
        optimalizer,
        lock,
        device,
        game_timeout,
        exploration,
        update=True
        ):
    res = []
    for _ in range(N):
        res.append(playout_value(game, model, optimalizer, lock, device, [], game_timeout, exploration, update))
    return sum(res)/len(res)


stockfish_path = r"src/rl/stockfish-ubuntu-x86-64-avx2"

def update_elo(current_elo, opponent_elo, result, k=32):
    expected_score = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
    new_elo = current_elo + k * (result - expected_score)
    return new_elo

# def evaluate_model(
#         model_move_function,
#         games_played,
#         model,
#         optimizer,
#         lock,
#         device,
#         game_timeout,
#         exploration,
#         eps,
#         games=100,
#         initial_elo=600
#         ):
#     stockfish = Stockfish(stockfish_path)
#     stockfish.set_elo_rating(initial_elo)
#     model_elo = initial_elo

#     for i in range(games):
#         board = chess.Board()
#         game = Game.from_board(board, False)
#         while not board.is_game_over():
#             if board.turn == chess.WHITE:
#                 move = model_move_function(
#                     game,
#                     games_played,
#                     model,
#                     optimizer,
#                     lock,
#                     device,
#                     game_timeout,
#                     exploration,
#                 )
#             else:
#                 stockfish.set_fen_position(board.fen())
#                 move = chess.Move.from_uci(stockfish.get_best_move())
#             game.make_move(move)

#         result = board.result()
#         if result == "1-0":
#             model_result = 1
#         elif result == "0-1":
#             model_result = 0
#         else:
#             model_result = 0.5

#         model_elo = update_elo(model_elo, model_elo, model_result)
#         stockfish.set_elo_rating(int(model_elo))
#         print(f"Game {i} run, res: {result}")
#     print(f"elo for eps {eps}: {model_elo}")

#     return model_elo

def monte_carlo_move_function(
        game: Game,
        games_played,
        model,
        optimizer,
        lock,
        device,
        game_timeout,
        exploration
    ):
    legal_moves = list(game.board.legal_moves)
    values = []
    for move in legal_moves:
        temp_game = game.copy()
        temp_game.make_move(move)
        results = monte_carlo_value(
            temp_game,
            games_played,
            model,
            optimizer,
            lock,
            device,
            game_timeout,
            exploration,
            update=False
            )
        move_val = sum(results) / len(results)
        values.append(move_val)
    
    if game.board.turn:
        idx = torch.argmax(torch.tensor(values)).item()
    else:
        idx = torch.argmin(torch.tensor(values)).item()
    return legal_moves[idx]