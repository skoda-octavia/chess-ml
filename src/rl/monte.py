import torch.optim as optim
import random
from game import Game
import torch
import chess


def record(game, score, model, optimalizer, lock):
    model.fit(game.state(), score, optimalizer, lock)

def heuristic_value(tensor, model):
    return model.predict(tensor)

def get_move(heus: torch.Tensor, moves: list[chess.Move], white_moves, to_consider=3):
    const_bias = 0.05
    heus = torch.squeeze(heus)
    if len(moves) == 1:
        return moves[0]
    if not white_moves:
        heus = heus * -1
    bias = torch.min(heus)
    if bias < 0:
        heus -= bias
    topk_values, topk_indices = torch.topk(heus, to_consider)
    probabilities = topk_values / torch.sum(topk_values)

    chosen_index = torch.multinomial(probabilities, 1)
    selected_index = topk_indices[chosen_index]
    return moves[selected_index.item()]


def playout_value(game: Game, model, optimalizer, lock, device):
    if game.over():
        print(game.board)
        print(f"---------------{game.score()}")
        score = torch.tensor(game.score())
        record(game, score, model, optimalizer, lock)
        return torch.tensor(score.item())

    next_states = []
    moves = []
    for move in game.valid_moves():
        moves.append(move)
        tempTensor = game.simulate_move(move)
        next_states.append(tempTensor)

    final = torch.stack(next_states).to(device)
    heus = model.predict(final)
    move = get_move(heus, moves, game.board.turn)

    next_game = game.copy()
    next_game.make_move(move)

    value = playout_value(next_game, model, optimalizer, lock, device)
    record(game, value, model, optimalizer, lock)

    return value

def monte_carlo_value(game, N, model, optimalizer, lock, device):
    res = []
    for _ in range(N):
        res.append(playout_value(game, model, optimalizer, lock, device))
    return res
