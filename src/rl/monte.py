import torch.optim as optim
import random
from game import Game
import torch
import chess
import gc


def record(game, score, model, optimalizer, lock, pos_stack: list, device):
    positions = torch.stack(pos_stack)
    scores = torch.full((len(positions),), score)
    scores = scores.float()
    scores, positions = scores.to(device), positions.to(device)
    model.fit(positions, scores, optimalizer, lock)

def heuristic_value(tensor, model):
    return model.predict(tensor)

def get_move(heus_gpu: torch.Tensor, moves: list[chess.Move], white_moves, to_consider=3):
    heus = heus_gpu.detach().cpu()
    del heus_gpu
    const_bias = 0.05
    heus = torch.squeeze(heus)
    if len(moves) == 1:
        return moves[0]
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
    return moves[selected_index.item()]


def playout_value(game: Game, model, optimalizer, lock, device, pos_stack):
    pos_stack.append(game.tensor)
    if game.over():
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
    move = get_move(heus, moves, game.board.turn)

    next_game = game.copy()
    next_game.make_move(move)

    value = playout_value(next_game, model, optimalizer, lock, device, pos_stack)
    if len(game.board.move_stack) == 0:
        record(game, value, model, optimalizer, lock, pos_stack, device)

    return value

def monte_carlo_value(game, N, model, optimalizer, lock, device):
    res = []
    for _ in range(N):
        res.append(playout_value(game, model, optimalizer, lock, device, []))
    gc.collect()
    torch.cuda.empty_cache()
    return res
