import torch.optim as optim
import random
from game import Game
import torch


def record(game, score, model, optimalizer, lock):
    model.fit(game.state(), score, optimalizer, lock)

def heuristic_value(tensor, model):
    return model.predict(tensor)

def get_move(action_dict: dict, white_moves):
    const_bias = 0.05
    if len(action_dict) == 1:
        return list(action_dict.keys())[0]
    moves = list(action_dict.keys())
    values = list(action_dict.values())
    if not white_moves:
        values = [-val for val in values]
    bias = min(values)
    if bias < 0:
        values = [(val+abs(bias)+const_bias) for val in values]
    try:
        choice = random.choices(values, weights=values, k=1)
    except ValueError:
        print(values)
        print(white_moves)
        print(action_dict)
        raise ValueError
    return moves[values.index(choice[0])]


def playout_value(game: Game, model, optimalizer, lock):
    if game.over():
        score = torch.tensor(game.score())
        record(game, score, model, optimalizer, lock)
        if score.item() != 0:
            print(game.board)
            print(f"--------------: {score.item()}")
        return torch.tensor(score.item())

    action_heuristic_dict = {}
    for move in game.valid_moves():
        tempTensor = game.simulate_move(move)
        heu = heuristic_value(tempTensor, model).item()
        action_heuristic_dict[move] = heu
    move = get_move(action_heuristic_dict, game.board.turn)

    next_game = game.copy()
    next_game.make_move(move)

    value = playout_value(next_game, model, optimalizer, lock)
    record(game, value, model, optimalizer, lock)

    return value

def monte_carlo_value(game, N, model, optimalizer, lock):
    res = []
    for _ in range(N):
        res.append(playout_value(game, model, optimalizer, lock))
    return res
