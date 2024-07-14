import chess
import chess.engine
from stockfish import Stockfish
import random

stockfish_path = r""

def update_elo(current_elo, opponent_elo, result, k=32):
    expected_score = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
    new_elo = current_elo + k * (result - expected_score)
    return new_elo

def evaluate_model(model_move_function, games=100, initial_elo=600):
    stockfish = Stockfish(stockfish_path)
    stockfish.set_elo_rating(initial_elo)
    model_elo = initial_elo

    for _ in range(games):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = model_move_function(board)
                board.push(move)
            else:
                stockfish.set_fen_position(board.fen())
                move = chess.Move.from_uci(stockfish.get_best_move())
                board.push(move)

        result = board.result()
        if result == "1-0":
            model_result = 1
        elif result == "0-1":
            model_result = 0
        else:
            model_result = 0.5

        model_elo = update_elo(model_elo, model_elo, model_result)
        stockfish_elo = update_elo(model_elo, model_elo, 1 - model_result)
        stockfish.set_elo_rating(int(stockfish_elo))

    return model_elo

def example_model_move_function(board):
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)

elo_rating = evaluate_model(example_model_move_function, games=20)
print(elo_rating)