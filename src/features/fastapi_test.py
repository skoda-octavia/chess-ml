import requests
import chess
import chess.pgn
from datetime import datetime

def print_(str):
    print(str)
    with open("play_rap.txt", "a") as f:
        f.write(str + "\n")

minimax_url = "http://localhost:8000/get-move/Minimax"
stockfish_url = "http://localhost:8000/get-move/Stockfish"
max_moves = 100
num_games = 10

start_positions = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
]

def get_move_from_api(api_url, fen):
    try:
        response = requests.get(f"{api_url}?fen={fen}")
        response.raise_for_status()
        data = response.json()
        return data.get("move"), data.get("eval")
    except Exception as e:
        print_(f"Error: {e}")
        return None, None

def play_game(start_fen, white_url, black_url):
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    node = game
    eval = 32

    move_count = 0
    while not board.is_game_over() and move_count < max_moves and eval >= 0 and eval <=  63:
        fen = board.fen()

        current_player = white_url if board.turn == chess.WHITE else black_url

        move, eval = get_move_from_api(current_player, fen)
        if move is None:
            print_("Nie udało się pobrać ruchu. Koniec gry.")
            break
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")

        print_(f"Wykonany ruch: {move} ({'Minimax' if current_player == minimax_url else 'Stockfish'}), Eval: {eval}, {current_time}")
        try:
            chess_move = board.push_uci(move)
            node = node.add_variation(chess_move)
        except ValueError:
            print_(f"Nieprawidłowy ruch: {move}. Koniec gry.")
            break

        move_count += 1
        if move_count % 10 == 0:
            print(str(game))

    print_("\nKoniec gry!")
    print_(str(board))
    print_(f"PGN:\n{str(game)}")

if __name__ == "__main__":
    for game_nr in range(10):
        for i, start_fen in enumerate(start_positions):
            if game_nr % 2 == 0:
                white_url = minimax_url
                black_url = stockfish_url
            else:
                white_url = stockfish_url
                black_url = minimax_url

            print_(f"\n=== Partia {game_nr+1} z pozycji startowej: {start_fen} ===")
            print_(f"Minimax gra jako {'Białe' if white_url == minimax_url else 'Czarne'}")
            print_(f"Stockfish gra jako {'Białe' if white_url == stockfish_url else 'Czarne'}")
            play_game(start_fen, white_url, black_url)
