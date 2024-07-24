import chess
import random
from gameEnv import Game

white_boards = []
black_boards = []

local_filenames = ["m8n2.txt", "m8n3.txt", "m8n4.txt"]
for local_filename in local_filenames:
    with open(local_filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if ',' not in line and '-' in line and '/' in line:
                board = chess.Board(line)
                if board.turn:
                    white_boards.append(board)
                else:
                    black_boards.append(board)

bigger = white_boards if len(white_boards) > len(black_boards) else black_boards 
smaller = black_boards if len(white_boards) > len(black_boards) else white_boards 

diff = abs(len(white_boards) - len(black_boards))
to_transform = int(diff/2)
random.shuffle(bigger)

for i in range(to_transform):
    elem = bigger.pop()
    transformed_game = Game.from_board(elem, True)
    smaller.append(transformed_game.board)



print(len(bigger))
print(len(smaller))

bigger.extend(smaller)
random.shuffle(bigger)

with open("fens.txt", 'w') as file:
    for board in bigger:
        fen = board.fen()
        file.write(fen + '\n')