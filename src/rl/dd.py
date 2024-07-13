import chess
from matplotlib import pyplot as plt

fens = []
local_filenames = [
    "m8n2.txt",
    "m8n3.txt",
    "m8n4.txt",
]
for local_filename in local_filenames:
    with open(local_filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if ',' not in line and '-' in line and '/' in line:
                fens.append(line)

fen_cnt = []
fen_white = []
fen_black = []

for i in range(32):
    cnt = 0
    whi = 0
    bl = 0
    for fen in fens:
        board = chess.Board(fen)
        if len(board.piece_map()) == i + 1:
            cnt += 1
            if board.turn:
                whi += 1
            else:
                bl += 1
    fen_cnt.append(cnt)
    fen_white.append(whi)
    fen_black.append(bl)

x = [i for i in range(1, 33)]
plt.plot(x, fen_cnt, label="puzzle_cnt")
plt.plot(x, fen_white, label="white_puzzle_cnt")
plt.plot(x, fen_black, label="black_puzzle_cnt")
plt.legend()
plt.title("puzzle count per piece num")
plt.xlabel("piece num")
plt.ylabel("puzzle cnt")
plt.show()

print(f"{cnt}/{len(fens)}")