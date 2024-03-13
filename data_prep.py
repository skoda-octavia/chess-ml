import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def board_to_matrix(board: chess.Board, move: chess.Move):
    matrix_board = torch.zeros((6, 8, 8))
    matrix_move = torch.zeros((8, 8))

    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                piece_type = piece.piece_type
                piece_color = piece.color
                index = piece_type - 1
                if piece_color == chess.WHITE:
                    matrix_board[index, 7-j, i] = 1
                else:
                    matrix_board[index, 7-j, i] = -1
                # print(matrix_board)
    if board.turn == chess.BLACK:
        matrix_board *= -1

    file = chess.square_file(move.from_square)
    rank = chess.square_rank(move.from_square)
    matrix_move[7-rank][file] = 1
    # print(matrix_board)
    return matrix_board.tolist(), matrix_move.tolist()

def main():
    pgn = open("full.pgn")
    cnt = 0
    board_matrix, piece_matrix = [], []

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        cnt+=1
        if cnt % 100 == 0:
            print(cnt)
            
        board = game.board()
        for move in game.mainline_moves():
            if board.turn == chess.WHITE:
                matrix_board, matrix_move = board_to_matrix(board, move)
                board_matrix.append(matrix_board)
                piece_matrix.append(matrix_move)
            board.push(move)


    X = torch.tensor(board_matrix)
    y = torch.tensor(piece_matrix)
    torch.save(X, "X.pt")
    torch.save(y, "y.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


def test():
    X = torch.load("X.pt")
    y = torch.load("y.pt")
    assert len(X) == len(y), "Długość danych wejściowych i etykiet musi być taka sama"

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(8*8*6, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8*8),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(device)
    X.to(device)
    y.to(device)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 1000
    eps = []
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X.view(-1, 8*8*6))
        loss = criterion(output, y.view(-1, 8*8))
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            eps.append(epoch)
            losses.append(loss.item())

    plt.plot(eps, losses)
    plt.xlabel("eps")
    plt.ylabel("loss")
    plt.show()


if __name__ == "__main__":
    # main()
    test()
