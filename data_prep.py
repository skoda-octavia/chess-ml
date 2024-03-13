import chess.pgn
import torch
import torch.nn as nn

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim

def board_to_matrix(board: chess.Board, move: chess.Move):
    # Inicjalizacja macierzy 8x8x6
    matrix_board = torch.zeros((8, 8, 6))
    matrix_move = torch.zeros((8, 8))

    # Indeksowanie od 1 do 8 dla planszy
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                # Kodowanie typu i koloru figury
                piece_type = piece.piece_type
                piece_color = piece.color
                index = piece_type - 1
                if piece_color == chess.WHITE:
                    matrix_board[i, j, index] = 1
                else:
                    matrix_board[i, j, index] = -1
    if board.turn == chess.BLACK:
        matrix_board *= -1

    file = chess.square_file(move.from_square) #col
    rank = chess.square_rank(move.from_square)
    matrix_move[7-rank][file] = 1
    return matrix_board, matrix_move

def main():
    # Wczytaj plik PGN
    pgn = open("example.pgn")
    cnt = 0
    matrices_board, matrices_move = [], []
    # Przetwórz każdą partię w pliku PGN
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        cnt+=1
        if cnt % 100 == 0:
            print(cnt)
        # Utwórz listę tensorów dla partii
        board = game.board()
        for move in game.mainline_moves():
            matrix_board, matrix_move = board_to_matrix(board, move)
            matrices_board.append(matrix_board)
            matrices_move.append(matrix_move)
            board.push(move)

    # Zapisz listę tensorów do plików
    torch.save(matrices_board, "matrices_board.pt")
    torch.save(matrices_move, "matrices_move.pt")

def test():
    X = torch.load("matrices_board.pt")
    y = torch.load("matrices_move.pt")

    # Upewnienie się, że X i y mają ten sam kształt
    assert len(X) == len(y), "Długość danych wejściowych i etykiet musi być taka sama"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sprawdzenie kształtów danych wejściowych i etykiet
    print("X_train shape:", X_train[0].shape)  # Sprawdzamy kształt jednego z tensorów w liście
    print("y_train shape:", y_train[0].shape)  # Sprawdzamy kształt jednego z tensorów w liście

    # Konwersja listy tensorów na pojedynczy tensor i przekształcenie kształtu
    X_train = torch.stack([x.permute(2, 0, 1) for x in X_train]).to(device)  # Zamiana kolejności wymiarów
    y_train = torch.stack([y.unsqueeze(0) for y in y_train]).to(device)  # Dodanie dodatkowego wymiaru

    # Utworzenie i trenowanie modelu MLP
    model = nn.Sequential(
        nn.Conv2d(6, 64, kernel_size=3, padding=1),  # Warstwa konwolucyjna
        nn.ReLU(),               # Funkcja aktywacji ReLU
        nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Druga warstwa konwolucyjna
        nn.ReLU(),               # Funkcja aktywacji ReLU
        nn.Flatten(),            # Spłaszczenie danych
        nn.Linear(128*8*8, 8*8), # Warstwa w pełni połączona
    )
    model.to(device)

    # Definicja funkcji straty i optymalizatora
    criterion = nn.CrossEntropyLoss()  # Założenie, że używamy Cross Entropy Loss dla problemu klasyfikacji wieloklasowej
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Wybór optymalizatora i współczynnika uczenia

    # Pętla trenowania
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in zip(X_train, y_train):
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(0))  # Dodanie wymiaru batch_size
            loss = criterion(outputs, labels.view(-1))  # Dostosowanie kształtu danych wejściowych i etykiet
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(X_train)}")



if __name__ == "__main__":
    # main()
    test()
