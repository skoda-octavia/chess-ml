import chess
import chess.pgn
import torch

def board_to_matrix(board: chess.Board, move: chess.Move):
    matrix_board = torch.zeros((6, 8, 8))
    matrix_move_from = torch.zeros((8, 8))
    matrix_move_to = torch.zeros((8, 8))

    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                piece_type = piece.piece_type
                piece_color = piece.color
                index = piece_type - 1
                y_idx = 7 - j if board.turn == chess.WHITE else j
                # x_idx = i if board.turn == chess.WHITE else 7 - i
                if piece_color == chess.WHITE:
                    matrix_board[index, y_idx, i] = 1
                else:
                    matrix_board[index, y_idx, i] = -1
    
                # print(matrix_board)
                    
    file_from = chess.square_file(move.from_square)
    rank_from = chess.square_rank(move.from_square)
    file_to = chess.square_file(move.to_square)
    rank_to = chess.square_rank(move.to_square)
    matrix_move_from[7-rank_from][file_from] = 1
    matrix_move_to[7-rank_to][file_to] = 1

    if board.turn == chess.BLACK:
        matrix_board *= -1
        matrix_move_from = torch.flip(matrix_move_from, dims=[0])
        matrix_move_to = torch.flip(matrix_move_to, dims=[0])

    matrix_move = torch.cat((matrix_board, matrix_move_from.unsqueeze(0)), 0),
    return matrix_board, matrix_move_from, matrix_move[0], matrix_move_to, board.piece_at(move.from_square).piece_type

def main():
    pgn = open("full.pgn")
    cnt = 0
    board_data, piece_from_data = [], []
    piece_moves_dict = {
        chess.PAWN : [[], []],
        chess.KNIGHT : [[], []],
        chess.BISHOP : [[], []],
        chess.ROOK : [[], []],
        chess.QUEEN : [[], []],
        chess.KING : [[], []]
    }

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        cnt+=1
        if cnt % 500 == 0:
            print(cnt)

        board = game.board()
        for move in game.mainline_moves():
            board_matrix, piece_from_matrix, matrix_move, matrix_move_to, piece_type  = board_to_matrix(board, move)
            piece_moves_dict[piece_type][0].append(matrix_move.tolist())
            piece_moves_dict[piece_type][1].append(matrix_move_to.tolist())
            board_data.append(board_matrix.tolist())
            piece_from_data.append(piece_from_matrix.tolist())
            board.push(move)


    X = torch.tensor(board_data)
    y = torch.tensor(piece_from_data)
    for piece, list in piece_moves_dict.items():
        tensor_data = torch.tensor(list[0])
        tensor_y = torch.tensor(list[1])
        torch.save(tensor_data, f"X_{chess.piece_name(piece)}.pt")
        torch.save(tensor_y, f"Y_{chess.piece_name(piece)}.pt")
    torch.save(X, "X_all.pt")
    torch.save(y, "y_all.pt")

main()