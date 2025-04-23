import chess
import numpy as np

NUM_PLANES = 17
BOARD_SIZE = 8

def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = _piece_to_plane(piece)
        row, col = divmod(square, 8)
        tensor[plane, row, col] = 1.0
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    for i, flag in enumerate(rights, start=13):
        tensor[i, :, :] = 1.0 if flag else 0.0
    tensor[16, :, :] = board.halfmove_clock / 100.0
    return tensor

def _piece_to_plane(piece: chess.Piece) -> int:
    base = piece.piece_type - 1
    if piece.color == chess.BLACK:
        base += 6
    return base
