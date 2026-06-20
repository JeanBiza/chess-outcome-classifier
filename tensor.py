import chess
import numpy as np

PIECE_MAP = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

def board_to_tensor(board):
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane = PIECE_MAP[piece.piece_type] + color_offset
            planes[plane][chess.square_rank(square)][chess.square_file(square)] = 1.0

    extra = np.array([
        1.0 if board.turn == chess.WHITE else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
        board.ep_square / 63.0 if board.ep_square is not None else 0.0,
        board.halfmove_clock / 100.0,
    ], dtype=np.float32)

    return planes, extra
