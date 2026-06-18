import chess.pgn
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np

PGN_FILE_PATH = os.getenv("PGN_FILE_PATH")
OUTPUT_DATA_PATH = "chess_data.npz"
MAX_GAMES_TO_PROCESS = 5

LABEL_MAP = {
    "1-0": 0,       # White Wins
    "0-1": 1,       # Black Wins
    "1/2-1/2": 2    # Draw
}

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN : 0,
        chess.KNIGHT : 1,
        chess.BISHOP : 2,
        chess.ROOK : 3,
        chess.QUEEN : 4,
        chess.KING : 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane = piece_map[piece.piece_type] + color_offset

            rank = chess.square_rank(square)
            file = chess.square_file(square)

            tensor[plane][rank][file] = 1.0
        
    return tensor.flatten()

def build_dataset():
    X = []
    y = []

    games_processed = 0

    print(f"Starting data extraction from: {PGN_FILE_PATH}")

    with open(PGN_FILE_PATH, "r", encoding="utf-8") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            game_result = game.headers.get("Result", "*")

            if game_result not in LABEL_MAP:
                continue

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            
            board_tensor = board_to_tensor(board)
            numeric_label = LABEL_MAP[game_result]

            X.append(board_tensor)
            y.append(numeric_label)

            games_processed += 1

            if games_processed % 1000 == 0:
                print(f"Processed {games_processed} games...")
            
    print("\nExtraction complete, Converting to numpy arrays...")

    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.int64)

    print(f"Features shape (X): {X_np.shape}")
    print(f"Labels shape (y): {y_np.shape}")

    np.savez_compressed(OUTPUT_DATA_PATH, features=X_np, labels=y_np)
    print(f"Dataset successfully saved to {OUTPUT_DATA_PATH}")



if __name__ == "__main__":
    build_dataset()