import chess.pgn
import os
import numpy as np
import glob

INPUT_DIR = "input"
OUTPUT_DATA_PATH = "chess_data.npz"

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
        
    flattened_board = tensor.flatten()
    turn_indicator = np.array([1.0 if board.turn == chess.WHITE else 0.0], dtype=np.float32)
    final_tensor = np.concatenate((flattened_board, turn_indicator))
    return final_tensor

def build_dataset():
    X = []
    y = []

    if not os.path.exists(INPUT_DIR):
        print(f"Creating file directory...")
        os.mkdir(INPUT_DIR)
        print(f"Please add pgn files in '{INPUT_DIR}'")
        return
    
    pgn_files = glob.glob(os.path.join(INPUT_DIR, "*.pgn"))

    if not pgn_files:
        print(f"No .pgn files found in the '{INPUT_DIR}' directory.")

    print(f"Found {len(pgn_files)} PGN file(s). Starting extraction...\n")

    total_games_processed = 0

    for file_path in pgn_files:
        print(f"--- Processing File: {os.path.basename(file_path)} ---")
        file_games_processed = 0

        with open(file_path, "r", encoding="utf-8") as pgn_file:
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

                file_games_processed += 1
                total_games_processed += 1

                if file_games_processed % 5000 == 0:
                    print(f"Extracted {file_games_processed} games from current file...")

        print(f"Finished {os.path.basename(file_path)}. Total valid games: {file_games_processed}\n")
            
    print(f"Extraction complete! Total games across all files: {total_games_processed}")

    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.int64)

    print(f"Features shape (X): {X_np.shape}")
    print(f"Labels shape (y): {y_np.shape}")

    np.savez_compressed(OUTPUT_DATA_PATH, features=X_np, labels=y_np)
    print(f"Dataset successfully saved to {OUTPUT_DATA_PATH}")



if __name__ == "__main__":
    build_dataset()