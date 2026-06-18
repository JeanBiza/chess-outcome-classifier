import chess.pgn
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np

PGN_FILE_PATH = os.getenv("PGN_FILE_PATH")
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

def extract_games():
    games_processed = 0
    with open(PGN_FILE_PATH, "r", encoding="utf-8") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)

            if game is None:
                break

            if games_processed >= MAX_GAMES_TO_PROCESS:
                break

            game_result = game.headers.get("Result", "*")

            if game_result not in ["1-0", "0-1", "1/2-1/2"]:
                continue

            print(f"\n--- Processing Game {games_processed+1} ---")

            board = game.board()

            for move in game.mainline_moves():
                board.push(move)
            
            board_tensor = board_to_tensor(board)
            
            print(f"Target Label: {game_result}")
            print(f"Tensor Shape: {board_tensor.shape}")
            print(f"First 10 tensor values: {board_tensor[:10]}")   

            games_processed += 1
    
    print(f"\nFinished processing {games_processed} games successfully.")

if __name__ == "__main__":
    extract_games()