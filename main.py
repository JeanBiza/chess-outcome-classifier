import chess.pgn
import chess
import chess.engine
import numpy as np
import glob
import random
from dotenv import load_dotenv
import os

load_dotenv()

INPUT_DIR = "input"
OUTPUT_DATA_PATH = "chess_data.npz"

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH")
EVAL_DEPTH = 8

LABEL_MAP = {
    "1-0": 0,       # White Wins
    "0-1": 1,       # Black Wins
    "1/2-1/2": 2    # Draw
}

PIECE_MAP = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

def get_stockfish_label(board, engine):
    info = engine.analyse(board, chess.engine.Limit(depth=EVAL_DEPTH))
    score = info["score"].white().score(mate_score=10000)
    
    if score is None:
        return None
    
    if score > 150:
        return 0
    elif score < -150:
        return 1
    else:
        return 2  

def board_to_tensor(board):
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane = PIECE_MAP[piece.piece_type] + color_offset
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            planes[plane][rank][file] = 1.0
        
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

def build_dataset():
    X_boards = []
    X_extra = []
    y = []

    if not os.path.exists(INPUT_DIR):
        print(f"Creating file directory...")
        os.mkdir(INPUT_DIR)
        print(f"Please add pgn files in '{INPUT_DIR}'")
        return
    
    pgn_files = glob.glob(os.path.join(INPUT_DIR, "*.pgn"))

    if not pgn_files:
        print(f"No .pgn files found in the '{INPUT_DIR}' directory.")
        return

    print(f"Found {len(pgn_files)} PGN file(s). Starting extraction...\n")

    total_games_processed = 0

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
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

                    variant = game.headers.get("Variant", "Standard")
                    if variant.lower() not in ("standard", "from position", ""):
                        continue

                    fen = game.headers.get("FEN", chess.STARTING_FEN)
                    if fen != chess.STARTING_FEN and game.headers.get("SetUp", "0") == "1":
                        continue

                    moves = list(game.mainline_moves())
                    if len(moves) < 15:
                        continue

                    sample_at = random.randint(10, len(moves) - 1)

                    board = game.board()
                    for i, move in enumerate(moves):
                        board.push(move)
                        if i == sample_at:
                            break

                    # ✅ Label de Stockfish en vez del resultado final
                    label = get_stockfish_label(board, engine)
                    if label is None:
                        continue

                    planes, extra = board_to_tensor(board)

                    X_boards.append(planes)
                    X_extra.append(extra)
                    y.append(label)

                    file_games_processed += 1
                    total_games_processed += 1

                    if file_games_processed % 1000 == 0:
                        print(f"  {file_games_processed} games processed...")

            print(f"Finished {os.path.basename(file_path)}: {file_games_processed} games\n")

    print(f"Extraction complete! Total: {total_games_processed}")

    X_boards_np = np.array(X_boards, dtype=np.float32)
    X_extra_np  = np.array(X_extra,  dtype=np.float32)
    y_np        = np.array(y,        dtype=np.int64)

    print(f"X_boards shape: {X_boards_np.shape}")
    print(f"X_extra shape:  {X_extra_np.shape}")
    print(f"y shape:        {y_np.shape}")

    np.savez_compressed(OUTPUT_DATA_PATH, boards=X_boards_np, extra=X_extra_np, labels=y_np)
    print(f"Saved to {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    build_dataset()