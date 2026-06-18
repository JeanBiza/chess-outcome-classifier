import chess.pgn as chess
import os
from dotenv import load_dotenv
load_dotenv()

PGN_FILE_PATH = os.getenv("PGN_FILE_PATH")
MAX_GAMES_TO_PROCESS = 5

def extract_games():
    games_processed = 0
    with open(PGN_FILE_PATH, "r", encoding="utf-8") as pgn:
        while True:
            game = chess.read_game(pgn)

            if game is None:
                break

            if games_processed >= MAX_GAMES_TO_PROCESS:
                break

            game_result = game.headers.get("Result", "*")

            if game_result not in ["1-0", "0-1", "1/2-1/2"]:
                continue

            print(f"\n--- Processing Game {games_processed+1} ---")
            print(f"Target Label: {game_result}")

            board = game.board()

            move_count = 0
            for move in game.mainline_moves():
                board.push(move)
                move_count += 1
            
            print(f"Total half-moves (ply) played: {move_count}")

            print("Final board state:")
            print(board)

            games_processed += 1
    
    print(f"\nFinished processing {games_processed} games successfully.")

if __name__ == "__main__":
    extract_games()