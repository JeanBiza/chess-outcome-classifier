import chess.pgn as chess
import os
from dotenv import load_dotenv
load_dotenv()

PGN_FILE_PATH = os.getenv("PGN_FILE_PATH")

def test_read_game():
    with open(PGN_FILE_PATH, "r", encoding="utf-8") as pgn:
        first_game = chess.read_game(pgn)

        if first_game is None:
            print("No se encontraron partidas.")
            return
        
        white_player = first_game.headers.get("White", "Unknown")
        black_player = first_game.headers.get("Black", "Unknown")
        result = first_game.headers.get("Result", '*')

        print(f"Partida leida con exito!")
        print(f"White: {white_player}")
        print(f"Black: {black_player}")
        print(f"Resultado: {result}")

if __name__ == "__main__":
    test_read_game()