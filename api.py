from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import numpy as np
import chess

from model import ChessOutcomeCNN

app = FastAPI(title="Chess AI Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIECE_MAP = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

device = torch.device("cpu")
model = ChessOutcomeCNN()
model.load_state_dict(torch.load("best_chess_model.pt", map_location=device))
model.eval()

class ChessPosition(BaseModel):
    fen: str

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

@app.post("/predict")
def predict_outcome(position: ChessPosition):
    try:
        board = chess.Board(position.fen)

        planes, extra = board_to_tensor(board)

        board_tensor = torch.tensor(planes, dtype=torch.float32).unsqueeze(0).to(device)
        extra_tensor = torch.tensor(extra,  dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(board_tensor, extra_tensor)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()

        return {
            "white_win_probability": float(probs[0] * 100),
            "black_win_probability": float(probs[1] * 100),
            "draw_probability":      float(probs[2] * 100)
        }

    except ValueError:
        return {"error": "Invalid FEN string provided."}