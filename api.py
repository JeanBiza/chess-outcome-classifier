from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import numpy as np
import chess
from tensor import board_to_tensor
from engine import best_move      

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
    
@app.post("/best_move")
def get_best_move(position: ChessPosition):
    try:
        board = chess.Board(position.fen)
        move = best_move(model, board, depth=3, device=device)
        if move is None:
            return {"error": "No legal moves available"}
        return {"best_move": move.uci()}  # ej: "e2e4"
    except ValueError:
        return {"error": "Invalid FEN string provided."}