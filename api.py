from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import numpy as np
import chess

from model import ChessOutcomeClassifier

app = FastAPI(title="Chess AI Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu") 
model = ChessOutcomeClassifier()
model.load_state_dict(torch.load("best_chess_model.pt", map_location=device))

model.eval() 

class ChessPosition(BaseModel):
    fen: str

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
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
    
    return np.concatenate((flattened_board, turn_indicator))

@app.post("/predict")
def predict_outcome(position: ChessPosition):
    try:
        board = chess.Board(position.fen)
        
        tensor = board_to_tensor(board)
        
        input_tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            raw_logits = model(input_tensor)
            
            probabilities = F.softmax(raw_logits, dim=1).squeeze().numpy()
            
        return {
            "white_win_probability": float(probabilities[0] * 100),
            "black_win_probability": float(probabilities[1] * 100),
            "draw_probability": float(probabilities[2] * 100)
        }
        
    except ValueError:
        return {"error": "Invalid FEN string provided."}