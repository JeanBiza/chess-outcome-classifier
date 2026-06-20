from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chess

from tensor import board_to_tensor
from engine import load_session, best_move, evaluate

app = FastAPI(title="Chess AI Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sess = load_session("chess_model.onnx")


class ChessPosition(BaseModel):
    fen: str


class MoveRequest(BaseModel):
    fen: str
    depth: int = 2


@app.post("/predict")
def predict_outcome(position: ChessPosition):
    try:
        board = chess.Board(position.fen)
        score = evaluate(sess, board)
        white = (score + 1) / 2 * 100
        black = 100 - white
        return {
            "white_win_probability": round(white, 2),
            "black_win_probability": round(black, 2),
            "draw_probability": 0.0,
        }
    except ValueError:
        return {"error": "Invalid FEN string provided."}


@app.post("/best_move")
def get_best_move(request: MoveRequest):
    try:
        board = chess.Board(request.fen)
        if request.depth not in (1, 2, 3):
            return {"error": "depth must be 1, 2 or 3"}
        mv = best_move(sess, board, depth=request.depth)
        if mv is None:
            return {"error": "No legal moves"}
        return {"best_move": mv.uci()}
    except ValueError:
        return {"error": "Invalid FEN string provided."}