# Chess Outcome Classifier

A full-stack chess AI project that evaluates positions and suggests moves using a CNN trained on Stockfish evaluations. Play against the model, analyze positions, or use it as a move advisor.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-green)
![React](https://img.shields.io/badge/React-19-61dafb)

## Features

- **Position evaluator** — neural network predicts win probability for white/black after each move
- **Move advisor** — finds the best move using minimax with alpha-beta pruning
- **Play vs AI** — play as white or black against the model
- **Move history** — full game record in algebraic notation with scroll
- **Export** — copy PGN or FEN at end of game
- **Board controls** — flip board, undo moves, reset game

## Architecture

```
Board (14×8×8 planes)               Extra features (7)
  12 planes: piece positions              turn to move
   2 planes: attack density               castling rights ×4
             (white/black)                en passant square
                  │                       halfmove clock
             Conv2d 14→32                      │
             Conv2d 32→64                      │
             Conv2d 64→128                     │
                  │                            │
             Flatten (8192) ──── Cat ──────────┘
                                  │
                            Linear 8199→512
                            Linear 512→128
                            Linear 128→3
                                  │
                   [White win, Black win, Draw]
```

**Labels:** Stockfish depth 12 evaluation
- White advantage (score > +1.5 pawns) → White wins
- Black advantage (score < -1.5 pawns) → Black wins
- Equal position → Draw

**Search:** Minimax + Alpha-Beta pruning with ONNX inference for speed
- Move ordering with top-k=10 candidates per node
- depth=2 → ~0.6s per move
- depth=3 → ~7s per move

## Model

| Detail | Value |
|--------|-------|
| Dataset | ~435k positions |
| Label source | Stockfish depth 12 |
| Test accuracy | **78.72%** |
| Input | 14×8×8 planes + 7 extra features |
| Architecture | CNN + MLP fusion |

A pre-trained model is available in the [releases](../../releases) — download `best_chess_model.pt`, `chess_model.onnx`, `chess_model.onnx.data` and place them in the project root.

## Setup

### Backend

```bash
git clone https://github.com/JeanBiza/chess-outcome-classifier
cd chess-outcome-classifier

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Stockfish is downloaded automatically on first run via `setup_stockfish.py` (only needed for dataset generation, not for the API).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Rebuild the dataset

Add `.pgn` files to the `input/` directory (broadcasts from [database.lichess.org](https://database.lichess.org) work well), then:

```bash
python main.py   # extract positions + Stockfish labels
python train.py  # train the CNN
python export_onnx.py  # export to ONNX for fast inference
```

## Project structure

```
chess-outcome-classifier/
├── api.py             # FastAPI endpoints (/predict, /best_move)
├── engine.py          # Minimax + Alpha-Beta + ONNX inference
├── tensor.py          # board_to_tensor (shared by all modules)
├── model.py           # CNN architecture
├── train.py           # training loop with early stopping
├── main.py            # PGN → dataset with Stockfish labels
├── export_onnx.py     # export PyTorch model to ONNX
├── setup_stockfish.py # auto-download Stockfish binary
├── requirements.txt
└── frontend/
    └── src/
        └── App.jsx    # React UI
```