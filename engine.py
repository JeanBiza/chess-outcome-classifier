import chess
import numpy as np
import onnxruntime as rt
import torch
import torch.nn.functional as F

from tensor import board_to_tensor

TOP_K = 10

def load_session(model_path="chess_model.onnx"):
    return rt.InferenceSession(model_path)


def _softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def evaluate_batch(sess, boards):
    boards_np = []
    extras_np = []

    for board in boards:
        planes, extra = board_to_tensor(board)
        boards_np.append(planes)
        extras_np.append(extra)

    logits = sess.run(["logits"], {
        "board": np.array(boards_np, dtype=np.float32),
        "extra": np.array(extras_np, dtype=np.float32),
    })[0]

    probs = _softmax(logits)
    return (probs[:, 0] - probs[:, 1]).tolist()


def evaluate(sess, board):
    return evaluate_batch(sess, [board])[0]


def _get_children(board):
    children = []
    for move in board.legal_moves:
        board.push(move)
        children.append((move, board.copy()))
        board.pop()
    return children


def _rank_moves(sess, children, maximizing, top_k):
    child_boards = [b for _, b in children]
    scores = evaluate_batch(sess, child_boards)

    ranked = sorted(
        zip(children, scores),
        key=lambda x: x[1],
        reverse=maximizing
    )

    return [(move, board, score) for (move, board), score in ranked[:top_k]]


def minimax(sess, board, depth, alpha, beta, maximizing):
    if board.is_game_over():
        return evaluate(sess, board)

    children = _get_children(board)

    if depth == 1:
        scores = evaluate_batch(sess, [b for _, b in children])
        return max(scores) if maximizing else min(scores)

    ranked = _rank_moves(sess, children, maximizing, TOP_K)

    if maximizing:
        best = float('-inf')
        for _, child_board, _ in ranked:
            score = minimax(sess, child_board, depth - 1, alpha, beta, False)
            best = max(best, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return best
    else:
        best = float('inf')
        for _, child_board, _ in ranked:
            score = minimax(sess, child_board, depth - 1, alpha, beta, True)
            best = min(best, score)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return best


def best_move(sess, board, depth=3):
    is_white = board.turn == chess.WHITE
    best_score = float('-inf') if is_white else float('inf')
    best_mv = None

    children = _get_children(board)

    if depth == 1:
        scores = evaluate_batch(sess, [b for _, b in children])
        for (move, _), score in zip(children, scores):
            if is_white and score > best_score:
                best_score = score
                best_mv = move
            elif not is_white and score < best_score:
                best_score = score
                best_mv = move
        return best_mv

    ranked = _rank_moves(sess, children, is_white, TOP_K)

    for move, child_board, _ in ranked:
        score = minimax(sess, child_board, depth - 1,
                        float('-inf'), float('inf'),
                        child_board.turn == chess.WHITE)

        if is_white and score > best_score:
            best_score = score
            best_mv = move
        elif not is_white and score < best_score:
            best_score = score
            best_mv = move

    return best_mv