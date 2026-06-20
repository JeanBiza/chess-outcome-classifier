import chess
import torch
import torch.nn.functional as F
import numpy as np

from tensor import board_to_tensor


def evaluate_batch(model, boards, device):
    boards_np = []
    extras_np = []

    for board in boards:
        planes, extra = board_to_tensor(board)
        boards_np.append(planes)
        extras_np.append(extra)

    bt = torch.tensor(np.array(boards_np), dtype=torch.float32).to(device)
    et = torch.tensor(np.array(extras_np), dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(bt, et)
        probs = F.softmax(logits, dim=1)

    scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()
    return scores.tolist()


def evaluate(model, board, device):
    return evaluate_batch(model, [board], device)[0]


def minimax(model, board, depth, alpha, beta, maximizing, device):
    if board.is_game_over():
        return evaluate(model, board, device)

    if depth == 1:
        child_boards = []
        moves = list(board.legal_moves)

        for move in moves:
            board.push(move)
            child_boards.append(board.copy())
            board.pop()

        scores = evaluate_batch(model, child_boards, device)

        if maximizing:
            return max(scores)
        else:
            return min(scores)

    if maximizing:
        best = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(model, board, depth - 1, alpha, beta, False, device)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return best
    else:
        best = float('inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(model, board, depth - 1, alpha, beta, True, device)
            board.pop()
            best = min(best, score)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return best


def best_move(model, board, depth=3, device='cpu'):
    is_white = board.turn == chess.WHITE
    best_score = float('-inf') if is_white else float('inf')
    best_mv = None

    moves = list(board.legal_moves)

    child_boards = []
    for move in moves:
        board.push(move)
        child_boards.append(board.copy())
        board.pop()

    if depth == 1:
        scores = evaluate_batch(model, child_boards, device)
        for move, score in zip(moves, scores):
            if is_white and score > best_score:
                best_score = score
                best_mv = move
            elif not is_white and score < best_score:
                best_score = score
                best_mv = move
        return best_mv

    for move, child_board in zip(moves, child_boards):
        score = minimax(
            model, child_board, depth - 1,
            float('-inf'), float('inf'),
            child_board.turn == chess.WHITE,
            device
        )

        if is_white and score > best_score:
            best_score = score
            best_mv = move
        elif not is_white and score < best_score:
            best_score = score
            best_mv = move

    return best_mv