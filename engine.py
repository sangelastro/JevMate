"""Il computer sceglie la mossa con OpenJev (cross-encoder NLI).

Per ogni mossa legale si scrive un'ipotesi in inglese ("The best move for Black is ...") arricchita da fatti
calcolati con python-chess (cosa cattura, se il pezzo resta in presa, se da' scacco, se lascia pezzi scoperti).
OpenJev legge la posizione come premessa e da' a ogni ipotesi una probabilita' di entailment: vince l'argmax.
Nessun motore scacchistico decide al posto suo: i fatti sono descrittivi, la scelta e' del modello.
"""
from __future__ import annotations

import os
import sys
import time

import chess
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models", "openjev")
sys.path.insert(0, MODELS)
from modeling_openjev import ENT, OpenJevCrossEncoder  # noqa: E402

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
COLOR = {chess.WHITE: "White", chess.BLACK: "Black"}


def _name(piece: chess.Piece) -> str:
    return chess.piece_name(piece.piece_type)


def _sq(sq: int) -> str:
    return chess.square_name(sq)


def _material(board: chess.Board, color: bool) -> int:
    return sum(VALUE[p.piece_type] for p in board.piece_map().values() if p.color == color)


def _en_prise(board: chess.Board, sq: int) -> bool:
    """Pezzo attaccato e non difeso, oppure attaccato da un pezzo che vale meno di lui."""
    piece = board.piece_at(sq)
    if piece is None or piece.piece_type == chess.KING:
        return False
    attackers = board.attackers(not piece.color, sq)
    if not attackers:
        return False
    if not board.attackers(piece.color, sq):
        return True
    cheapest = min(VALUE[board.piece_at(a).piece_type] for a in attackers)
    return cheapest < VALUE[piece.piece_type]


def _hanging(board: chess.Board, color: bool) -> list[int]:
    return [sq for sq, p in board.piece_map().items() if p.color == color and _en_prise(board, sq)]


def describe_position(board: chess.Board) -> str:
    me, opp = board.turn, not board.turn
    lines = [
        f"Chess position (FEN {board.fen()}). {COLOR[me]} is to move.",
        f"Material: {COLOR[me]} {_material(board, me)}, {COLOR[opp]} {_material(board, opp)} "
        "(pawn 1, knight 3, bishop 3, rook 5, queen 9).",
    ]
    if board.move_stack:
        last = board.pop()
        san = board.san(last)
        board.push(last)
        lines.append(f"{COLOR[opp]} just played {san}.")
    if board.is_check():
        lines.append(f"{COLOR[me]} is in check and must get out of check.")
    mine = _hanging(board, me)
    if mine:
        lines.append(f"{COLOR[me]} pieces under threat: "
                     + ", ".join(f"{_name(board.piece_at(s))} on {_sq(s)}" for s in mine) + ".")
    theirs = _hanging(board, opp)
    if theirs:
        lines.append(f"{COLOR[opp]} pieces that can be won: "
                     + ", ".join(f"{_name(board.piece_at(s))} on {_sq(s)}" for s in theirs) + ".")
    lines.append("A good move wins material, avoids losing pieces, keeps the king safe and improves the position.")
    return " ".join(lines)


def describe_move(board: chess.Board, move: chess.Move) -> str:
    me = board.turn
    piece = board.piece_at(move.from_square)
    san = board.san(move)
    facts = [f"The best move for {COLOR[me]} is {san}:"]

    if board.is_castling(move):
        facts.append("castling, which puts the king in safety.")
    else:
        facts.append(f"the {_name(piece)} moves from {_sq(move.from_square)} to {_sq(move.to_square)}.")

    gain = 0
    if board.is_capture(move):
        victim = chess.PAWN if board.is_en_passant(move) else board.piece_at(move.to_square).piece_type
        gain = VALUE[victim]
        facts.append(f"It captures a {chess.piece_name(victim)} (worth {gain}).")
    if move.promotion:
        facts.append(f"It promotes the pawn to a {chess.piece_name(move.promotion)}.")

    if board.fullmove_number <= 12 and not board.is_castling(move):
        back_rank = 0 if me == chess.WHITE else 7
        file_to = chess.square_file(move.to_square)
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(move.from_square) == back_rank:
            facts.append(f"It develops the {_name(piece)} in the opening.")
        elif piece.piece_type == chess.PAWN and file_to in (3, 4):
            facts.append("It takes space in the centre.")
        elif piece.piece_type == chess.PAWN and file_to in (0, 7):
            facts.append("It is a slow edge pawn move that does not develop.")
        elif piece.piece_type == chess.QUEEN:
            facts.append("It brings the queen out early.")
        elif piece.piece_type in (chess.KING, chess.ROOK):
            facts.append(f"It moves the {_name(piece)} and gives up the right to castle.") if board.has_castling_rights(me) else None

    threatened_before = set(_hanging(board, me))
    board.push(move)
    try:
        if board.is_checkmate():
            facts.append("It is checkmate and wins the game immediately.")
            return " ".join(facts)
        if board.is_stalemate():
            facts.append("It causes stalemate, so the game ends in a draw.")
        elif board.is_check():
            facts.append("It gives check.")

        lands_hanging = _en_prise(board, move.to_square)
        if lands_hanging:
            moved = board.piece_at(move.to_square)
            facts.append(f"After it the {_name(moved)} on {_sq(move.to_square)} can be captured "
                         f"(it would lose {VALUE[moved.piece_type]}).")
        elif move.from_square in threatened_before:
            facts.append(f"It saves the threatened {_name(piece)}.")

        left = [s for s in _hanging(board, me) if s != move.to_square]
        if left:
            facts.append("It leaves undefended: "
                         + ", ".join(f"{_name(board.piece_at(s))} on {_sq(s)}" for s in left) + ".")
        targets = _hanging(board, not me)
        if targets:
            facts.append("It attacks: "
                         + ", ".join(f"{_name(board.piece_at(s))} on {_sq(s)}" for s in targets) + ".")
        if not lands_hanging and not left and gain == 0 and not board.is_check():
            facts.append("It is a quiet, safe move.")
    finally:
        board.pop()
    return " ".join(facts)


class OpenJevPlayer:
    def __init__(self, subfolder: str | None = None):
        cuda = torch.cuda.is_available()
        self.device = "cuda" if cuda else "cpu"
        # Il 2B pesa 4,4 GB in bf16: serve memoria libera anche per attivazioni e cuBLAS
        free_gb = torch.cuda.mem_get_info()[0] / 1e9 if cuda else 0
        self.subfolder = subfolder or os.environ.get("OPENJEV_MODEL") or (
            "qwen3.5-2b-nli-v5" if free_gb > 6.5 else "qwen3.5-0.8b-nli-v2s-long")
        dtype = torch.bfloat16 if cuda else torch.float32
        t = time.time()
        self.jev = OpenJevCrossEncoder(MODELS, subfolder=self.subfolder, device=self.device, dtype=dtype)
        self.load_s = time.time() - t

    def choose(self, board: chess.Board, chunk: int = 24) -> dict:
        moves = list(board.legal_moves)
        premise = describe_position(board)
        hyps = [describe_move(board, m) for m in moves]
        t = time.time()
        probs = []
        for i in range(0, len(hyps), chunk):
            probs.append(self.jev.predict_hypotheses(premise, hyps[i:i + chunk]))
        p = np.concatenate(probs, 0)
        ms = int((time.time() - t) * 1000)

        ent = p[:, ENT]
        share = ent / ent.sum() if ent.sum() > 0 else np.full(len(ent), 1 / len(ent))
        order = np.argsort(-ent)
        best = moves[int(order[0])]
        return {
            "move": best,
            "ms": ms,
            "premise": premise,
            "candidates": [{
                "uci": moves[i].uci(),
                "san": board.san(moves[i]),
                "entailment": round(float(ent[i]), 4),
                "share": round(float(share[i]), 4),
                "reason": hyps[i],
            } for i in order[:8]],
            "n_moves": len(moves),
        }
