"""Il computer sceglie la mossa con OpenJev (cross-encoder NLI).

Per ogni mossa legale si scrive un'ipotesi in inglese ("The best move for Black is ...") arricchita da fatti
calcolati con python-chess: bilancio degli scambi, scacco, pezzi lasciati in presa, minacce, sviluppo, pedoni
passati. OpenJev legge la posizione e il criterio della fase di gioco come premessa e da' a ogni ipotesi una
probabilita' di entailment: vince l'argmax.
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

# Criterio di "buona mossa" per fase di gioco. Due formulazioni da confrontare con bench.py.
RUBRICS = {
    "A": {
        "opening": "This is the opening. A good move develops knights and bishops, fights for the centre, castles "
                   "early, does not bring the queen out early and never loses material.",
        "middlegame": "This is the middlegame. A good move wins material, never loses material, keeps the king safe "
                      "and attacks the opponent's pieces.",
        "endgame": "This is the endgame. A good move wins material, never loses material, activates the king and "
                   "pushes passed pawns towards promotion.",
    },
    "B": {
        "opening": "The best move is the one with the best overall material result. Among moves that keep material "
                   "equal, the best one develops a knight or bishop, takes the centre or castles.",
        "middlegame": "The best move is the one with the best overall material result. Among moves that keep "
                      "material equal, the best one gives check, attacks an enemy piece or keeps the king safe.",
        "endgame": "The best move is the one with the best overall material result. Among moves that keep material "
                   "equal, the best one pushes a passed pawn or brings the king towards the centre.",
    },
}


def _name(piece: chess.Piece) -> str:
    return chess.piece_name(piece.piece_type)


def _sq(sq: int) -> str:
    return chess.square_name(sq)


def _val(piece_type: int) -> int:
    return 100 if piece_type == chess.KING else VALUE[piece_type]


def _material(board: chess.Board, color: bool) -> int:
    return sum(VALUE[p.piece_type] for p in board.piece_map().values() if p.color == color)


def game_phase(board: chess.Board) -> str:
    """opening / middlegame / endgame, dal numero di mossa e dal materiale non di pedoni rimasto."""
    heavy = sum(VALUE[p.piece_type] for p in board.piece_map().values()
                if p.piece_type not in (chess.PAWN, chess.KING))
    if heavy <= 26:  # all'incirca una torre e un pezzo minore a testa, o meno
        return "endgame"
    if board.fullmove_number <= 10:
        return "opening"
    return "middlegame"


def see(board: chess.Board, sq: int) -> int:
    """Materiale che chi ha il tratto puo' vincere iniziando gli scambi su `sq` (0 se non conviene).

    Static exchange evaluation sulle mosse legali: a ogni passo cattura il pezzo meno prezioso, e ognuno puo'
    fermarsi quando continuare non conviene. Le inchiodature e i pezzi "a raggi X" sono gestiti dalle regole.
    """
    target = board.piece_at(sq)
    if target is None or target.color == board.turn:
        return 0
    b = board.copy(stack=False)
    gains, on_square = [], _val(target.piece_type)
    while True:
        caps = [m for m in b.legal_moves if m.to_square == sq and b.piece_at(sq) is not None]
        if not caps:
            break
        m = min(caps, key=lambda mv: _val(b.piece_at(mv.from_square).piece_type))
        gains.append(on_square - (gains[-1] if gains else 0))
        on_square = _val(m.promotion) if m.promotion else _val(b.piece_at(m.from_square).piece_type)
        b.push(m)
    for i in range(len(gains) - 1, 0, -1):
        gains[i - 1] = -max(-gains[i - 1], gains[i])
    return max(0, gains[0]) if gains else 0


def _threats(board: chess.Board, victim: bool) -> list[tuple[int, int]]:
    """Pezzi di `victim` che l'avversario puo' vincere: [(casa, materiale)]. Richiede l'avversario al tratto."""
    if board.turn == victim:
        if board.is_check():
            return []
        board = board.copy(stack=False)
        board.push(chess.Move.null())
    out = []
    for sq, p in board.piece_map().items():
        if p.color == victim and p.piece_type != chess.KING:
            g = see(board, sq)
            if g > 0:
                out.append((sq, g))
    return sorted(out, key=lambda x: -x[1])


def _passed(board: chess.Board, sq: int) -> bool:
    p = board.piece_at(sq)
    f, r = chess.square_file(sq), chess.square_rank(sq)
    ahead = range(r + 1, 8) if p.color == chess.WHITE else range(0, r)
    for ff in (f - 1, f, f + 1):
        if 0 <= ff < 8:
            for rr in ahead:
                q = board.piece_at(chess.square(ff, rr))
                if q and q.piece_type == chess.PAWN and q.color != p.color:
                    return False
    return True


def _centre_distance(sq: int) -> float:
    return abs(chess.square_file(sq) - 3.5) + abs(chess.square_rank(sq) - 3.5)


def _list(board: chess.Board, items) -> str:
    return ", ".join(f"{_name(board.piece_at(s))} on {_sq(s)} ({g})" for s, g in items)


class Describer:
    def __init__(self, rubric: str = "A"):
        self.rubric = RUBRICS[rubric]

    def position(self, board: chess.Board) -> str:
        me, opp = board.turn, not board.turn
        phase = game_phase(board)
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
        mine = _threats(board, me)
        if mine:
            lines.append(f"{COLOR[me]} pieces under threat (material at risk): {_list(board, mine)}.")
        theirs = [(s, see(board, s)) for s, p in board.piece_map().items() if p.color == opp]
        theirs = sorted([(s, g) for s, g in theirs if g > 0], key=lambda x: -x[1])
        if theirs:
            lines.append(f"{COLOR[opp]} pieces that {COLOR[me]} can win now: {_list(board, theirs)}.")
        lines.append(self.rubric[phase])
        return " ".join(lines)

    def move(self, board: chess.Board, move: chess.Move) -> str:
        me, opp = board.turn, not board.turn
        piece = board.piece_at(move.from_square)
        san = board.san(move)
        phase = game_phase(board)
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
            gain += VALUE[move.promotion] - 1
            facts.append(f"It promotes the pawn to a {chess.piece_name(move.promotion)}.")

        if phase == "opening" and not board.is_castling(move):
            back_rank = 0 if me == chess.WHITE else 7
            file_to = chess.square_file(move.to_square)
            if piece.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(move.from_square) == back_rank:
                facts.append(f"It develops the {_name(piece)}.")
            elif piece.piece_type == chess.PAWN and file_to in (3, 4):
                facts.append("It takes space in the centre.")
            elif piece.piece_type == chess.PAWN and file_to in (0, 7):
                facts.append("It is a slow edge pawn move that does not develop.")
            elif piece.piece_type == chess.QUEEN:
                facts.append("It brings the queen out early.")
            elif piece.piece_type in (chess.KING, chess.ROOK) and board.has_castling_rights(me):
                facts.append(f"It moves the {_name(piece)} and gives up the right to castle.")

        if phase == "endgame":
            if piece.piece_type == chess.PAWN and _passed(board, move.from_square):
                rank = chess.square_rank(move.to_square) + 1 if me == chess.WHITE else 8 - chess.square_rank(move.to_square)
                facts.append(f"It pushes a passed pawn to the {rank}th rank.")
            if piece.piece_type == chess.KING:
                closer = _centre_distance(move.to_square) < _centre_distance(move.from_square)
                facts.append("It brings the king closer to the centre." if closer
                             else "It moves the king away from the centre.")

        threatened_before = dict(_threats(board, me))
        board.push(move)
        try:
            if board.is_checkmate():
                facts.append("It is checkmate and wins the game immediately.")
                return " ".join(facts)
            if board.is_stalemate():
                facts.append("It causes stalemate, so the game ends in a draw.")
            elif board.is_check():
                facts.append("It gives check.")

            # bilancio degli scambi: cosa puo' vincere l'avversario dopo la mossa
            after = _threats(board, me)
            loss_here = next((g for s, g in after if s == move.to_square), 0)
            loss_else = [(s, g) for s, g in after if s != move.to_square]
            if loss_here:
                facts.append(f"After it {COLOR[opp]} can win {loss_here} by capturing on {_sq(move.to_square)}.")
            elif move.from_square in threatened_before:
                facts.append(f"It saves the threatened {_name(piece)}.")
            if loss_else:
                facts.append(f"It leaves en prise: {_list(board, loss_else)}.")

            targets = _threats(board, opp) if not board.is_check() else []
            if targets:
                facts.append(f"It threatens to win: {_list(board, targets)}.")

            net = gain - max([loss_here] + [g for _, g in loss_else])
            if net > 0:
                facts.append(f"Overall result: {COLOR[me]} wins {net} point{'s' if net != 1 else ''} of material.")
            elif net < 0:
                facts.append(f"Overall result: {COLOR[me]} loses {-net} point{'s' if net != -1 else ''} of material.")
            else:
                facts.append("Overall result: material stays equal.")
        finally:
            board.pop()
        return " ".join(facts)


def describe_position(board: chess.Board, rubric: str = "A") -> str:
    return Describer(rubric).position(board)


def describe_move(board: chess.Board, move: chess.Move, rubric: str = "A") -> str:
    return Describer(rubric).move(board, move)


class OpenJevPlayer:
    def __init__(self, subfolder: str | None = None, rubric: str | None = None):
        cuda = torch.cuda.is_available()
        self.device = "cuda" if cuda else "cpu"
        # Il 2B pesa 4,4 GB in bf16: serve memoria libera anche per attivazioni e cuBLAS
        free_gb = torch.cuda.mem_get_info()[0] / 1e9 if cuda else 0
        self.subfolder = subfolder or os.environ.get("OPENJEV_MODEL") or (
            "qwen3.5-2b-nli-v5" if free_gb > 6.5 else "qwen3.5-0.8b-nli-v2s-long")
        self.describer = Describer(rubric or os.environ.get("OPENJEV_RUBRIC", "A"))
        dtype = torch.bfloat16 if cuda else torch.float32
        t = time.time()
        self.jev = OpenJevCrossEncoder(MODELS, subfolder=self.subfolder, device=self.device, dtype=dtype)
        self.load_s = time.time() - t

    def choose(self, board: chess.Board, chunk: int = 24) -> dict:
        moves = list(board.legal_moves)
        premise = self.describer.position(board)
        hyps = [self.describer.move(board, m) for m in moves]
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
