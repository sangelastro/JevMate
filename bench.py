"""Banco di prova per la mente di OpenJev: misura la qualità delle mosse con Stockfish come arbitro.

Stockfish non sceglie niente: valuta soltanto la mossa scelta da OpenJev rispetto alla migliore.

    python bench.py --label v1 --engine v1           # motore della prima versione (bench/engine_v1.py)
    python bench.py --label v2-A --rubric A          # motore attuale, criterio A
    python bench.py --label v2-B --rubric B --games 6   # più partite contro Stockfish al livello minimo

Metriche sulle posizioni (sempre le stesse, in bench/positions.json):
  ACPL      perdita media in centipedoni rispetto alla mossa migliore (più bassa è meglio)
  top-1     % di mosse uguali alla prima scelta di Stockfish
  blunder   % di mosse che perdono 200 centipedoni o più
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
SF = os.path.join(HERE, "tools", "stockfish.exe")
BENCH = os.path.join(HERE, "bench")
CAP = 1000


def phase_of(board: chess.Board) -> str:
    from engine import game_phase
    return game_phase(board)


def make_positions(sf: chess.engine.SimpleEngine, n: int, seed: int = 7, only: str | None = None,
                   plies_range: tuple[int, int] | None = None) -> list[dict]:
    """Posizioni realistiche: partite tra mosse forti scelte a caso tra le prime 3 di Stockfish."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        plies = rng.randint(*plies_range) if plies_range else             rng.choice([rng.randint(4, 16), rng.randint(17, 40), rng.randint(41, 80)])
        for _ in range(plies):
            if board.is_game_over():
                break
            info = sf.analyse(board, chess.engine.Limit(depth=8), multipv=3)
            board.push(rng.choice(info)["pv"][0])
        if board.is_game_over() or board.legal_moves.count() < 2:
            continue
        if only and phase_of(board) != only:
            continue
        out.append({"fen": board.fen(), "phase": phase_of(board)})
    return out


def score(info, pov) -> int:
    s = info["score"].pov(pov).score(mate_score=1500)
    return max(-1500, min(1500, s))


def eval_positions(player: OpenJevPlayer, sf, positions, depth: int) -> list[dict]:
    rows = []
    for k, pos in enumerate(positions):
        board = chess.Board(pos["fen"])
        me = board.turn
        best = sf.analyse(board, chess.engine.Limit(depth=depth))
        best_move, best_cp = best["pv"][0], score(best, me)
        choice = player.choose(board)
        move = choice["move"]
        board.push(move)
        chosen_cp = best_cp if move == best_move else score(sf.analyse(board, chess.engine.Limit(depth=depth)), me)
        board.pop()
        loss = max(0, min(CAP, best_cp - chosen_cp))
        rows.append({"fen": pos["fen"], "phase": pos["phase"], "move": board.san(move), "best": board.san(best_move),
                     "loss": loss, "ms": choice["ms"]})
        print(f"\r  posizione {k + 1}/{len(positions)}", end="", flush=True)
    print()
    return rows


def play_games(player: OpenJevPlayer, sf, n: int) -> dict:
    sf.configure({"Skill Level": 0})
    res = {"win": 0, "draw": 0, "loss": 0}
    for g in range(n):
        board = chess.Board()
        jev_color = chess.WHITE if g % 2 == 0 else chess.BLACK
        while not board.is_game_over(claim_draw=True) and board.ply() < 240:
            if board.turn == jev_color:
                board.push(player.choose(board)["move"])
            else:
                board.push(sf.play(board, chess.engine.Limit(time=0.05)).move)
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            res["draw"] += 1
        elif outcome.winner == jev_color:
            res["win"] += 1
        else:
            res["loss"] += 1
        print(f"  partita {g + 1}/{n}: {board.result(claim_draw=True)} (OpenJev col "
              f"{'bianco' if jev_color else 'nero'}, {board.ply()} semimosse)")
    sf.configure({"Skill Level": 20})
    return res


def summarize(rows) -> dict:
    def agg(rs):
        if not rs:
            return None
        return {"n": len(rs),
                "acpl": round(sum(r["loss"] for r in rs) / len(rs), 1),
                "top1": round(100 * sum(r["move"] == r["best"] for r in rs) / len(rs), 1),
                "blunder": round(100 * sum(r["loss"] >= 200 for r in rs) / len(rs), 1)}
    out = {"tutte": agg(rows)}
    for ph in ("opening", "middlegame", "endgame"):
        out[ph] = agg([r for r in rows if r["phase"] == ph])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=80, help="posizioni da generare la prima volta")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--engine", choices=["current", "v1"], default="current")
    ap.add_argument("--rubric", choices=["A", "B"], default="A")
    args = ap.parse_args()

    os.makedirs(os.path.join(BENCH, "results"), exist_ok=True)
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    sf.configure({"Threads": 4, "Hash": 128})
    pos_file = os.path.join(BENCH, "positions.json")
    if not os.path.exists(pos_file):
        print(f"Genero {args.n} posizioni fisse…")
        json.dump(make_positions(sf, args.n), open(pos_file, "w"), indent=1)
    positions = json.load(open(pos_file))
    if sum(p["phase"] == "endgame" for p in positions) < 20:
        print("Aggiungo 20 posizioni di finale…")
        positions += make_positions(sf, 20, seed=11, only="endgame", plies_range=(70, 140))
        json.dump(positions, open(pos_file, "w"), indent=1)

    if args.engine == "v1":
        import sys
        sys.path.insert(0, BENCH)
        from engine_v1 import OpenJevPlayer
        player = OpenJevPlayer()
    else:
        from engine import OpenJevPlayer
        player = OpenJevPlayer(rubric=args.rubric)
    print(f"Modello {player.subfolder} su {player.device}, {len(positions)} posizioni")
    t = time.time()
    rows = eval_positions(player, sf, positions, args.depth)
    summary = summarize(rows)
    games = play_games(player, sf, args.games) if args.games else None
    sf.quit()

    result = {"label": args.label, "engine": args.engine, "rubric": args.rubric, "model": player.subfolder, "depth": args.depth, "summary": summary,
              "games_vs_sf_level0": games, "seconds": round(time.time() - t), "rows": rows}
    json.dump(result, open(os.path.join(BENCH, "results", f"{args.label}.json"), "w"), indent=1)

    print(f"\n== {args.label} ==")
    print(f"{'fase':<11}{'n':>4}{'ACPL':>8}{'top-1':>8}{'blunder':>9}")
    for ph, s in summary.items():
        if s:
            print(f"{ph:<11}{s['n']:>4}{s['acpl']:>8}{s['top1']:>7}%{s['blunder']:>8}%")
    if games:
        print(f"contro Stockfish livello 0: {games}")


if __name__ == "__main__":
    main()
