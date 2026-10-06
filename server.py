"""JevMate: server locale (Flask). Avvio: .venv\\Scripts\\python server.py  ->  http://127.0.0.1:7373"""
from __future__ import annotations

import os
import threading

import chess
from flask import Flask, jsonify, request, send_from_directory

from engine import OpenJevPlayer

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)

_player: OpenJevPlayer | None = None
_lock = threading.Lock()


def player() -> OpenJevPlayer:
    global _player
    with _lock:
        if _player is None:
            _player = OpenJevPlayer()
        return _player


def replay(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"mossa illegale: {uci}")
        board.push(move)
    return board


def state(board: chess.Board) -> dict:
    outcome = board.outcome(claim_draw=True)
    result = None
    if outcome:
        reason = {
            chess.Termination.CHECKMATE: "scacco matto",
            chess.Termination.STALEMATE: "stallo",
            chess.Termination.INSUFFICIENT_MATERIAL: "materiale insufficiente",
            chess.Termination.SEVENTYFIVE_MOVES: "regola delle 75 mosse",
            chess.Termination.FIVEFOLD_REPETITION: "ripetizione",
            chess.Termination.FIFTY_MOVES: "regola delle 50 mosse",
            chess.Termination.THREEFOLD_REPETITION: "triplice ripetizione",
        }.get(outcome.termination, outcome.termination.name.lower())
        winner = None if outcome.winner is None else ("white" if outcome.winner else "black")
        result = {"winner": winner, "reason": reason}
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn else "black",
        "check": board.is_check(),
        "legal": [m.uci() for m in board.legal_moves],
        "result": result,
    }


@app.get("/")
def index():
    return send_from_directory(os.path.join(HERE, "static"), "index.html")


@app.get("/api/info")
def info():
    p = _player
    return jsonify({"loaded": p is not None, "model": p.subfolder if p else None, "device": p.device if p else None})


@app.post("/api/warmup")
def warmup():
    p = player()
    return jsonify({"model": p.subfolder, "device": p.device, "load_s": round(p.load_s, 1)})


@app.post("/api/state")
def api_state():
    try:
        board = replay(request.json.get("moves", []))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(state(board))


@app.post("/api/move")
def api_move():
    moves = request.json.get("moves", [])
    try:
        board = replay(moves)
        move = chess.Move.from_uci(request.json["uci"])
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    if move not in board.legal_moves:
        return jsonify({"error": "mossa illegale"}), 400
    san = board.san(move)
    capture = board.is_capture(move)
    board.push(move)
    return jsonify({"san": san, "uci": move.uci(), "capture": capture, **state(board)})


@app.post("/api/ai")
def api_ai():
    try:
        board = replay(request.json.get("moves", []))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if board.is_game_over(claim_draw=True):
        return jsonify({"error": "partita finita"}), 400
    p = player()
    with _lock:
        choice = p.choose(board)
    move = choice["move"]
    san = board.san(move)
    capture = board.is_capture(move)
    board.push(move)
    return jsonify({
        "san": san, "uci": move.uci(), "capture": capture,
        "ms": choice["ms"], "candidates": choice["candidates"], "n_moves": choice["n_moves"],
        "premise": choice["premise"], "model": p.subfolder, "device": p.device,
        **state(board),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7373))
    print(f"JevMate -> http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
