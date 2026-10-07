"""Da dove arriva la perdita di una misura di bench.py? Non serve la GPU.

    python bench/diagnose.py bench/results/v2-A.json

Per ogni posizione confronta il bilancio di materiale ("Overall result" nelle ipotesi) della mossa scelta con
quello della mossa migliore secondo Stockfish e con il massimo disponibile:
  A  fatti pesati male: c'era un bilancio migliore e la mossa migliore lo aveva
  B  bilancio peggiore del massimo, ma la mossa migliore non è quella col bilancio massimo
  C  stesso bilancio della mossa migliore: mancano fatti (mosse tranquille, posizionali)
  D  la mossa migliore ha un bilancio peggiore: sacrificio o tattica oltre una mossa
"""
import collections
import json
import os
import re
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Describer  # noqa: E402

LABELS = {
    "A": "fatti pesati male",
    "B": "bilancio peggiore, ma la migliore non è quella col bilancio massimo",
    "C": "stesso bilancio della migliore: mancano fatti",
    "D": "la migliore sacrifica materiale (oltre una mossa)",
    "ok": "mossa buona (perdita sotto 50 cp)",
}


def net(text: str) -> int:
    if "checkmate" in text:
        return 99
    m = re.search(r"Overall result: \w+ (wins|loses) (\d+)", text)
    if not m:
        return 0
    return int(m.group(2)) if m.group(1) == "wins" else -int(m.group(2))


def main(path: str, rubric: str = "A"):
    rows = json.load(open(path))["rows"]
    d = Describer(rubric)
    cat = collections.defaultdict(list)
    for x in rows:
        b = chess.Board(x["fen"])
        nets = {b.san(m): net(d.move(b, m)) for m in b.legal_moves}
        top, chosen, best = max(nets.values()), nets[x["move"]], nets[x["best"]]
        if x["loss"] < 50:
            k = "ok"
        elif chosen < top and best > chosen:
            k = "A"
        elif chosen < top:
            k = "B"
        elif best == chosen:
            k = "C"
        else:
            k = "D"
        cat[k].append(x["loss"])
    total = sum(x["loss"] for x in rows) or 1
    print(f"{'caso':<5}{'pos.':>5}{'% perdita':>11}{'media cp':>10}  descrizione")
    for k in ("ok", "A", "B", "C", "D"):
        v = cat.get(k, [])
        if v:
            print(f"{k:<5}{len(v):>5}{100 * sum(v) / total:>10.1f}%{sum(v) / len(v):>10.0f}  {LABELS[k]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
