"""Scarica da Hugging Face i checkpoint OpenJev usati dal gioco in models/openjev/.

    python scarica_modelli.py            # 0.8B (1,7 GB), basta per GPU da 6 GB o per la CPU
    python scarica_modelli.py --anche-2b # aggiunge il 2B (4,4 GB), serve una GPU con almeno 8 GB
"""
import argparse
import os

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--anche-2b", action="store_true", help="scarica anche il checkpoint 2B")
args = parser.parse_args()

patterns = ["modeling_openjev.py", "qwen3.5-0.8b-nli-v2s-long/*"]
if args.anche_2b:
    patterns.append("qwen3.5-2b-nli-v5/*")

path = snapshot_download("AlexWortega/openjev", allow_patterns=patterns,
                         local_dir=os.path.join(HERE, "models", "openjev"))
print("Modelli in", path)
