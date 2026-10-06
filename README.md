# JevMate — Scacchi dei Maghi

Scacchi umano contro computer, dove il computer sceglie la mossa con
[OpenJev](https://huggingface.co/AlexWortega/openjev), un cross-encoder NLI basato su Qwen3.5. Non c'è un motore
scacchistico.

## Come sceglie la mossa

- **Premessa:** la posizione (FEN, materiale, ultima mossa, scacco, pezzi minacciati) più un criterio:
  *"una buona mossa vince materiale, evita di perdere pezzi, tiene il re al sicuro"*.
- **Ipotesi:** una per ogni mossa legale, del tipo *"The best move for Black is fxg5: …"*, con i fatti calcolati da
  python-chess (cattura, scacco, pezzo che resta in presa, pezzi lasciati scoperti, sviluppo in apertura).
- **Scelta:** OpenJev dà a ogni ipotesi una probabilità di entailment e vince la più alta. Il pannello
  "La mente di OpenJev" mostra le 8 mosse migliori.

Gioca debole: i fatti gli evitano gli errori grossolani, ma non vede più avanti di una mossa.

## Installazione (Windows)

```bat
py -m venv .venv
rem torch: scegli la build adatta (https://pytorch.org/get-started/locally/)
rem   GPU NVIDIA con driver recente:  --index-url https://download.pytorch.org/whl/cu128
rem   driver vecchio (CUDA 11.x):     torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
rem   solo CPU:                       --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scarica_modelli.py
```

## Avvio

```bat
avvia.cmd
```

Apre http://127.0.0.1:7373. Il primo caricamento del modello richiede qualche secondo.

**Modello usato:** il 2B se la GPU ha più di 6,5 GB liberi, altrimenti lo 0.8B (anche su CPU, più lento). Per
forzarne uno: `set OPENJEV_MODEL=qwen3.5-0.8b-nli-v2s-long`.

## Come si gioca

Clic sul pezzo, poi clic sulla casella di arrivo. Punto dorato = casella libera, cerchio arancione = cattura.
Per arroccare: re → g1/c1. Opzioni: gioca coi neri, annulla, voce dei comandi (it-IT), mostra i pensieri.

## Struttura

| File | Ruolo |
|---|---|
| `engine.py` | descrizione della posizione e delle mosse, scelta con OpenJev |
| `server.py` | server Flask: regole (python-chess) e API |
| `static/index.html` | interfaccia |
| `scarica_modelli.py` | download dei checkpoint da Hugging Face |

## Licenze

Codice di questo progetto: licenza da definire. OpenJev e `modeling_openjev.py` (scaricati in `models/`): MIT,
© AlexWortega. Base: Qwen3.5 (Alibaba).
