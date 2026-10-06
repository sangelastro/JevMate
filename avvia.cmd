@echo off
rem JevMate - avvia il server locale e apre il browser
cd /d "%~dp0"
set SSL_CERT_FILE=
set HF_HUB_OFFLINE=1
start "" http://127.0.0.1:7373
".venv\Scripts\python.exe" server.py
