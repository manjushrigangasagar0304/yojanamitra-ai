#!/usr/bin/env bash
# One-click setup and launch for macOS / Linux
set -e
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
else
  source venv/bin/activate
fi
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Paste your Gemini key into .env, then run ./run.sh again."
  exit 0
fi
[ -f vectorstore/index.faiss ] || python ingest.py
streamlit run Home.py
