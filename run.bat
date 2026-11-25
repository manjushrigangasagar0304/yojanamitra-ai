@echo off
REM One-click setup and launch for Windows
if not exist venv (
  python -m venv venv
  call venv\Scripts\activate
  pip install -r requirements.txt
) else (
  call venv\Scripts\activate
)
if not exist .env (
  copy .env.example .env
  echo Paste your Gemini key into .env, then run this file again.
  notepad .env
  exit /b
)
if not exist vectorstore\index.faiss python ingest.py
streamlit run Home.py
