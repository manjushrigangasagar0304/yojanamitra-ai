"""Shared settings: paths, embedding model and LLM."""
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", message=".*langchain-community.*")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SCHEMES_DIR = DATA_DIR / "schemes"
SCHEMES_JSON = DATA_DIR / "schemes.json"
VECTORSTORE_DIR = ROOT / "vectorstore"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Local MiniLM embeddings (free, no API key).

    Set EMBEDDINGS=fake only for offline tests where the model can't be downloaded.
    """
    if os.getenv("EMBEDDINGS") == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=384)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_llm():
    """Chat model with tool calling. Gemini by default, Groq as a free fallback."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI
    # gemini-flash-latest always points to Google's current Flash model
    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"))
