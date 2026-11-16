"""Central configuration loaded from the project's .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
# Load local settings without overriding environment variables already set by the shell.
load_dotenv(ROOT_DIR / ".env", override=False)

DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"


def configured_path(variable: str, default: Path) -> Path:
    """Resolve relative .env paths against the project root, not the shell's CWD."""
    value = Path(os.getenv(variable, str(default))).expanduser()
    return value if value.is_absolute() else ROOT_DIR / value


PDF_PATH = configured_path("EU_AI_ACT_PDF", DATA_DIR / "EU_AI_Act.pdf")
CHROMA_DIR = configured_path("CHROMA_DIR", STORAGE_DIR / "chroma")
CHUNKS_PATH = configured_path("CHUNKS_PATH", STORAGE_DIR / "chunks.jsonl")
EVAL_PATH = configured_path("EVAL_PATH", DATA_DIR / "eval_questions.json")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "eu_ai_act_v1")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
VECTOR_CANDIDATES = int(os.getenv("VECTOR_CANDIDATES", "20"))
BM25_CANDIDATES = int(os.getenv("BM25_CANDIDATES", "20"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "12"))
FINAL_K = int(os.getenv("FINAL_K", "5"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
