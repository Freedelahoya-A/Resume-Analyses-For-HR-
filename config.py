import os
from pathlib import Path

DATA_DIR       = Path(__file__).parent / "data"
INDEX_FILE     = Path(__file__).parent / "index.json"
CHUNK_SIZE     = 120
CHUNK_OVERLAP  = 30
TOP_K          = 5
VECTOR_DIM     = 128
BM25_K1        = 1.5
BM25_B         = 0.75
HYBRID_ALPHA   = 0.5   # 0 = BM25 only, 1 = vector only

# Ollama local endpoint and model
OLLAMA_HOST     = os.environ.get("CLIPLLM_OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME      = os.environ.get("RESUME_RAG_MODEL", os.environ.get("CLIPLLM_MODEL", "llama3.1:8b"))
REQUEST_TIMEOUT = int(os.environ.get("CLIPLLM_TIMEOUT", "120"))
