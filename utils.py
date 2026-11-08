import os
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

from . import config


def load_resumes(data_dir: Path) -> Dict[str, str]:
    resumes: Dict[str, str] = {}
    for entry in sorted(os.listdir(data_dir)):
        path = data_dir / entry
        if path.is_file() and path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                resumes[path.stem] = text
    return resumes


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str) -> List[str]:
    text = _clean(text)
    words = text.split()
    size, overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        if i + size >= len(words):
            break
        i += size - overlap
    return chunks


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_tfidf_vocab(corpus: List[str]) -> List[str]:
    df: Dict[str, int] = {}
    for doc in corpus:
        for tok in set(tokenize(doc)):
            df[tok] = df.get(tok, 0) + 1
    N = len(corpus)
    idf = {t: math.log((N + 1) / (f + 1)) + 1 for t, f in df.items()}
    return sorted(idf, key=lambda t: -idf[t])[: config.VECTOR_DIM]


def tfidf_vector(text: str, vocab: List[str], corpus_size: int, doc_freq: Dict[str, int]) -> List[float]:
    tokens = tokenize(text)
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    vec: List[float] = []
    for term in vocab:
        t = tf.get(term, 0) / total
        idf = math.log((corpus_size + 1) / (doc_freq.get(term, 0) + 1)) + 1
        vec.append(t * idf)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def build_corpus_stats(chunks: List[str]) -> Tuple[List[str], Dict[str, int]]:
    vocab = build_tfidf_vocab(chunks)
    doc_freq: Dict[str, int] = {}
    for chunk in chunks:
        for tok in set(tokenize(chunk)):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    return vocab, doc_freq


def embed_chunks(chunks: List[str], vocab: List[str], doc_freq: Dict[str, int]) -> List[List[float]]:
    n = len(chunks)
    return [tfidf_vector(c, vocab, n, doc_freq) for c in chunks]


def get_query_embedding(query: str, vocab: List[str], doc_freq: Dict[str, int], corpus_size: int) -> List[float]:
    return tfidf_vector(query, vocab, corpus_size, doc_freq)
