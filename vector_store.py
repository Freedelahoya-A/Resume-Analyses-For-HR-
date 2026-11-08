import json
import math
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

from . import config
from .utils import tokenize


class VectorIndex:
    def __init__(self):
        self.vectors: List[List[float]] = []
        self.ids: List[str] = []

    def _norm(self, v: List[float]) -> List[float]:
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v

    def _dot(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def add(self, vectors: List[List[float]], ids: List[str]) -> None:
        for v, i in zip(vectors, ids):
            self.vectors.append(self._norm(v))
            self.ids.append(i)

    def search(self, query_vec: List[float], k: int) -> List[Tuple[str, float]]:
        if not self.vectors:
            return []
        q = self._norm(query_vec)
        scored = [(self._dot(q, v), i) for i, v in enumerate(self.vectors)]
        scored.sort(reverse=True)
        return [(self.ids[i], s) for s, i in scored[:k]]


class BM25Index:
    def __init__(self):
        self.corpus: List[List[str]] = []
        self.ids: List[str] = []
        self.df: Dict[str, int] = {}
        self.avg_len: float = 0.0

    def add(self, chunks: List[str], ids: List[str]) -> None:
        for chunk, cid in zip(chunks, ids):
            tokens = tokenize(chunk)
            self.corpus.append(tokens)
            self.ids.append(cid)
            for tok in set(tokens):
                self.df[tok] = self.df.get(tok, 0) + 1
        lengths = [len(d) for d in self.corpus]
        self.avg_len = sum(lengths) / len(lengths) if lengths else 1.0

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        if not self.corpus:
            return []
        N = len(self.corpus)
        q_terms = tokenize(query)
        scores: List[float] = []
        for tokens in self.corpus:
            tf_map = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in q_terms:
                tf = tf_map.get(term, 0)
                idf = math.log((N - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5) + 1)
                denom = tf + config.BM25_K1 * (1 - config.BM25_B + config.BM25_B * dl / self.avg_len)
                score += idf * (tf * (config.BM25_K1 + 1)) / denom if denom else 0
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(self.ids[i], scores[i]) for i in ranked[:k]]


class HybridStore:
    def __init__(self):
        self.vector_idx = VectorIndex()
        self.bm25_idx = BM25Index()
        self.chunks: List[str] = []
        self.ids: List[str] = []
        self.vocab: List[str] = []
        self.doc_freq: Dict[str, int] = {}

    def add(self, chunks: List[str], vectors: List[List[float]], ids: List[str]) -> None:
        self.chunks.extend(chunks)
        self.ids.extend(ids)
        self.vector_idx.add(vectors, ids)
        self.bm25_idx.add(chunks, ids)

    def search(self, query: str, query_vec: List[float], k: int = None) -> List[Tuple[str, float, str]]:
        k = k or config.TOP_K
        v_results = {cid: s for cid, s in self.vector_idx.search(query_vec, k * 2)}
        b_results = {cid: s for cid, s in self.bm25_idx.search(query, k * 2)}

        all_ids = set(v_results) | set(b_results)
        v_max = max(v_results.values(), default=1) or 1
        b_max = max(b_results.values(), default=1) or 1

        hybrid: List[Tuple[float, str]] = []
        for cid in all_ids:
            v_norm = v_results.get(cid, 0) / v_max
            b_norm = b_results.get(cid, 0) / b_max
            combined = config.HYBRID_ALPHA * v_norm + (1 - config.HYBRID_ALPHA) * b_norm
            hybrid.append((combined, cid))

        hybrid.sort(reverse=True)
        results = []
        for score, cid in hybrid[:k]:
            idx = self.ids.index(cid)
            results.append((cid, score, self.chunks[idx]))
        return results

    def save(self, path: Path) -> None:
        data = {
            "ids": self.ids,
            "chunks": self.chunks,
            "vectors": self.vector_idx.vectors,
            "bm25_corpus": self.bm25_idx.corpus,
            "bm25_df": self.bm25_idx.df,
            "bm25_avg_len": self.bm25_idx.avg_len,
            "vocab": self.vocab,
            "doc_freq": self.doc_freq,
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.ids = data["ids"]
        self.chunks = data["chunks"]
        self.vocab = data["vocab"]
        self.doc_freq = data["doc_freq"]
        self.vector_idx.vectors = data["vectors"]
        self.vector_idx.ids = self.ids
        self.bm25_idx.corpus = data["bm25_corpus"]
        self.bm25_idx.ids = self.ids
        self.bm25_idx.df = data["bm25_df"]
        self.bm25_idx.avg_len = data["bm25_avg_len"]

    def __len__(self) -> int:
        return len(self.ids)
