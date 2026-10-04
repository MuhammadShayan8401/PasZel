"""Lexical retrieval (BM25) over stored page text. Keyword matching only - no embeddings."""
import math
import re
from collections import Counter


def tokens(s) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


def chunk_pages(d: dict, window: int = 10, stride: int = 5) -> list[dict]:
    out = []
    for doc_id, pages in d.get("pages", {}).items():
        for pno, text in pages.items():
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            for k, i in enumerate(range(0, max(len(lines), 1), stride)):
                seg = lines[i:i + window]
                if seg:
                    out.append({"id": f"{doc_id}:{pno}:{k}", "doc": doc_id, "page": int(pno), "text": " ".join(seg)})
                if i + window >= len(lines):
                    break
    return out


def search(d: dict, query: str, k: int = 5, k1: float = 1.5, b: float = 0.75) -> list[dict]:
    chunks, q = chunk_pages(d), set(tokens(query))
    if not chunks or not q:
        return []
    docs = [Counter(tokens(c["text"])) for c in chunks]
    avg = (sum(sum(x.values()) for x in docs) / len(docs)) or 1
    res = []
    for c, tf in zip(chunks, docs):
        L, score = sum(tf.values()), 0.0
        for t in q:
            f = tf.get(t, 0)
            if f:
                n = sum(1 for x in docs if t in x)
                score += math.log(1 + (len(docs) - n + .5) / (n + .5)) * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avg))
        if score > 0:
            res.append({**c, "score": round(score, 3)})
    return sorted(res, key=lambda r: -r["score"])[:k]
