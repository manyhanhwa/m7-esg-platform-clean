import os, json
import numpy as np
import faiss
from openai import OpenAI

INDEX_DIR = "index"
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI()

def load_index():
    idx_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "meta.jsonl")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        raise RuntimeError("인덱스가 없습니다. 먼저 `python3 ingest.py`를 실행하세요.")

    index = faiss.read_index(idx_path)
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return index, meta

def embed_query(q: str) -> np.ndarray:
    res = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(res.data[0].embedding, dtype="float32")[None, :]
    faiss.normalize_L2(v)
    return v

def retrieve(q: str, k=8):
    index, meta = load_index()
    v = embed_query(q)
    scores, ids = index.search(v, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        m = meta[int(idx)]
        results.append({**m, "score": float(score)})
    return results