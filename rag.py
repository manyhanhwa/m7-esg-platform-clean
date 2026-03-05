import os
import json
import faiss
import numpy as np
from openai import OpenAI

INDEX_DIR = "index"
EMBED_MODEL = "text-embedding-3-small"

# ✅ 키가 없으면 client를 만들지 않는다 (데모 모드/배포 초기 대응)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


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


def embed(text: str) -> np.ndarray:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 없어 임베딩을 생성할 수 없습니다.")
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    return v


def retrieve(query: str, k: int = 8):
    """
    hits: [{"text":..., "source":..., "page":..., "score":...}]
    """
    # ✅ 키 없으면 RAG는 꺼진 것으로 처리
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 없어 RAG를 실행할 수 없습니다.")

    index, meta = load_index()
    qv = embed(query)

    D, I = index.search(qv.reshape(1, -1), k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        row = meta[idx]
        hits.append(
            {
                "text": row.get("text", ""),
                "source": row.get("source", "unknown"),
                "page": row.get("page", -1),
                "score": float(score),
            }
        )
    return hits