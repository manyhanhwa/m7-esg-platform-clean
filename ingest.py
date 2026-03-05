import os, glob, json
import numpy as np
import faiss
from pypdf import PdfReader
from pypdf.errors import PdfStreamError
from openai import OpenAI

DATA_DIR = "data"
INDEX_DIR = "index"
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI()

def read_pdf(path: str) -> list[dict]:
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            out.append({"source": os.path.basename(path), "page": i+1, "text": text})
    return out

def chunk(text: str, size=1200, overlap=200):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def embed_texts(texts: list[str]) -> np.ndarray:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in res.data]
    return np.array(vecs, dtype="float32")

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    pdfs = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdfs:
        print("❌ data/ 폴더에 PDF가 없습니다.")
        return

    docs = []
    bad = []

    for pdf in pdfs:
        try:
            pages = read_pdf(pdf)
            if not pages:
                print(f"⚠️ 텍스트 없음(스캔/이미지 PDF 가능): {os.path.basename(pdf)}")
            for p in pages:
                for c in chunk(p["text"]):
                    docs.append({"source": p["source"], "page": p["page"], "text": c})
            print(f"✅ loaded: {os.path.basename(pdf)}  (pages with text: {len(pages)})")
        except PdfStreamError as e:
            bad.append((os.path.basename(pdf), str(e)))
            print(f"❌ broken pdf skipped: {os.path.basename(pdf)}  | {e}")
        except Exception as e:
            bad.append((os.path.basename(pdf), str(e)))
            print(f"❌ error pdf skipped: {os.path.basename(pdf)}  | {e}")

    if not docs:
        print("❌ 어떤 PDF에서도 텍스트를 추출하지 못했습니다.")
        return

    vectors = []
    meta = []
    batch = 64

    for i in range(0, len(docs), batch):
        part = docs[i:i+batch]
        vec = embed_texts([x["text"] for x in part])
        vectors.append(vec)
        meta.extend(part)

    vectors = np.vstack(vectors)
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "meta.jsonl"), "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"\n✅ Indexed chunks: {len(meta)}")
    print(f"✅ PDFs processed: {len(pdfs) - len(bad)} / {len(pdfs)}")
    if bad:
        print("⚠️ Skipped PDFs:")
        for name, err in bad:
            print(f" - {name}: {err}")

if __name__ == "__main__":
    main()