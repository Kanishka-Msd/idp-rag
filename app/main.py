from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import os
import uuid
from pathlib import Path
from fastapi import HTTPException
import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer

app = FastAPI(title="IDP + RAG (Starter)")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)
FAISS_DIR = Path("storage/faiss")
FAISS_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # ensures folder exists

TEXT_DIR = Path("storage/text")
TEXT_DIR.mkdir(parents=True, exist_ok=True)


class AskRequest(BaseModel):
    question: str


class DocRequest(BaseModel):
    doc_id: str

class RagAskRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 4

class FieldExtractRequest(BaseModel):
    doc_id: str
    doc_type: str  # for now only "invoice"


@app.get("/health")
def health_check():
    return {"status": "running"}


@app.post("/ask")
def ask_question(payload: AskRequest):
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": payload.question,
            "stream": False
        },
        timeout=120
    )
    r.raise_for_status()
    data = r.json()

    return {
        "model": data.get("model"),
        "answer": data.get("response"),
        "done": data.get("done")
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1) Validate file type
    if file.content_type not in ["application/pdf"]:
        return {"error": "Only PDF files are allowed."}

    # 2) Create a unique doc_id
    doc_id = str(uuid.uuid4())

    # 3) Save file to disk
    save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "saved_as": str(save_path),
        "bytes": len(content),
    }   
    
def find_pdf_by_doc_id(doc_id: str) -> Path:
    matches = list(UPLOAD_DIR.glob(f"{doc_id}_*.pdf"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF found for doc_id={doc_id}. Use /upload first and copy the returned doc_id."
        )
    return matches[0]


def extract_text_pdfplumber(pdf_path: Path) -> str:
    text_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def extract_text_ocr(pdf_path: Path) -> str:
    images = convert_from_path(str(pdf_path))
    ocr_parts = []
    for img in images:
        ocr_parts.append(pytesseract.image_to_string(img))
    return "\n".join(ocr_parts).strip()

def load_full_text(doc_id: str) -> str:
    text_path = TEXT_DIR / f"{doc_id}.txt"
    if not text_path.exists():
        raise HTTPException(status_code=404, detail="Extracted text not found. Run /extract-text first.")
    return text_path.read_text(encoding="utf-8", errors="ignore")

    

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    chunks = []
    i = 0
    n = len(text)

    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        if end == n:
            break
        i = end - overlap
        if i < 0:
            i = 0

    return chunks

def load_chunks_for_doc(doc_id: str) -> list[str]:
    chunks_path = TEXT_DIR / f"{doc_id}.chunks.txt"
    if not chunks_path.exists():
        raise HTTPException(status_code=404, detail="Chunks not found. Run /extract-text first.")

    raw = chunks_path.read_text(encoding="utf-8", errors="ignore")
    chunks = [c.strip() for c in raw.split("\n\n---CHUNK---\n\n") if c.strip()]
    return chunks


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (works well if vectors normalized)
    faiss.normalize_L2(vectors)     # normalize for cosine similarity
    index.add(vectors)
    return index


def save_faiss_for_doc(doc_id: str, index: faiss.Index, chunks: list[str]):
    index_path = FAISS_DIR / f"{doc_id}.index"
    meta_path = FAISS_DIR / f"{doc_id}.meta.json"

    faiss.write_index(index, str(index_path))

    meta = {
        "doc_id": doc_id,
        "num_chunks": len(chunks),
        "embed_model": EMBED_MODEL_NAME
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def load_faiss_index_for_doc(doc_id: str) -> faiss.Index:
    index_path = FAISS_DIR / f"{doc_id}.index"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="FAISS index not found. Run /ingest first.")
    return faiss.read_index(str(index_path))


def retrieve_top_chunks(doc_id: str, question: str, top_k: int = 4) -> list[dict]:
    chunks = load_chunks_for_doc(doc_id)
    index = load_faiss_index_for_doc(doc_id)

    # Embed question
    q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)

    # Search
    scores, ids = index.search(q_vec, top_k)
    results = []
    for rank, idx in enumerate(ids[0]):
        if idx == -1:
            continue
        results.append({
            "rank": rank + 1,
            "chunk_id": int(idx),
            "score": float(scores[0][rank]),
            "text": chunks[int(idx)]
        })
    return results

def load_full_text(doc_id: str) -> str:
    text_path = TEXT_DIR / f"{doc_id}.txt"
    if not text_path.exists():
        raise HTTPException(status_code=404, detail="Extracted text not found. Run /extract-text first.")
    return text_path.read_text(encoding="utf-8", errors="ignore")


@app.post("/extract-text")
def extract_text(payload: DocRequest):
    pdf_path = find_pdf_by_doc_id(payload.doc_id)

    # Extract text
    text = extract_text_pdfplumber(pdf_path)
    method = "pdf_text"

    # Fallback to OCR if little text found
    if len(text) < 200:
        text = extract_text_ocr(pdf_path)
        method = "ocr"

    # Save full extracted text
    text_path = TEXT_DIR / f"{payload.doc_id}.txt"
    text_path.write_text(text, encoding="utf-8", errors="ignore")

    # Chunk the text
    chunks = chunk_text(text, chunk_size=800, overlap=120)

    # Save chunks
    chunks_path = TEXT_DIR / f"{payload.doc_id}.chunks.txt"
    chunks_path.write_text(
        "\n\n---CHUNK---\n\n".join(chunks),
        encoding="utf-8",
        errors="ignore"
    )

    return {
        "doc_id": payload.doc_id,
        "method": method,
        "text_length": len(text),
        "text_saved_to": str(text_path),
        "num_chunks": len(chunks),
        "first_chunk_preview": chunks[0][:500] if chunks else ""
    }


@app.post("/ingest")
def ingest_doc(payload: DocRequest):
    chunks = load_chunks_for_doc(payload.doc_id)

    # Embed chunks
    vectors = embedder.encode(chunks, convert_to_numpy=True).astype("float32")

    # Build + save FAISS
    index = build_faiss_index(vectors)
    save_faiss_for_doc(payload.doc_id, index, chunks)

    return {
        "doc_id": payload.doc_id,
        "num_chunks_indexed": len(chunks),
        "faiss_index_saved_to": str(FAISS_DIR / f"{payload.doc_id}.index")
    }

@app.post("/ask-rag")
def ask_rag(payload: RagAskRequest):
    retrieved = retrieve_top_chunks(payload.doc_id, payload.question, payload.top_k)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No chunks retrieved. Check ingestion.")

    # Build context with chunk ids for citations
    context_blocks = []
    for r in retrieved:
        context_blocks.append(f"[CHUNK {r['chunk_id']}]\n{r['text']}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a document assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say: "Not found in the document."

Return a short, clear answer.
Also include citations like (CHUNK 3) or (CHUNK 1).

QUESTION:
{payload.question}

CONTEXT:
{context}
""".strip()

    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()

    return {
        "doc_id": payload.doc_id,
        "question": payload.question,
        "answer": data.get("response"),
        "citations": [{"chunk_id": x["chunk_id"], "score": x["score"]} for x in retrieved]
    }

def extract_json_object(text: str) -> dict:
    # remove ```json fences if model returns them
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise HTTPException(status_code=500, detail="LLM did not return a JSON object.")

    candidate = text[start:end+1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from LLM: {e}")



def heuristic_vendor_name(text: str) -> str | None:
    patterns = [
        r"Bundl Technologies Private Limited",
        r"Sold By[:\s]+(.+)",
        r"Billed By[:\s]+(.+)",
        r"From[:\s]+(.+)",
        r"Vendor[:\s]+(.+)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip() if m.groups() else m.group(0).strip()
    return None

def heuristic_currency(text: str) -> str | None:
    t = text.lower()

    # INR patterns
    if "₹" in text or " rs" in t or "rs." in t or "inr" in t:
        return "INR"

    # USD patterns
    if "$" in text or " usd" in t:
        return "USD"

    # EUR patterns
    if "€" in text or " eur" in t:
        return "EUR"

    return None


@app.post("/extract-fields")
def extract_fields(payload: FieldExtractRequest):
    try:
        if payload.doc_type.lower() != "invoice":
            raise HTTPException(status_code=400, detail="Only invoice supported for now.")

        text = load_full_text(payload.doc_id)

        # Keep prompt smaller to avoid Ollama slowdowns/timeouts
        text_for_llm = text[:3500]

        prompt = f"""
Return ONLY valid JSON.
Do NOT include markdown fences (```), and do NOT include any extra text.
Your response MUST start with {{ and end with }}.

Vendor name is the entity issuing the invoice.
It usually appears near labels like:
"Sold By", "Billed By", "From", "Supplier", "Merchant", "Company", "Vendor", "Issuer".
Do NOT extract customer name.

Fields:
- invoice_number
- invoice_date
- vendor_name
- total_amount
- currency
- line_items (array of objects with: description, quantity, unit_price, line_total)

If a field is missing, set it to null.

INVOICE TEXT:
{text_for_llm}
""".strip()

        # Retry once if Ollama times out
        last_err = None
        for attempt in range(2):
            try:
                r = requests.post(
                    OLLAMA_URL,
                    json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                    timeout=600,
                )
                r.raise_for_status()
                data = r.json()
                last_err = None
                break
            except requests.exceptions.ReadTimeout as e:
                last_err = e
                if attempt == 1:
                    raise

        raw = (data.get("response") or "").strip()
        fields = extract_json_object(raw)

        # ---------- Post-processing ----------
        # Vendor fallback
        if not fields.get("vendor_name"):
            v = heuristic_vendor_name(text)
            if v:
                fields["vendor_name"] = v

        # Currency fallback + normalization
        cur = fields.get("currency")
        if isinstance(cur, str) and cur.strip() in ["Rs.", "Rs", "INR ₹", "₹"]:
            fields["currency"] = "INR"
        elif not cur:
            inferred = heuristic_currency(text)
            if inferred:
                fields["currency"] = inferred

        # Normalize invoice_date to YYYY-MM-DD when possible
        date_val = fields.get("invoice_date")

        if isinstance(date_val, str):
            # Format: DD-MM-YYYY or DD/MM/YYYY
            m1 = re.match(r"^\s*(\d{2})[-/](\d{2})[-/](\d{4})\s*$", date_val)
            if m1:
                dd, mm, yyyy = m1.group(1), m1.group(2), m1.group(3)
                fields["invoice_date"] = f"{yyyy}-{mm}-{dd}"
            else:
                # Format: Sun, 21 Dec, 2025
                m2 = re.match(r".*?(\d{1,2})\s+([A-Za-z]{3})\s*,?\s*(\d{4})", date_val)
                if m2:
                    day = m2.group(1).zfill(2)
                    month_str = m2.group(2).lower()
                    year = m2.group(3)

                    month_map = {
                        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
                        "may": "05", "jun": "06", "jul": "07", "aug": "08",
                        "sep": "09", "oct": "10", "nov": "11", "dec": "12"
                    }

                    if month_str in month_map:
                        fields["invoice_date"] = f"{year}-{month_map[month_str]}-{day}"

        # ---------- Save prediction for evaluation ----------
        Path("eval/predictions").mkdir(parents=True, exist_ok=True)
        inv_id = fields.get("invoice_number") or payload.doc_id
        Path(f"eval/predictions/{inv_id}.json").write_text(
            json.dumps(fields, indent=2),
            encoding="utf-8"
        )

        return {"doc_id": payload.doc_id, "fields": fields}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))