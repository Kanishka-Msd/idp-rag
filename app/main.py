from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
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
import mlflow
import time
from sentence_transformers import SentenceTransformer
from groq import Groq

app = FastAPI(
    title="IDP + RAG System",
    description="""
## AI-Driven Intelligent Document Processing

Processes any PDF document using OCR, LLMs, and RAG.

### Features
- PDF upload and text extraction
- OCR for scanned documents  
- LLM-powered field extraction
- RAG-based document Q&A
- Semantic search with FAISS
- MLflow experiment tracking

### Model Routing
- Simple documents → llama-3.1-8b-instant (560 tok/sec)
- Complex documents → llama-3.3-70b-versatile (280 tok/sec)
    """,
    version="2.0.0"
)

# ─── MLflow Setup ───
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("idp-rag-tracking")

# ─── Groq Client Setup ───
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")
groq_client = Groq(api_key=GROQ_API_KEY)

# Model routing
FAST_MODEL = "llama-3.1-8b-instant"
SMART_MODEL = "llama-3.3-70b-versatile"
COMPLEXITY_THRESHOLD = 2000

def get_model(text_length: int, task: str = "extract") -> str:
    if task == "rag" or text_length > COMPLEXITY_THRESHOLD:
        return SMART_MODEL
    return FAST_MODEL

def call_groq(prompt: str, text_length: int = 0, task: str = "extract") -> str:
    model = get_model(text_length, task)
    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2048,
    )
    return response.choices[0].message.content, model

# ─── Embeddings + Storage Setup ───
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

FAISS_DIR = Path("storage/faiss")
FAISS_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TEXT_DIR = Path("storage/text")
TEXT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Request Models ───
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
    doc_type: str = "invoice"


# ─── Health Check ───
@app.get("/health")
def health_check():
    return {
        "status": "running",
        "version": "2.0.0",
        "llm_provider": "Groq",
        "fast_model": FAST_MODEL,
        "smart_model": SMART_MODEL,
        "embedding_model": EMBED_MODEL_NAME
    }


# ─── General Q&A ───
@app.post("/ask")
def ask_question(payload: AskRequest):
    answer, model_used = call_groq(payload.question, task="simple")
    return {
        "model_used": model_used,
        "answer": answer
    }


# ─── PDF Upload ───
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf"]:
        return {"error": "Only PDF files are allowed."}

    doc_id = str(uuid.uuid4())
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


# ─── Helper Functions ───
def find_pdf_by_doc_id(doc_id: str) -> Path:
    matches = list(UPLOAD_DIR.glob(f"{doc_id}_*.pdf"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF found for doc_id={doc_id}. Use /upload first."
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
        raise HTTPException(
            status_code=404,
            detail="Extracted text not found. Run /extract-text first."
        )
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
        raise HTTPException(
            status_code=404,
            detail="Chunks not found. Run /extract-text first."
        )
    raw = chunks_path.read_text(encoding="utf-8", errors="ignore")
    chunks = [c.strip() for c in raw.split("\n\n---CHUNK---\n\n") if c.strip()]
    return chunks


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
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
        raise HTTPException(
            status_code=404,
            detail="FAISS index not found. Run /ingest first."
        )
    return faiss.read_index(str(index_path))


def retrieve_top_chunks(doc_id: str, question: str, top_k: int = 4) -> list[dict]:
    chunks = load_chunks_for_doc(doc_id)
    index = load_faiss_index_for_doc(doc_id)
    q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
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


def extract_json_object(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise HTTPException(
            status_code=500,
            detail="LLM did not return a JSON object."
        )
    candidate = text[start:end+1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON from LLM: {e}"
        )


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
    if "₹" in text or " rs" in t or "rs." in t or "inr" in t:
        return "INR"
    if "$" in text or " usd" in t:
        return "USD"
    if "€" in text or " eur" in t:
        return "EUR"
    return None


# ─── Extract Text ───
@app.post("/extract-text")
def extract_text(payload: DocRequest):
    start_time = time.time()
    pdf_path = find_pdf_by_doc_id(payload.doc_id)
    text = extract_text_pdfplumber(pdf_path)
    method = "pdf_text"

    if len(text) < 200:
        text = extract_text_ocr(pdf_path)
        method = "ocr"

    text_path = TEXT_DIR / f"{payload.doc_id}.txt"
    text_path.write_text(text, encoding="utf-8", errors="ignore")

    chunks = chunk_text(text, chunk_size=800, overlap=120)
    chunks_path = TEXT_DIR / f"{payload.doc_id}.chunks.txt"
    chunks_path.write_text(
        "\n\n---CHUNK---\n\n".join(chunks),
        encoding="utf-8",
        errors="ignore"
    )

    duration = time.time() - start_time

    # Track with MLflow
    with mlflow.start_run(run_name=f"extract-{payload.doc_id[:8]}"):
        mlflow.log_param("doc_id", payload.doc_id)
        mlflow.log_param("extraction_method", method)
        mlflow.log_metric("text_length", len(text))
        mlflow.log_metric("num_chunks", len(chunks))
        mlflow.log_metric("extraction_time_seconds", duration)

    return {
        "doc_id": payload.doc_id,
        "method": method,
        "text_length": len(text),
        "model_will_use": get_model(len(text)),
        "num_chunks": len(chunks),
        "first_chunk_preview": chunks[0][:500] if chunks else ""
    }


# ─── Ingest ───
@app.post("/ingest")
def ingest_doc(payload: DocRequest):
    start_time = time.time()
    chunks = load_chunks_for_doc(payload.doc_id)
    vectors = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
    index = build_faiss_index(vectors)
    save_faiss_for_doc(payload.doc_id, index, chunks)
    duration = time.time() - start_time

    # Track with MLflow
    with mlflow.start_run(run_name=f"ingest-{payload.doc_id[:8]}"):
        mlflow.log_param("doc_id", payload.doc_id)
        mlflow.log_param("embedding_model", EMBED_MODEL_NAME)
        mlflow.log_metric("num_chunks_indexed", len(chunks))
        mlflow.log_metric("ingestion_time_seconds", duration)

    return {
        "doc_id": payload.doc_id,
        "num_chunks_indexed": len(chunks),
        "embedding_model": EMBED_MODEL_NAME,
        "faiss_index_saved_to": str(FAISS_DIR / f"{payload.doc_id}.index")
    }


# ─── RAG Q&A ───
@app.post("/ask-rag")
def ask_rag(payload: RagAskRequest):
    start_time = time.time()
    retrieved = retrieve_top_chunks(payload.doc_id, payload.question, payload.top_k)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No chunks retrieved.")

    context_blocks = []
    for r in retrieved:
        context_blocks.append(f"[CHUNK {r['chunk_id']}]\n{r['text']}")
    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a document assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say: "Not found in the document."
Return a short, clear answer with citations like (CHUNK 3).

QUESTION:
{payload.question}

CONTEXT:
{context}
""".strip()

    answer, model_used = call_groq(prompt, text_length=len(context), task="rag")
    duration = time.time() - start_time

    avg_score = sum(r["score"] for r in retrieved) / len(retrieved) if retrieved else 0

    # Track with MLflow
    with mlflow.start_run(run_name=f"rag-{payload.doc_id[:8]}"):
        mlflow.log_param("doc_id", payload.doc_id)
        mlflow.log_param("model_used", model_used)
        mlflow.log_metric("top_k", payload.top_k)
        mlflow.log_metric("chunks_retrieved", len(retrieved))
        mlflow.log_metric("avg_relevance_score", avg_score)
        mlflow.log_metric("response_time_seconds", duration)

    return {
        "doc_id": payload.doc_id,
        "question": payload.question,
        "answer": answer,
        "model_used": model_used,
        "citations": [{"chunk_id": x["chunk_id"], "score": x["score"]} for x in retrieved]
    }


# ─── Field Extraction ───
@app.post("/extract-fields")
def extract_fields(payload: FieldExtractRequest):
    try:
        start_time = time.time()
        text = load_full_text(payload.doc_id)
        text_for_llm = text[:4000]

        prompt = f"""
Return ONLY valid JSON. No markdown, no extra text.
Your response MUST start with {{ and end with }}.

Extract these fields from the document:
- invoice_number (or document number)
- invoice_date (normalize to YYYY-MM-DD)
- vendor_name (the entity issuing the document)
- total_amount (numeric value)
- currency (ISO code: USD, INR, EUR etc.)
- line_items (array of: description, quantity, unit_price, line_total)

Vendor name appears near: "Sold By", "Billed By", "From", "Supplier", "Merchant", "Vendor"
If a field is missing, set it to null.

DOCUMENT TEXT:
{text_for_llm}
""".strip()

        answer, model_used = call_groq(prompt, text_length=len(text))
        fields = extract_json_object(answer)

        # Post-processing
        if not fields.get("vendor_name"):
            v = heuristic_vendor_name(text)
            if v:
                fields["vendor_name"] = v

        cur = fields.get("currency")
        if isinstance(cur, str) and cur.strip() in ["Rs.", "Rs", "INR ₹", "₹"]:
            fields["currency"] = "INR"
        elif not cur:
            inferred = heuristic_currency(text)
            if inferred:
                fields["currency"] = inferred

        date_val = fields.get("invoice_date")
        if isinstance(date_val, str):
            m1 = re.match(r"^\s*(\d{2})[-/](\d{2})[-/](\d{4})\s*$", date_val)
            if m1:
                dd, mm, yyyy = m1.group(1), m1.group(2), m1.group(3)
                fields["invoice_date"] = f"{yyyy}-{mm}-{dd}"
            else:
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

        duration = time.time() - start_time
        fields_extracted = len([v for v in fields.values() if v and v != []])

        # Track with MLflow
        with mlflow.start_run(run_name=f"extract-fields-{payload.doc_id[:8]}"):
            mlflow.log_param("doc_id", payload.doc_id)
            mlflow.log_param("doc_type", payload.doc_type)
            mlflow.log_param("model_used", model_used)
            mlflow.log_metric("text_length", len(text))
            mlflow.log_metric("fields_extracted", fields_extracted)
            mlflow.log_metric("has_vendor", 1 if fields.get("vendor_name") else 0)
            mlflow.log_metric("has_amount", 1 if fields.get("total_amount") else 0)
            mlflow.log_metric("has_date", 1 if fields.get("invoice_date") else 0)
            mlflow.log_metric("extraction_time_seconds", duration)

        # Save prediction
        Path("eval/predictions").mkdir(parents=True, exist_ok=True)
        inv_id = fields.get("invoice_number") or payload.doc_id
        Path(f"eval/predictions/{inv_id}.json").write_text(
            json.dumps(fields, indent=2), encoding="utf-8"
        )

        return {
            "doc_id": payload.doc_id,
            "model_used": model_used,
            "fields": fields
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))