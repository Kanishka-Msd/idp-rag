# AI-Driven Intelligent Document Processing (IDP) + RAG System 🤖

An AI-powered system for processing PDF invoices using OCR, LLMs, and RAG.

## 🌐 Live Demo
- **GitHub:** https://github.com/Kanishka-Msd/idp-rag

## 🎯 What this project does
Automatically extracts structured data from PDF invoices using:
- OCR for scanned documents
- LLMs for intelligent field extraction
- RAG for document Q&A
- FAISS for semantic search

## 🏆 Results
- **Extraction Accuracy:** 100% on labeled invoices
- **Fields Extracted:** invoice_number, date, vendor, amount, currency
- **Q&A:** Grounded answers with citations

## 🛠️ Tech Stack

| Category | Tool |
|----------|------|
| Backend API | FastAPI |
| LLM | Llama3 via Ollama |
| OCR | pytesseract + pdfplumber |
| Embeddings | SentenceTransformers |
| Vector DB | FAISS |
| Language | Python |

## 📁 Project Structure

idp-rag/
├── app/
│   └── main.py          # FastAPI endpoints
├── eval/
│   ├── evaluate.py      # Accuracy evaluation
│   ├── ground_truth/    # Labeled invoices
│   └── predictions/     # Model outputs
├── storage/
│   ├── uploads/         # PDF files
│   ├── text/            # Extracted text
│   └── faiss/           # Vector indexes
└── tools/               # Utility scripts

## 🚀 API Endpoints

| Endpoint | Purpose |
|----------|---------|
| POST /upload | Upload PDF invoice |
| POST /extract-text | OCR + text extraction |
| POST /extract-fields | LLM field extraction |
| POST /ingest | Generate embeddings + FAISS |
| POST /ask-rag | Grounded Q&A |

## 🔄 Pipeline Flow

Upload PDF → OCR/Parse → Chunk Text →
Generate Embeddings → FAISS Index →
LLM Extraction → Structured Output

## 🚀 Quick Start
```bash
git clone https://github.com/Kanishka-Msd/idp-rag
cd idp-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 📈 Next Steps
- [ ] Docker containerization
- [ ] GitHub Actions CI/CD
- [ ] Cloud deployment
- [ ] Batch processing pipeline
- [ ] Fine-tune extraction model

## 👨‍💻 Author
**Kanishka** — Junior ML/AI Engineer
- GitHub: https://github.com/Kanishka-Msd
- MLOps Platform: https://github.com/Kanishka-Msd/mlops-platform

