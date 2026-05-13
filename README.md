---
title: IDP RAG
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AI-Driven Intelligent Document Processing (IDP) + RAG System 🤖

An AI-powered system for processing ANY PDF document using OCR, LLMs, and RAG.

## 🌐 Live Demo

| Link | Description |
|------|-------------|
| [**Live API**](https://kanishka76-idp-rag.hf.space/docs) | Interactive Swagger UI |
| [**GitHub**](https://github.com/Kanishka-Msd/idp-rag) | Source code |

## 🎯 What this project does

Automatically processes any PDF document:
- **OCR** for scanned/image documents
- **LLMs** for intelligent field extraction
- **RAG** for document Q&A with citations
- **FAISS** for semantic vector search
- **MLflow** for experiment tracking

## 🏆 Results

| Metric | Value |
|--------|-------|
| Extraction Accuracy | 100% on labeled documents |
| Fields Extracted | invoice_number, date, vendor, amount, currency |
| RAG Response Time | < 3 seconds |
| Supported Documents | Invoices, contracts, policies, reports |

## 🤖 Smart Model Routing

```
Short/Simple docs  →  llama-3.1-8b-instant  (560 tok/sec) ⚡
Complex/Large docs →  llama-3.3-70b-versatile (280 tok/sec) 🧠
```

## 🛠️ Tech Stack

| Category | Tool |
|----------|------|
| Backend API | FastAPI |
| LLM Provider | Groq (Llama3) |
| OCR | pytesseract + pdfplumber |
| Embeddings | SentenceTransformers |
| Vector DB | FAISS |
| UI | Streamlit |
| Tracking | MLflow |
| Container | Docker |
| CI/CD | GitHub Actions |

## 📁 Project Structure

```
idp-rag/
├── app/
│   ├── main.py               # FastAPI endpoints + MLflow tracking
│   └── streamlit_app.py      # Beautiful web UI
├── eval/
│   ├── evaluate.py           # Accuracy evaluation pipeline
│   ├── ground_truth/         # Labeled invoice data
│   └── predictions/          # Model extraction outputs
├── storage/
│   ├── uploads/              # Uploaded PDF files
│   ├── text/                 # Extracted text + chunks
│   └── faiss/                # Vector similarity indexes
├── tools/                    # Utility scripts
├── Dockerfile                # Container definition
├── docker-compose.yml        # Multi-service setup
├── requirements.txt          # Python dependencies
└── .github/
    └── workflows/
        └── idp-pipeline.yml  # CI/CD pipeline
```

## 🚀 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System health check |
| `/upload` | POST | Upload PDF document |
| `/extract-text` | POST | OCR + text extraction |
| `/ingest` | POST | Generate embeddings + FAISS index |
| `/extract-fields` | POST | LLM field extraction |
| `/ask-rag` | POST | Grounded document Q&A |
| `/ask` | POST | General LLM Q&A |

## 🔄 Pipeline Flow

```
Upload PDF
    ↓
OCR / PDF Parser
    ↓
Text Chunking (800 chars, 120 overlap)
    ↓
Embeddings (all-MiniLM-L6-v2)
    ↓
FAISS Vector Index
    ↓
┌─────────────────────────┐
│  LLM Field Extraction   │  → invoice_number, vendor, amount...
│  RAG Q&A                │  → grounded answers with citations
└─────────────────────────┘
    ↓
MLflow Tracking
```

## 🚀 Quick Start

```bash
git clone https://github.com/Kanishka-Msd/idp-rag
cd idp-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"
uvicorn app.main:app --reload
```

Visit: `http://localhost:8000/docs`

## 🐳 Docker

```bash
docker build -t idp-rag .
docker run -p 8000:8000 -p 8501:8501 \
  -e GROQ_API_KEY=your-key-here \
  idp-rag
```

## 👨‍💻 Author

**Kanishka** — Junior ML/AI Engineer
- MLOps Platform: [github.com/Kanishka-Msd/mlops-platform](https://github.com/Kanishka-Msd/mlops-platform)
- IDP System: [github.com/Kanishka-Msd/idp-rag](https://github.com/Kanishka-Msd/idp-rag)

