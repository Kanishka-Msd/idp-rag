import streamlit as st
import requests
import json

# ─── Config ───
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="IDP + RAG System",
    page_icon="🤖",
    layout="wide"
)

# ─── Header ───
st.title("🤖 AI Document Processing System")
st.markdown("**Upload any PDF → Extract fields → Ask questions**")
st.divider()

# ─── Sidebar ───
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.header("📋 How to use")
    st.markdown("""
    1. **Upload** your PDF document
    2. **Extract** text from document
    3. **Ingest** for RAG search
    4. **Extract Fields** automatically
    5. **Ask Questions** about document
    """)
    st.divider()
    st.header("🤖 Model Routing")
    st.markdown("""
    - **Short docs** → llama-3.1-8b-instant ⚡
    - **Complex docs** → llama-3.3-70b-versatile 🧠
    """)
    st.divider()

    # Health check
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")

# ─── Session State ───
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "extracted" not in st.session_state:
    st.session_state.extracted = False
if "ingested" not in st.session_state:
    st.session_state.ingested = False

# ─── Step 1: Upload ───
st.header("📤 Step 1: Upload Document")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload any PDF — invoice, contract, report, policy"
)

if uploaded_file:
    if st.button("🚀 Upload Document", type="primary"):
        with st.spinner("Uploading..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            r = requests.post(f"{API_URL}/upload", files=files)
            if r.status_code == 200:
                data = r.json()
                st.session_state.doc_id = data["doc_id"]
                st.session_state.extracted = False
                st.session_state.ingested = False
                st.success(f"✅ Uploaded! Doc ID: `{data['doc_id']}`")
                st.json(data)
            else:
                st.error("Upload failed!")

st.divider()

# ─── Step 2: Extract Text ───
st.header("📝 Step 2: Extract Text")

if st.session_state.doc_id:
    if st.button("🔍 Extract Text", type="primary"):
        with st.spinner("Extracting text..."):
            r = requests.post(
                f"{API_URL}/extract-text",
                json={"doc_id": st.session_state.doc_id}
            )
            if r.status_code == 200:
                data = r.json()
                st.session_state.extracted = True
                st.success(f"✅ Extracted {data['text_length']:,} characters!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Method", data["method"])
                col2.metric("Characters", f"{data['text_length']:,}")
                col3.metric("Chunks", data["num_chunks"])
                st.info(f"🤖 Will use: **{data['model_will_use']}**")
                with st.expander("Preview extracted text"):
                    st.text(data["first_chunk_preview"])
            else:
                st.error("Extraction failed!")
else:
    st.warning("⚠️ Upload a document first!")

st.divider()

# ─── Step 3: Ingest ───
st.header("🧠 Step 3: Ingest for RAG")

if st.session_state.extracted:
    if st.button("⚡ Create Embeddings + Index", type="primary"):
        with st.spinner("Creating embeddings and FAISS index..."):
            r = requests.post(
                f"{API_URL}/ingest",
                json={"doc_id": st.session_state.doc_id}
            )
            if r.status_code == 200:
                data = r.json()
                st.session_state.ingested = True
                st.success(f"✅ Indexed {data['num_chunks_indexed']} chunks!")
                st.json(data)
            else:
                st.error("Ingestion failed!")
elif not st.session_state.doc_id:
    st.warning("⚠️ Upload a document first!")
else:
    st.warning("⚠️ Extract text first!")

st.divider()

# ─── Step 4: Extract Fields ───
st.header("📊 Step 4: Extract Fields")

if st.session_state.extracted:
    doc_type = st.selectbox(
        "Document type",
        ["invoice", "contract", "report", "policy"],
        help="Select the type of document"
    )
    if st.button("🤖 Extract Fields with AI", type="primary"):
        with st.spinner("AI is extracting fields..."):
            r = requests.post(
                f"{API_URL}/extract-fields",
                json={
                    "doc_id": st.session_state.doc_id,
                    "doc_type": doc_type
                }
            )
            if r.status_code == 200:
                data = r.json()
                st.success(f"✅ Fields extracted using **{data['model_used']}**!")
                fields = data["fields"]

                # Show fields nicely
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📋 Extracted Fields")
                    for key, value in fields.items():
                        if key != "line_items" and value:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

                with col2:
                    if fields.get("line_items"):
                        st.subheader("📦 Line Items")
                        for item in fields["line_items"]:
                            st.markdown(f"- **{item.get('description', 'N/A')}**: {item.get('line_total', 'N/A')}")

                with st.expander("View raw JSON"):
                    st.json(fields)
            else:
                st.error(f"Extraction failed! {r.text}")
elif not st.session_state.doc_id:
    st.warning("⚠️ Upload a document first!")
else:
    st.warning("⚠️ Extract text first!")

st.divider()

# ─── Step 5: RAG Q&A ───
st.header("💬 Step 5: Ask Questions")

if st.session_state.ingested:
    question = st.text_input(
        "Ask anything about your document",
        placeholder="What is the total amount? Who is the vendor? What are the terms?"
    )
    top_k = st.slider("Number of chunks to retrieve", 2, 8, 4)

    if st.button("🔍 Ask", type="primary") and question:
        with st.spinner("Searching and generating answer..."):
            r = requests.post(
                f"{API_URL}/ask-rag",
                json={
                    "doc_id": st.session_state.doc_id,
                    "question": question,
                    "top_k": top_k
                }
            )
            if r.status_code == 200:
                data = r.json()
                st.success("✅ Answer found!")
                st.subheader("🤖 Answer")
                st.markdown(data["answer"])
                st.info(f"Model used: **{data['model_used']}**")

                with st.expander("View citations"):
                    for c in data["citations"]:
                        st.markdown(f"- Chunk {c['chunk_id']} (score: {c['score']:.3f})")
            else:
                st.error("RAG failed!")

    # Quick questions
    st.subheader("💡 Quick Questions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What is the total amount?"):
            st.session_state.quick_q = "What is the total amount?"
    with col2:
        if st.button("Who is the vendor?"):
            st.session_state.quick_q = "Who is the vendor?"
    with col3:
        if st.button("What is the date?"):
            st.session_state.quick_q = "What is the date?"

elif not st.session_state.doc_id:
    st.warning("⚠️ Upload a document first!")
elif not st.session_state.extracted:
    st.warning("⚠️ Extract text first!")
else:
    st.warning("⚠️ Ingest document first!")

st.divider()
st.markdown("Built with ❤️ using FastAPI + Groq + FAISS + Streamlit")