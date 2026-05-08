import json
from pathlib import Path
import streamlit as st
import requests

API = "http://127.0.0.1:8000"

PDF_DIR = Path("data/invoices_pdf")
PRED_DIR = Path("eval/predictions")
GT_DIR = Path("eval/ground_truth")

PRED_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Invoice Labeler (Fast)", layout="wide")
st.title("Invoice Ground Truth Labeler — Save + Next ⚡")

# ---------- Helpers ----------
def list_pdfs():
    return sorted(PDF_DIR.rglob("*.pdf"))

def list_pred_files():
    return sorted(PRED_DIR.glob("*.json"))

def list_gt_files():
    return set(p.stem for p in GT_DIR.glob("*.json"))

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_gt(invoice_id: str, gt_obj: dict):
    out = GT_DIR / f"{invoice_id}.json"
    out.write_text(json.dumps(gt_obj, indent=2), encoding="utf-8")
    return out

def run_pipeline_on_pdf(pdf_path: Path):
    # 1) upload
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        r = requests.post(f"{API}/upload", files=files, timeout=120)
    r.raise_for_status()
    doc_id = r.json()["doc_id"]

    # 2) extract text
    r = requests.post(f"{API}/extract-text", json={"doc_id": doc_id}, timeout=180)
    r.raise_for_status()

    # 3) extract fields
    r = requests.post(
        f"{API}/extract-fields",
        json={"doc_id": doc_id, "doc_type": "invoice"},
        timeout=240
    )
    r.raise_for_status()
    resp = r.json()
    fields = resp.get("fields", {})

    # Determine prediction filename
    inv_id = fields.get("invoice_number") or doc_id

    # Save prediction
    pred_path = PRED_DIR / f"{inv_id}.json"
    pred_path.write_text(json.dumps(fields, indent=2), encoding="utf-8")
    return pred_path

def get_next_pred_to_label():
    gt_done = list_gt_files()
    pred_files = list_pred_files()
    for p in pred_files:
        if p.stem not in gt_done:
            return p
    return None


# ---------- Sidebar ----------
st.sidebar.header("Controls")

pdfs = list_pdfs()
if not pdfs:
    st.warning("No PDFs found in data/invoices_pdf. Put your PDFs there first.")
    st.stop()

st.sidebar.write(f"PDFs found: {len(pdfs)}")
st.sidebar.write(f"Predictions: {len(list_pred_files())}")
st.sidebar.write(f"Ground truth labeled: {len(list_gt_files())}")

auto_next = st.sidebar.checkbox("Auto-load next unlabeled prediction", value=True)

if "current_pred_path" not in st.session_state:
    st.session_state.current_pred_path = None

# Button: run pipeline for a chosen PDF (optional)
st.sidebar.subheader("Generate prediction (optional)")
pdf_choice = st.sidebar.selectbox("Pick a PDF to run pipeline", [p.name for p in pdfs])
pdf_path = next(p for p in pdfs if p.name == pdf_choice)

if st.sidebar.button("Run pipeline on selected PDF"):
    try:
        pred_path = run_pipeline_on_pdf(pdf_path)
        st.session_state.current_pred_path = pred_path
        st.sidebar.success(f"Prediction created: {pred_path.name}")
    except Exception as e:
        st.sidebar.error(f"Pipeline failed: {e}")

# Auto-load next unlabeled
if auto_next:
    nxt = get_next_pred_to_label()
    if nxt and (st.session_state.current_pred_path is None or st.session_state.current_pred_path != nxt):
        st.session_state.current_pred_path = nxt

pred_path = st.session_state.current_pred_path
if pred_path is None:
    st.info("No predictions to label yet. Run batch extraction or use sidebar pipeline button.")
    st.stop()

if not pred_path.exists():
    st.error("Selected prediction file not found on disk.")
    st.stop()

fields = load_json(pred_path)
current_id = pred_path.stem

# ---------- Main UI ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction (auto)")
    st.write(f"**Prediction file:** {pred_path.name}")
    st.json(fields)

with col2:
    st.subheader("Ground Truth Editor (fix values, then Save + Next)")

    inv_no = st.text_input("invoice_number", value=str(fields.get("invoice_number") or ""))
    inv_date = st.text_input("invoice_date (YYYY-MM-DD)", value=str(fields.get("invoice_date") or ""))
    vendor = st.text_input("vendor_name", value=str(fields.get("vendor_name") or ""))
    total = st.text_input("total_amount", value=str(fields.get("total_amount") or ""))
    currency = st.text_input("currency (INR/USD/EUR)", value=str(fields.get("currency") or ""))

    gt_obj = {
        "invoice_number": inv_no or None,
        "invoice_date": inv_date or None,
        "vendor_name": vendor or None,
        "total_amount": float(total) if str(total).strip() else None,
        "currency": currency or None
    }

    save_as = inv_no.strip() if inv_no.strip() else current_id
    st.caption(f"Will save as: eval/ground_truth/{save_as}.json")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Save ✅"):
            out = save_gt(save_as, gt_obj)
            st.success(f"Saved: {out.name}")

    with c2:
        if st.button("Save + Next ⚡"):
            out = save_gt(save_as, gt_obj)
            st.success(f"Saved: {out.name}")

            # move to next unlabeled prediction
            nxt = get_next_pred_to_label()
            if nxt:
                st.session_state.current_pred_path = nxt
                st.rerun()
            else:
                st.info("No more unlabeled predictions left 🎉")

    with c3:
        if st.button("Skip → Next"):
            nxt = get_next_pred_to_label()
            if nxt:
                st.session_state.current_pred_path = nxt
                st.rerun()
            else:
                st.info("No more unlabeled predictions left 🎉")