"""
Financial RAG Assistant - Standalone Streamlit App
API key loaded from .env file only — no sidebar input needed.
Run: streamlit run frontend/streamlit_app.py
"""

import os, re, io, sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load .env from project root
load_dotenv(os.path.join(ROOT, ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="💰",
    layout="wide",
)

st.markdown("""
<style>
.stTabs [data-baseweb="tab"] { font-size:16px; padding:10px 28px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "messages": [], "chunks": [], "doc_names": [],
    "bm25_index": None, "indexed": False, "prefill": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  — no API key input, just settings
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("💰 Financial RAG")
    st.caption("Powered by Groq (Free) · No server needed")
    st.divider()

    # Show key status (loaded from .env)
    if GROQ_API_KEY and "your_groq" not in GROQ_API_KEY:
        st.success("✅ Groq API Key loaded from `.env`")
    else:
        st.error("❌ Groq API Key not found in `.env`")
        st.markdown("""
**To fix:**
1. Open the `.env` file in the project root
2. Set your key:
```
GROQ_API_KEY=gsk_your_key_here
```
3. Get a free key at [console.groq.com](https://console.groq.com)
4. Restart the app
        """)

    st.divider()
    st.subheader("⚙️ Settings")
    top_k         = st.slider("Top K chunks",   1, 15, 5)
    chunk_size    = st.slider("Chunk size",    200, 2000, 500, 100)
    chunk_overlap = st.slider("Chunk overlap",   0,  300,  50,  25)
    show_sources  = st.toggle("Show source excerpts", value=True)

    st.divider()
    st.subheader("📋 Sample Questions")
    for s in [
        "What was total revenue?",
        "What was net income?",
        "What were the key risk factors?",
        "What was earnings per share?",
        "What are the debt obligations?",
        "What was the operating margin?",
        "Summarize the document",
        "What is this document about?",
        "What are the main features?",
    ]:
        if st.button(s, use_container_width=True, key=f"sq_{s}"):
            st.session_state.prefill = s
            st.rerun()

    st.divider()
    if st.session_state.doc_names:
        st.subheader("📁 Loaded Files")
        for n in st.session_state.doc_names:
            st.caption(f"📄 {n}")
        st.caption(f"🔢 {len(st.session_state.chunks)} chunks indexed")

    if st.button("🗑️ Clear everything", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.chunks     = []
        st.session_state.doc_names  = []
        st.session_state.bm25_index = None
        st.session_state.indexed    = False
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("💰 Financial RAG Assistant")
st.caption("Upload financial documents · Ask questions · Get grounded AI answers · Powered by Groq (Free)")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clean(text):
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.replace("\x00", "").strip()


def extract_pdf(file_bytes, filename):
    company = filename.replace(".pdf","").split("_")[0]

    # Method 1: pdfplumber (best)
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                for table in page.extract_tables():
                    for row in table:
                        row_text = " | ".join([str(c) for c in row if c])
                        if row_text.strip():
                            text += "\n" + row_text
                text = clean(text)
                if text.strip():
                    pages.append({"text": text, "source": filename,
                                  "page": i+1, "company": company, "type": "pdf"})
        if pages and sum(len(p["text"]) for p in pages) > 500:
            return pages
    except ImportError:
        pass

    # Method 2: pypdf layout mode
    import pypdf
    pages = []
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    for i, page in enumerate(reader.pages):
        text = ""
        try:    text = page.extract_text(extraction_mode="layout") or ""
        except: text = page.extract_text() or ""
        text = clean(text)
        if text.strip():
            pages.append({"text": text, "source": filename,
                          "page": i+1, "company": company, "type": "pdf"})
    if pages and sum(len(p["text"]) for p in pages) > 500:
        return pages

    # Method 3: full text block fallback
    reader    = pypdf.PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for i, page in enumerate(reader.pages):
        try:    t = page.extract_text(extraction_mode="layout") or ""
        except: t = page.extract_text() or ""
        if t.strip():
            full_text += f"\n\n[Page {i+1}]\n{t}"
    full_text = clean(full_text)
    if full_text.strip():
        return [{"text": full_text, "source": filename,
                 "page": 0, "company": company, "type": "pdf_full"}]
    return []


def extract_txt(file_bytes, filename):
    text = clean(file_bytes.decode("utf-8", errors="ignore"))
    return [{"text": text, "source": filename, "page": 1,
             "company": filename.split("_")[0], "type": "earnings_call"}]


def extract_csv(file_bytes, filename):
    import pandas as pd
    df = pd.read_csv(io.BytesIO(file_bytes))
    return [{"text": f"Headline: {row.get('headline', str(row.iloc[0]))}",
             "source": filename, "page": 1, "company": "news", "type": "market_news"}
            for _, row in df.iterrows()]


def make_chunks(pages, size=500, overlap=50):
    chunks = []
    for page in pages:
        text = page["text"]
        if not text.strip():
            continue
        paragraphs = re.split(r"\n\n+", text)
        current, idx = "", 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 2 <= size:
                current = (current + "\n\n" + para).strip()
            else:
                if current.strip():
                    chunks.append({**page, "text": current.strip(), "chunk_idx": idx})
                    idx += 1
                    current = (current[-overlap:] + "\n\n" + para).strip() if overlap else para
                else:
                    for sent in re.split(r"(?<=[.!?])\s+", para):
                        if len(current) + len(sent) + 1 <= size:
                            current = (current + " " + sent).strip()
                        else:
                            if current.strip():
                                chunks.append({**page, "text": current.strip(), "chunk_idx": idx})
                                idx += 1
                            current = sent.strip()
        if current.strip():
            chunks.append({**page, "text": current.strip(), "chunk_idx": idx})
    return chunks


def build_bm25(chunks):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([c["text"].lower().split() for c in chunks])


def bm25_retrieve(query, chunks, bm25_index, k=5):
    scores = bm25_index.get_scores(query.lower().split())
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top    = [c for c, _ in ranked[:k]]
    # Keyword fallback if BM25 scores are all near zero
    if all(s < 0.01 for _, s in ranked[:k]):
        keywords = [w for w in query.lower().split() if len(w) > 3]
        fallback = sorted(
            [(c, sum(1 for kw in keywords if kw in c["text"].lower())) for c in chunks],
            key=lambda x: x[1], reverse=True
        )
        top = [c for c, hits in fallback[:k] if hits > 0] or top
    return top


def generate_answer(question, chunks, api_key):
    from groq import Groq
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']} | Page: {c.get('page','?')} | Company: {c['company']}]\n{c['text']}"
        for c in chunks
    ])
    prompt = f"""You are a Senior Financial Analyst and document expert.
Read the context carefully and answer the question thoroughly.

RULES:
1. Use ONLY the context below. Do not use outside knowledge.
2. If the answer IS in the context, give a detailed answer with exact figures.
3. If the answer is NOT in the context, say: "I could not find this in the uploaded documents."
4. Quote exact numbers and figures as they appear.
5. Mention the source document and page number.
6. Be detailed and professional.

Context:
{context}

Question: {question}

**Answer:**

**Key Insights:**

**Sources:**
"""
    resp = Groq(api_key=api_key).chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1500,
    )
    return resp.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📂  Upload Documents", "💬  Ask Questions"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📤 Upload Your Financial Documents")
    st.markdown("Upload **PDF reports, TXT earnings calls, or CSV news files**. Indexed instantly — no server required.")

    with st.expander("💡 For better PDF extraction (recommended)"):
        st.code("pip install pdfplumber", language="bash")
        st.caption("Handles tables, multi-column layouts, and complex PDFs. Restart app after installing.")

    uploaded = st.file_uploader(
        "📁 Drag & drop files here, or click Browse files",
        type=["pdf","txt","csv","md"],
        accept_multiple_files=True,
    )

    with st.expander("ℹ️ Supported formats"):
        st.markdown("""
| Format | Purpose |
|--------|---------|
| `.pdf` | Annual reports, financial statements |
| `.txt` | Earnings call transcripts |
| `.md`  | Markdown reports |
| `.csv` | Market news (`headline,date` columns) |
        """)

    if uploaded:
        st.success(f"**{len(uploaded)} file(s) selected:**")
        for f in uploaded:
            st.markdown(f"&nbsp;&nbsp;📄 `{f.name}` — {round(len(f.getvalue())/1024,1)} KB")

        if st.button("🚀 Process & Index All Files", type="primary"):
            all_pages, errors = [], []
            prog = st.progress(0, "Reading files...")

            for i, uf in enumerate(uploaded):
                prog.progress((i+0.3)/len(uploaded), f"Extracting `{uf.name}`...")
                try:
                    raw = uf.read()
                    ext = os.path.splitext(uf.name)[1].lower()
                    if   ext == ".pdf":          pages = extract_pdf(raw, uf.name)
                    elif ext in (".txt",".md"):  pages = extract_txt(raw, uf.name)
                    elif ext == ".csv":          pages = extract_csv(raw, uf.name)
                    else:
                        errors.append(f"`{uf.name}`: unsupported"); continue

                    if not pages:
                        errors.append(f"`{uf.name}`: no text extracted — try converting to .txt"); continue

                    total_chars = sum(len(p["text"]) for p in pages)
                    st.write(f"✅ `{uf.name}`: {len(pages)} page(s), **{total_chars:,} characters**")
                    all_pages.extend(pages)

                    dest = {".pdf":"annual_reports",".txt":"earnings_calls",
                            ".md":"earnings_calls",".csv":"market_news"}[ext]
                    save_dir = os.path.join(ROOT,"data","raw",dest)
                    os.makedirs(save_dir, exist_ok=True)
                    with open(os.path.join(save_dir, uf.name),"wb") as fh:
                        fh.write(raw)
                    if uf.name not in st.session_state.doc_names:
                        st.session_state.doc_names.append(uf.name)
                    prog.progress((i+1)/len(uploaded), f"✅ `{uf.name}` done")
                except Exception as e:
                    errors.append(f"`{uf.name}`: {e}")

            if all_pages:
                total_chars = sum(len(p["text"]) for p in all_pages)
                prog.progress(0.85, "Chunking...")
                new_chunks = make_chunks(all_pages, chunk_size, chunk_overlap)
                st.session_state.chunks.extend(new_chunks)
                prog.progress(0.95, "Building BM25 index...")
                st.session_state.bm25_index = build_bm25(st.session_state.chunks)
                st.session_state.indexed    = True
                prog.progress(1.0, "Done!")
                prog.empty()
                st.success(
                    f"✅ **Indexed!** {total_chars:,} chars → "
                    f"**{len(st.session_state.chunks)} chunks** from **{len(uploaded)-len(errors)} file(s)**"
                )
                for e in errors: st.error(f"❌ {e}")
                if len(st.session_state.chunks) < 10:
                    st.warning("⚠️ Very few chunks. Install `pdfplumber` and re-upload, or convert PDF to .txt")
                else:
                    st.info("👉 Switch to **💬 Ask Questions** tab to start querying.")
                    st.balloons()
            else:
                prog.empty()
                for e in errors: st.error(f"❌ {e}")

    st.markdown("---")
    st.subheader("📁 Currently Indexed Documents")
    if not st.session_state.doc_names:
        st.info("No documents yet. Upload files above.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("📄 Files",  len(st.session_state.doc_names))
        c2.metric("🔢 Chunks", len(st.session_state.chunks))
        c3.metric("✅ Status", "Ready ✅" if st.session_state.indexed else "Not ready")

        if st.session_state.chunks:
            with st.expander("🔍 Preview extracted chunks"):
                for i, chunk in enumerate(st.session_state.chunks[:5]):
                    st.markdown(f"**Chunk {i+1}** | `{chunk['source']}` | Page {chunk.get('page','?')}")
                    st.caption(chunk["text"][:300] + "...")
                    st.divider()

        for name in list(st.session_state.doc_names):
            ca, cb = st.columns([6,1])
            ca.markdown(f"📄 `{name}`")
            if cb.button("🗑️", key=f"del_{name}"):
                st.session_state.doc_names.remove(name)
                st.session_state.chunks = [c for c in st.session_state.chunks if c["source"] != name]
                st.session_state.bm25_index = build_bm25(st.session_state.chunks) if st.session_state.chunks else None
                st.session_state.indexed = bool(st.session_state.chunks)
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("💬 Ask Questions About Your Documents")

    if not GROQ_API_KEY or "your_groq" in GROQ_API_KEY:
        st.error("❌ Groq API key not found in `.env` file.")
        st.markdown("""
**Steps to fix:**
1. Open `financial-rag-system/.env`
2. Add your key:
```
GROQ_API_KEY=gsk_your_actual_key_here
```
3. Get a **free** key at 👉 https://console.groq.com
4. Save the file and restart the app
        """)
        st.stop()

    if not st.session_state.indexed:
        st.warning("⚠️ Upload documents first in the **📂 Upload Documents** tab.")
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and show_sources:
                with st.expander("📄 Source Chunks"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['source']}** | Page {s.get('page','?')} | `{s['company']}`")
                        st.caption(s["excerpt"])
                        st.divider()

    prefill = st.session_state.get("prefill","")
    if prefill:
        st.session_state.prefill = ""

    question = st.chat_input("Ask a financial question about your documents...") or prefill

    if question:
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    top_chunks = bm25_retrieve(question, st.session_state.chunks,
                                               st.session_state.bm25_index, k=top_k)
                    answer  = generate_answer(question, top_chunks, GROQ_API_KEY)
                    sources = [{"source":c["source"],"company":c["company"],
                                "page":c.get("page","?"),"excerpt":c["text"][:300]}
                               for c in top_chunks]
                    st.markdown(answer)
                    if sources and show_sources:
                        with st.expander("📄 Source Chunks"):
                            for s in sources:
                                st.markdown(f"**{s['source']}** | Page {s['page']} | `{s['company']}`")
                                st.caption(s["excerpt"])
                                st.divider()
                    st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})
                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    if "401" in str(e) or "auth" in str(e).lower():
                        st.info("💡 Check your GROQ_API_KEY in the .env file.")
                    elif "429" in str(e):
                        st.info("💡 Rate limit — wait a few seconds and try again.")
                    st.session_state.messages.append({"role":"assistant","content":err})
