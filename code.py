import os
import io
import re
import time
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from typing import List, Dict, Tuple

# Embeddings & Vector search
from sentence_transformers import SentenceTransformer
import faiss

# --- UI ---
st.set_page_config(page_title="StudyMate — PDF Q&A", page_icon="📚", layout="wide")
st.title("📚 StudyMate — AI PDF Q&A for Students")
st.caption("Upload PDFs → Ask a question → Get answers grounded in your material.")

# --- Caching heavy objects ---
@st.cache_resource(show_spinner=False)
def load_embedder():
    # Small, fast, good quality
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_resource(show_spinner=False)
def init_faiss(dim: int):
    index = faiss.IndexFlatIP(dim)  # cosine via normalized dot product
    return index

# --- Helpers ---
def read_pdf_bytes(file_bytes: bytes) -> Tuple[str, List[int]]:
    """Return full text and a list mapping chunk index to page number."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    texts = []
    for i, page in enumerate(doc):
        t = page.get_text("text")
        if t:
            texts.append(t)
            pages.extend([i + 1] * len(t))  # coarse map (refined later)
    return "\n".join(texts), list(range(1, len(texts) + 1))

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200) -> List[str]:
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        # try neat sentence boundary cut
        if end < len(text):
            m = re.search(r"[\.!?]\s", text[end:end+120])
            if m:
                end = end + m.end()
                chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, end) if end >= len(text) else end - overlap
        if start < 0: start = 0
        if end == len(text): break
    return chunks

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")

def call_llm(prompt: str) -> str:
    # Prefer Groq (fast + free dev tier)
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are StudyMate, a helpful academic assistant. Use ONLY the provided context. If unsure, say you don't know."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            return resp.choices[0].message.content
    except Exception as e:
        st.warning(f"Groq error: {e}")

    # Fallback: OpenAI if available
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are StudyMate, a helpful academic assistant. Use ONLY the provided context. If unsure, say you don't know."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            return resp.choices[0].message.content
    except Exception as e:
        st.warning(f"OpenAI error: {e}")

    return "No LLM API key detected. Add GROQ_API_KEY (recommended) or OPENAI_API_KEY in your environment/secrets and try again."

# --- Sidebar: PDF upload ---
st.sidebar.header("📄 Upload PDFs")
files = st.sidebar.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.meta = []  # (source_name, page_hint)

if st.sidebar.button("Build Index", type="primary"):
    if not files:
        st.sidebar.error("Please upload at least one PDF first.")
    else:
        with st.spinner("Extracting, chunking, and indexing… (first time can take a minute)"):
            embedder = load_embedder()
            all_chunks, meta = [], []
            for f in files:
                name = f.name
                data = f.read()
                text, _ = read_pdf_bytes(data)
                chunks = chunk_text(text)
                for c in chunks:
                    all_chunks.append(c)
                    # crude page hint: search for page-like tokens if any (optional simple heuristic)
                    meta.append((name, None))
            vecs = embed_texts(embedder, all_chunks)
            index = init_faiss(vecs.shape[1])
            index.add(vecs)
            st.session_state.index = index
            st.session_state.chunks = all_chunks
            st.session_state.meta = meta
        st.sidebar.success(f"Indexed {len(st.session_state.chunks)} chunks from {len(files)} PDF(s).")

# --- Main QA box ---
q = st.text_input("Ask a question about your PDFs:", placeholder="e.g., Explain power factor improvement methods and why they matter.")

col1, col2 = st.columns([1,1])
with col1:
    top_k = st.slider("How many passages to use", 3, 10, 5)
with col2:
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2)

if st.button("Answer", type="primary"):
    if st.session_state.index is None:
        st.error("Please upload and Build Index first.")
    elif not q.strip():
        st.error("Type a question.")
    else:
        with st.spinner("Searching your PDFs and asking the model…"):
            # Retrieve
            embedder = load_embedder()
            qvec = embed_texts(embedder, [q])
            scores, idxs = st.session_state.index.search(qvec, top_k)
            idxs = idxs[0]
            scores = scores[0]
            contexts = []
            for rank, (i, sc) in enumerate(zip(idxs, scores), start=1):
                passage = st.session_state.chunks[i]
                src_name, page_hint = st.session_state.meta[i]
                contexts.append((rank, sc, passage, src_name, page_hint))

            # Build prompt
            ctx_text = "\n\n".join([f"[Source {r}] {p}" for r, _, p, _, _ in contexts])
            prompt = (
                "You are answering strictly from the following context.\n\n" +
                ctx_text +
                "\n\nQuestion: " + q +
                "\n\nInstructions: Give a clear, student‑friendly answer. Cite sources as [Source N]. If not in context, say you don't know."
            )

            # Answer
            answer = call_llm(prompt)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show supporting passages (citations)"):
            for r, sc, passage, src_name, page_hint in contexts:
                st.markdown(f"*Source {r}* — {src_name}  |  score: {sc:.3f}")
                st.write(passage)
                st.divider()

st.info("Tip: Index multiple PDFs together to ask cross‑document questions (e.g., combine textbook + notes).")