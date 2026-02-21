from typing import List, Dict
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from contract_qa_dashboard import utils
from pypdf import PdfReader
import docx2txt

#VECTOR_DB = "faiss"   # options: faiss | chroma
VECTOR_DB = "chroma"



# --- Config / filenames ---
INDEX_DIR = "data"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
PICKLE_PATH = os.path.join(INDEX_DIR, "faiss_index.pkl")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # general-purpose; swap for legal-tuned

# --- Clause/heading regex ---
HEADING_RE = re.compile(
    r'(?mi)^\s*#{0,6}\s*'                          
    r'(?:(?:Clause|Section|Article)\s+)?'          
    r'(?P<num>\d+(?:\.\d+)*)'                    
    r'(?:[\)\.\:\-]|\s-\s)?\s*'                   
    r'(?P<title>.+?)\s*$'                        
)

# --- Model singleton ---
_model = None
def _get_model():
    global _model
    if _model is None:
        print(f"[ingest] Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

# --- Document readers ---
def _read_pdf(path: str) -> str:
    text = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    except Exception as e:
        print(f"[ingest] ERROR reading PDF {path}: {e}")
    return "\n".join(text)

def _read_docx(path: str) -> str:
    try:
        return docx2txt.process(path) or ""
    except Exception as e:
        print(f"[ingest] ERROR reading DOCX {path}: {e}")
        return ""

def _read_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _read_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    elif ext == ".docx":
        return _read_docx(path)
    elif ext == ".md":
        return _read_markdown(path)
    else:
        print(f"[ingest] Unsupported file type: {path}")
        return ""

# --- Clause extractor ---
def extract_clause_metadata(text: str) -> List[Dict]:
    """
    Extract clauses/sections from text using regex headings.
    Returns list of dicts with {clause_id, section, span_text}.
    """
    matches = list(HEADING_RE.finditer(text))
    clauses = []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause_id = m.group("num")
        title = m.group("title").strip()
        span_text = text[start:end].strip()

        clauses.append({
            "clause_id": clause_id,
            "section": title,
            "text": span_text
        })

    return clauses

# --- Main ingestion ---
def ingest(document_paths: List[str]) -> None:
    """
    Ingest documents into FAISS index and save metadata + chunks + full text.
    Supports PDF, DOCX, MD.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    model = _get_model()

    documents = []   # full document text per file (for BM25)
    all_chunks = []  # split chunks of text (for FAISS embeddings)
    metadata = []    # metadata per chunk

    for doc_path in document_paths:
        doc_path = utils.normalize_path(doc_path)
        if not os.path.exists(doc_path):
            print(f"[ingest] WARNING: file not found: {doc_path}")
            continue

        content = _read_document(doc_path)
        if not content.strip():
            print(f"[ingest] WARNING: no text extracted from {doc_path}")
            continue

        documents.append(content)
        doc_id = os.path.basename(doc_path)

        # --- Extract clause metadata ---
        clauses = extract_clause_metadata(content)
        if clauses:
            for clause in clauses:
                chunks = utils.chunk_text_by_words(
                    clause["text"],
                    min_words=80,
                    max_words=250,
                    overlap_words=30
                )
                for cidx, chunk in enumerate(chunks):
                    entry = {
                        "doc_id": doc_id,
                        "chunk_index": cidx,
                        "clause_id": clause["clause_id"],
                        "section": clause["section"],
                        "text": chunk
                    }
                    metadata.append(entry)
                    all_chunks.append(chunk)
        else:
            # fallback: chunk full document if no clauses detected
            chunks = utils.chunk_text_by_words(
                content,
                min_words=120,
                max_words=300,
                overlap_words=40
            )
            for cidx, chunk in enumerate(chunks):
                entry = {
                    "doc_id": doc_id,
                    "chunk_index": cidx,
                    "clause_id": "?",
                    "section": "Document",
                    "text": chunk
                }
                metadata.append(entry)
                all_chunks.append(chunk)

    if not metadata:
        print("[ingest] No valid documents to ingest.")
        return

    print(f"[ingest] Computing embeddings for {len(all_chunks)} chunks...")

    vectors = model.encode(
        all_chunks,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    from contract_qa_dashboard.vectorstores.chroma_store import ChromaStore

    chroma = ChromaStore()
    chroma.add(all_chunks, metadata, vectors)

    utils.save_json(metadata, METADATA_PATH)

    print("[ingest] Chroma DB persisted successfully")


