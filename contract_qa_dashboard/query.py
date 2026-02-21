import os
import json
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

VECTOR_DB = "chroma"



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# models (load once)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

PICKLE_PATH = os.path.join("data", "faiss_index.pkl")


def load_index():
    """
    Loads Chroma DB + metadata + BM25 index
    """

    from contract_qa_dashboard.vectorstores.chroma_store import ChromaStore

    # load chroma vector store
    chroma = ChromaStore()

    # load metadata
    metadata_path = os.path.join("data", "metadata.json")

    if not os.path.exists(metadata_path):
        raise RuntimeError("metadata.json not found. Run ingestion first.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # rebuild chunks list
    chunks = [m["text"] for m in metadata]

    # rebuild BM25
    tokenized_corpus = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # FAISS index no longer used
    index = None

    return chroma, index, chunks, metadata, bm25

def normalize_scores(score_dict):
    """
    Min-max normalize scores to range [0,1]
    """
    if not score_dict:
        return score_dict

    values = list(score_dict.values())
    min_v = min(values)
    max_v = max(values)

    # avoid divide-by-zero
    if max_v - min_v == 0:
        return {k: 1.0 for k in score_dict}

    return {
        k: (v - min_v) / (max_v - min_v)
        for k, v in score_dict.items()
    }


def hybrid_search(query, chroma, index, chunks, metadata, bm25, top_k=10, alpha=0.5):

    """
    Dense (FAISS) + Sparse (BM25) hybrid.
    Returns list of tuples: (chunk_text, meta_dict, combined_score)
    """

    if VECTOR_DB == "chroma":

        q_emb = embedding_model.encode([query], convert_to_numpy=True)

        chroma_results = chroma.search(q_emb, top_k)

        dense_scores = {}
        dense_data = {}

        # ---- Dense results (Chroma) ----
        # Chroma returns DISTANCE (lower is better)
        for i, (text, meta, distance) in enumerate(chroma_results):

            # convert distance → similarity
            similarity = 1.0 / (1.0 + float(distance))

            dense_scores[i] = similarity
            dense_data[i] = (text, meta)

        # normalize dense scores
        dense_scores = normalize_scores(dense_scores)

        # ---- Sparse BM25 ----
        tokenized = query.split()
        bm25_raw = bm25.get_scores(tokenized)

        sparse_scores = {}
        sparse_data = {}

        top_sparse_idx = np.argsort(bm25_raw)[::-1][:top_k]

        for idx in top_sparse_idx:
            if idx >= len(chunks):
                continue

            sparse_scores[idx] = float(bm25_raw[idx])
            sparse_data[idx] = (chunks[idx], metadata[idx])

        # normalize sparse scores
        sparse_scores = normalize_scores(sparse_scores)

        # ---- Combine ----
        combined = {}

        # dense contribution
        for k, score in dense_scores.items():
            text, meta = dense_data[k]
            combined[k] = {
                "text": text,
                "meta": meta,
                "score": alpha * score
            }

        # sparse contribution
        for k, score in sparse_scores.items():
            text, meta = sparse_data[k]

            if k in combined:
                combined[k]["score"] += (1 - alpha) * score
            else:
                combined[k] = {
                    "text": text,
                    "meta": meta,
                    "score": (1 - alpha) * score
                }

        ranked = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        return [(r["text"], r["meta"], r["score"]) for r in ranked]



    q_emb = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(q_emb, dtype="float32"), top_k)

    combined = {}
    # Dense results (I contains chunk indices)
    for rank_pos, chunk_idx in enumerate(I[0]):
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            continue
        score = float(D[0][rank_pos])
        combined[chunk_idx] = {
            "text": chunks[chunk_idx],
            "meta": metadata[chunk_idx],
            "score": alpha * score
        }

    # Sparse results (BM25)
    tokenized = query.split()
    bm25_scores = bm25.get_scores(tokenized)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k]
    for idx in bm25_top_idx:
        if idx < 0 or idx >= len(chunks):
            continue
        score = float(bm25_scores[idx])
        if idx in combined:
            combined[idx]["score"] += (1 - alpha) * score
        else:
            combined[idx] = {
                "text": chunks[idx],
                "meta": metadata[idx],
                "score": (1 - alpha) * score
            }

    # Sort by combined score
    ranked = sorted(combined.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
    # return list of (text, meta, score)
    return [(v["text"], v["meta"], v["score"]) for _, v in ranked]


def rerank(query, candidates, top_k=5):
    """
    candidates: list of (text, meta, score)
    returns: list of (text, meta) reranked by CrossEncoder
    """
    if not candidates:
        return []
    pairs = [(query, c[0]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
    # return list of (text, meta)
    return [(c[0][0], c[0][1]) for c in ranked]


def synthesize_answer(query, passages, model="gpt-4o-mini", use_top_n=1):
    """
    passages: list[(chunk_text, meta)]
    returns: dict { answer: str, references: [str], passages: [{meta, snippet}] }
    """
    # local dedupe
    seen = set()
    unique = []
    for chunk, meta in passages:
        if chunk in seen:
            continue
        seen.add(chunk)
        unique.append((chunk, meta))

    if not unique:
        return {
            "answer": "I could not find relevant information in the indexed documents.",
            "references": [],
            "passages": []
        }

    # choose top N passages to give the LLM (default 1)
    use_n = min(len(unique), use_top_n)
    chosen = unique[:use_n]

    # build context and references (numbered [1], [2], ...)
    context = ""
    references = []
    passages_for_ui = []
    for idx, (chunk, meta) in enumerate(chosen, start=1):
        context += f"[{idx}] {chunk}\n"
        clause_id = meta.get("clause_id", "Unknown")
        section = meta.get("section", "Unknown")
        doc_name = meta.get("doc_id", "Unknown")
        references.append(f"[{idx}] Clause {clause_id}: {section} (under {doc_name})")
        passages_for_ui.append({
            "index": idx,
            "doc_id": doc_name,
            "clause_id": clause_id,
            "section": section,
            "chunk_index": meta.get("chunk_index"),
            "snippet": chunk[:800].strip()
        })

    # prompt the LLM to only use the passages
    system_prompt = """
    You are a legal contract analysis assistant.

    Analyze the provided contract passages and return ONLY valid JSON.

    Output format:

    {
    "summary": "short plain English summary",
    "clause_type": "Termination | Compensation | Confidentiality | Liability | Other",
    "risk_level": "Low | Medium | High",
    "explanation": "brief reasoning strictly based on passages"
    }

    Rules:
    - Use ONLY provided passages.
    - Do NOT add extra text.
    - Do NOT use markdown.
    - Return JSON only.
    """

    user_prompt = f"""
    Passages:
    {context}

    Question:
    {query}
    """


    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=500
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        structured = json.loads(raw_output)
    except json.JSONDecodeError:
        # fallback safety (LLM sometimes adds text)
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        structured = json.loads(raw_output[start:end])

    return structured


def query(questions, top_k=5, use_top_n=1):
    """
    Main entry point.
    Returns a list of dicts (one per question) with keys:
      - answer (str)
      - references (list[str])
      - passages (list[meta dicts])
    """
    chroma, index, chunks, metadata, bm25 = load_index()

    results = []

    for q in questions:
        # retrieve (chunk text, meta, score)
        candidates_with_scores = hybrid_search(q, chroma, index, chunks, metadata, bm25, top_k=10)
        # rerank returns (text, meta)
        reranked = rerank(q, candidates_with_scores, top_k=top_k)
        # synthesize using top N passages (use_top_n controls it)
        out = synthesize_answer(q, reranked, use_top_n=use_top_n)
        results.append(out)

    return results
