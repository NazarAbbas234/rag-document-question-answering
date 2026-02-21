import json
import os
from typing import List, Dict, Tuple, Optional


class CitationManager:
    """
    Manages citations and generates References section.
    Supports hierarchy: section + parent + doc_id.
    """
    def __init__(self):
        self.map = {}
        self.order = []

    def cite(self, doc_id: str, section: str, parent: Optional[str] = None) -> str:
        """
        Register a citation and return its inline reference (e.g. [1]).
        """
        key = (doc_id, section, parent)
        if key not in self.map:
            self.order.append(key)
            self.map[key] = len(self.order)  # 1-based index
        return f"[{self.map[key]}]"

    def references_text(self) -> str:
        """
        Generate a Vancouver-style References section.
        """
        lines = []
        for i, (doc_id, section, parent) in enumerate(self.order, start=1):
            if parent:
                lines.append(f"[{i}] {section} (under {parent}, {doc_id})")
            else:
                lines.append(f"[{i}] {section} ({doc_id})")
        return "\n".join(lines)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text_by_words(text: str, min_words=120, max_words=300, overlap_words=30) -> List[str]:
    """
    Chunk text by words with overlap to preserve context.
    Returns list of chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    n = len(words)

    while i < n:
        end = i + max_words
        chunk = words[i:end]
        if len(chunk) >= min_words or end >= n:
            chunks.append(" ".join(chunk))
        i = end - overlap_words
        if i < 0:
            i = 0

    return chunks


def normalize_path(p: str) -> str:
    return os.path.abspath(p)
