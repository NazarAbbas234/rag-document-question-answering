from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_vectordb = None


def load_vector_db():
    global _vectordb

    if _vectordb is not None:
        return _vectordb

    print("Loading persisted Chroma DB...")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    _vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding
    )

    return _vectordb
