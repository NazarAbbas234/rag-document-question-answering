import chromadb
import numpy as np
import os

class ChromaStore:

    def __init__(self, collection_name="contracts"):
        # ensure directory exists
        persist_dir = "data/chroma_db"
        os.makedirs(persist_dir, exist_ok=True)

        # create persistent client
        self.client = chromadb.PersistentClient(path=persist_dir)

        # create or get a collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(self, texts, metadatas, embeddings):
        ids = [str(i) for i in range(len(texts))]

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
            ids=ids
        )

        print("[ingest] Added chunks to Chroma DB")

    def search(self, query_embedding, top_k=5):
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )

        outputs = []
        for doc, meta, score in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            outputs.append((doc, meta, score))

        return outputs
