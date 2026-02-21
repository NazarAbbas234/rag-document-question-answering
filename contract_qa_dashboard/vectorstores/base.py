from abc import ABC, abstractmethod

class VectorStore(ABC):

    @abstractmethod
    def add(self, texts, metadatas, embeddings):
        pass

    @abstractmethod
    def search(self, query_embedding, top_k):
        pass

    @abstractmethod
    def persist(self):
        pass
