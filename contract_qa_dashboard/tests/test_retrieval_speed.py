import time
from vector_store import load_vector_db

db = load_vector_db()

query = "What is the main topic?"

start = time.time()

results = db.similarity_search(query, k=5)

end = time.time()

print("Retrieval time:", end - start)
