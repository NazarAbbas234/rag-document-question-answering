from api.vector_store import load_vector_db

print("Loading persisted Chroma DB...")

db = load_vector_db()

print("Running similarity search...")

results = db.similarity_search("test query", k=3)

for r in results:
    print(r.page_content[:100])
