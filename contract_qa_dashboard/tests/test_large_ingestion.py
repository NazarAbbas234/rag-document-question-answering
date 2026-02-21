import shutil
import os
import time
from pathlib import Path

# Import your functions
from contract_qa_dashboard.ingest import ingest
from api.vector_store import load_vector_db

# Base directories
BASE_DIR = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = BASE_DIR / "docs"
TEMP_DIR = DATA_DIR / "contracts_large"

def get_document_paths(folder):
    """Return list of .txt, .pdf, or .md files in folder"""
    files = []
    for f in os.listdir(folder):
        if f.endswith((".pdf", ".txt", ".md")): 
            files.append(os.path.join(folder, f))
    return files

def create_large_dataset(multiplier=40):

    large_dir = DATA_DIR / "contracts_large"

    # clean previous dataset (important for repeat tests)
    if large_dir.exists():
        shutil.rmtree(large_dir)

    large_dir.mkdir()

    # ONLY take files from docs root (ignore folders)
    files = [
        f for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in [".txt", ".pdf", ".md"]
    ]

    print(f"Found {len(files)} source documents")

    count = 0
    for i in range(multiplier):
        for file in files:
            dst = large_dir / f"{i}_{file.name}"
            shutil.copy(file, dst)

            count += 1
            if count % 20 == 0:
                print(f"Copied {count} files...")


def run_scalability_test():
    print("\n===  Creating Large Dataset ===")
    create_large_dataset(40)  # adjust multiplier here

    print("\n=== Running Ingestion ===")
    start = time.time()
    # Get all document paths in the large dataset folder
    document_paths = get_document_paths(TEMP_DIR)
    if not document_paths:
        print("No valid documents found in", TEMP_DIR)
        return
    ingest(document_paths)
    print(f"Ingestion time: {time.time() - start:.2f}s")

    print("\n===Loading Vector DB ===")
    db = load_vector_db()

    print("\n===Query Speed Test ===")
    queries = [
        "termination clause",
        "salary payment",
        "confidential information",
        "employee role"
    ]

    for q in queries:
        t0 = time.time()
        results = db.similarity_search(q, k=3)
        elapsed = time.time() - t0

        print(f"\nQuery: {q}")
        print(f"Search time: {elapsed:.4f}s")
        for r in results:
            print("-", r.metadata.get("clause_id", "N/A"),
                  r.page_content[:80])

if __name__ == "__main__":
    run_scalability_test()
