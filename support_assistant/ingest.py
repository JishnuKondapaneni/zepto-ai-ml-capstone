from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"


# ---------------------------------------------------------
# Load policy documents
# ---------------------------------------------------------

def load_documents():
    documents = []
    ids = []
    metadatas = []

    for doc_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = doc_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        # One document = one chunk.
        documents.append(text)
        ids.append(f"{doc_path.stem}_chunk_01")
        metadatas.append(
            {
                "source": doc_path.name,
                "document_id": doc_path.stem,
            }
        )

    if len(documents) != 8:
        raise ValueError(
            f"Expected exactly 8 policy documents, found {len(documents)}."
        )

    return documents, ids, metadatas


# ---------------------------------------------------------
# Create ChromaDB collection
# ---------------------------------------------------------

def build_vector_store():
    documents, ids, metadatas = load_documents()

    print(f"Loaded {len(documents)} policy documents.")

    # Local embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings...")
    embeddings = model.encode(documents).tolist()

    # Persistent local ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Remove old records so the script can safely be run again.
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Store documents + embeddings
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Stored {collection.count()} chunks in ChromaDB.")
    print("Ingestion completed successfully.")


# ---------------------------------------------------------
# Run ingestion
# ---------------------------------------------------------

if __name__ == "__main__":
    build_vector_store()