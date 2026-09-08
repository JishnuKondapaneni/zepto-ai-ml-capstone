from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load local embedding model and ChromaDB
# ---------------------------------------------------------

print("Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# ---------------------------------------------------------
# Retrieve relevant policy chunks
# ---------------------------------------------------------

def retrieve(query: str, top_k: int = 3):
    """
    Retrieve the most relevant policy chunks using
    cosine similarity.
    """

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "document": document,
                "source": metadata["source"],
                "document_id": metadata["document_id"],
                "distance": float(distance),
            }
        )

    return retrieved


# ---------------------------------------------------------
# Mock answer generation
# ---------------------------------------------------------

def mock_answer(retrieved_chunks):
    """
    Deterministic answer required by the assignment.
    """

    if not retrieved_chunks:
        return "Based on the retrieved context: No relevant policy found."

    top_chunk = retrieved_chunks[0]["document"]

    # Keep the answer concise while preserving the required
    # deterministic format.
    snippet = top_chunk[:300].strip()

    return f"Based on the retrieved context: {snippet}"


# ---------------------------------------------------------
# Test retrieval directly
# ---------------------------------------------------------

if __name__ == "__main__":
    test_query = "How long does Zepto delivery take?"

    print("\nTest query:")
    print(test_query)

    results = retrieve(test_query, top_k=3)

    print("\nRetrieved results:")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"Source: {result['source']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['document'][:200]}...")

    answer = mock_answer(results)

    print("\nMock answer:")
    print(answer)