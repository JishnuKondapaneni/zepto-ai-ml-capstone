from fastapi import FastAPI, HTTPException

from support_assistant.graph import build_graph
from support_assistant.models import QueryRequest, QueryResponse


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Zepto Policy Support Assistant",
    description="A local RAG-based Zepto policy question-answering API.",
    version="1.0.0",
)


# Build the LangGraph application once when the API starts.
graph_app = build_graph()


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Zepto Policy Support Assistant is running."
    }


# ---------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """
    Ask a question about Zepto policies.
    """

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    initial_state = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    }

    result = graph_app.invoke(initial_state)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )