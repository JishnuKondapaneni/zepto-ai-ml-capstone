import os
from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph

from support_assistant.prompt_template import build_prompt
from support_assistant.retriever import mock_answer, retrieve


MOCK_LLM = os.getenv("MOCK_LLM", "1")


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


class SupportState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[dict]
    answer: str
    sources: List[str]
    confidence: float


def classify_intent(state: SupportState) -> dict:
    """Classify the user's query as a policy or general question."""

    query = state["query"].lower()

    if any(keyword in query for keyword in POLICY_KEYWORDS):
        return {"intent": "policy_question"}

    return {"intent": "general_question"}


def retrieve_and_answer(state: SupportState) -> dict:
    """Retrieve the top-3 policy chunks and generate an answer."""

    chunks = retrieve(state["query"], top_k=3)

    if not chunks:
        return {
            "retrieved_chunks": [],
            "answer": "I could not find relevant Zepto policy information.",
            "sources": [],
            "confidence": 0.0,
        }

    sources = [chunk["document_id"] for chunk in chunks]

    # MOCK_LLM=1 is the default graded baseline.
    if MOCK_LLM != "0":
        answer = mock_answer(chunks)
        return {
            "retrieved_chunks": chunks,
            "answer": answer,
            "sources": sources,
            "confidence": 1.0,
        }

    # Optional real-LLM path.
    context = "\n\n".join(
        f"[{chunk['document_id']}] {chunk['document']}"
        for chunk in chunks
    )

    prompt = build_prompt(
        query=state["query"],
        retrieved_context=context,
    )

    response = generate_real_llm_answer(prompt)

    return {
        "retrieved_chunks": chunks,
        "answer": response["answer"],
        "sources": response["sources"],
        "confidence": response["confidence"],
    }


def generate_real_llm_answer(prompt: str) -> dict:
    """
    Optional real LLM generation.

    The mock mode remains the default. The real path is enabled only
    when MOCK_LLM=0.
    """

    try:
        from pydantic import BaseModel, Field
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Real LLM mode requires the openai package. "
            "Install it with: python -m pip install openai"
        ) from exc

    class LLMResponse(BaseModel):
        answer: str
        sources: List[str] = Field(default_factory=list)
        confidence: float = Field(ge=0.0, le=1.0)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires the OPENAI_API_KEY environment variable."
        )

    client = OpenAI(api_key=api_key)

    last_error = None

    # Initial attempt + up to 2 additional retries.
    for attempt in range(3):
        try:
            response = client.beta.chat.completions.parse(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only a structured response matching "
                            "the requested schema."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format=LLMResponse,
            )

            parsed = response.choices[0].message.parsed

            if parsed is None:
                raise ValueError("The LLM returned no structured response.")

            return {
                "answer": parsed.answer,
                "sources": parsed.sources,
                "confidence": parsed.confidence,
            }

        except Exception as exc:
            last_error = exc

            if attempt < 2:
                continue

    raise RuntimeError(
        f"LLM response validation failed after 3 attempts: {last_error}"
    )


def direct_answer(state: SupportState) -> dict:
    """Handle questions outside the Zepto policy domain."""

    return {
        "answer": "I can only answer questions about Zepto policies right now.",
        "sources": [],
        "confidence": 1.0,
    }


def route_after_classification(state: SupportState) -> str:
    """Route the graph according to the classified intent."""

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


def build_graph():
    """Build and compile the LangGraph support assistant."""

    graph = StateGraph(SupportState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()
