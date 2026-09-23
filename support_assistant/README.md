# Zepto Support Assistant

## Architecture and request flow

The application is a local retrieval-augmented question-answering API. Its policy knowledge base lives in `docs/`; ingestion creates document embeddings and stores them in a persistent ChromaDB collection. At request time, LangGraph classifies intent, retrieves policy context when appropriate, and generates the response.

```text
Ingestion: docs/*.txt -> load_documents -> all-MiniLM-L6-v2 embeddings -> ChromaDB
Request:   query -> classify_intent -> (policy: retrieve -> generate) | (general: fixed response)
```

## File and function map

| File | Responsibility |
| --- | --- |
| `docs/doc_*.txt` | Eight policy source documents; one document is stored as one chunk. |
| `ingest.py` | `load_documents()` loads the policy text and metadata; `build_vector_store()` embeds and writes it to ChromaDB. |
| `retriever.py` | Loads the embedding model and persistent collection; `retrieve()` returns the top matching chunks; `mock_answer()` makes deterministic output. |
| `prompt_template.py` | Defines the policy-only generation instructions and builds the real-LLM prompt. |
| `graph.py` | `classify_intent()`, `retrieve_and_answer()`, `direct_answer()`, `route_after_classification()`, and `build_graph()` define the LangGraph flow. `generate_real_llm_answer()` validates structured output and retries failures. |
| `models.py` | Pydantic request and response schemas. |
| `app.py` | FastAPI health route and `POST /ask` endpoint. |
| `Dockerfile` | Installs dependencies, builds the local vector store into the image, and starts Uvicorn. |

## Ingestion, embeddings, and retrieval

Run `python -m support_assistant.ingest` from the repository root to read the eight non-empty `docs/doc_*.txt` files. Each file becomes one chunk with an ID such as `doc_01_chunk_01` and metadata containing `source` and `document_id`. The `all-MiniLM-L6-v2` Sentence Transformers model embeds each chunk. ChromaDB persists the embeddings locally under `support_assistant/chroma_db/` in the collection named `zepto_policies`, configured with cosine distance. At query time the same model embeds the query and retrieval returns the top three chunks.

## LangGraph nodes and intent behavior

The graph has three nodes:

* `classify_intent` checks for policy keywords: `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, and `support hours`.
* `retrieve_and_answer` retrieves up to three policy chunks, then uses deterministic mock generation or the optional real LLM.
* `direct_answer` handles general questions with: `I can only answer questions about Zepto policies right now.`

Policy intent routes through retrieval and generation. General intent skips retrieval and returns the fixed policy-scope answer with empty sources. A policy question with no retrieved chunks returns the no-relevant-policy message, empty sources, and confidence `0.0`.

## Examples and raw JSON responses

Policy query:

```json
{"query":"How long does delivery take?"}
```

Typical mock response (the answer text is the retrieved top chunk, truncated to 300 characters):

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.","sources":["doc_01"],"confidence":1.0}
```

General query:

```json
{"query":"What is the capital of France?"}
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

The actual policy response source IDs depend on retrieved documents and are returned in `sources`.

## Mock and real LLM modes

`MOCK_LLM=1` is the default and produces deterministic text from the highest-ranked retrieved chunk; it needs no API key. Any value other than `0` selects this mode. Set `MOCK_LLM=0` to enable the OpenAI structured-output path. It requires `OPENAI_API_KEY`; `LLM_MODEL` optionally selects the model and defaults to `gpt-4o-mini`. The real model must return `answer` (string), `sources` (list of strings), and `confidence` (number from 0 to 1). The parser validates that schema, and generation makes one initial attempt plus up to two retries if parsing or validation fails. The graph preserves the parsed fields in its final response.

## FastAPI endpoint and schema

Start the API from the repository root with `uvicorn support_assistant.app:app --host 0.0.0.0 --port 7860`. `GET /` is a health message. `POST /ask` accepts:

```json
{"query":"How do returns work?"}
```

`QueryRequest` contains `query: string`. `QueryResponse` contains `answer: string`, `sources: string[]`, and `confidence: number` constrained to `[0, 1]`. Interactive API documentation is at `http://127.0.0.1:7860/docs`.

## Build and run with Docker

From the repository root:

```powershell
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run --rm -p 7860:7860 zepto-support-assistant
```

The image builds the local policy index during image creation and defaults to `MOCK_LLM=1`. To use the real LLM mode, pass `-e MOCK_LLM=0 -e OPENAI_API_KEY=...` to `docker run` and optionally set `LLM_MODEL`.

## Verification

The repository's previous local verification record reports successful FastAPI policy/general requests and Docker build/start in mock mode. To repeat the core checks after ingestion, run the server and exercise both examples through `/ask` or `/docs`; verify policy output has source IDs and general output has `sources: []`. Build and launch with the Docker commands above to verify container startup. Real-LLM behavior requires a valid API key and was not part of the deterministic mock verification.
