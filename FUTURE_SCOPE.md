# Future Scope

Directions to take the RAG Generator beyond the current assessment-sized build.

## 1. Agentic query processing

Today each question runs one retrieval. An agentic query step would plan before retrieving:

- **Query decomposition:** split a large or multi-part question ("Compare the CSF Tiers
  with PEP 8's approach to consistency") into focused sub-queries, retrieve for each, and
  merge the evidence before answering.
- **Metadata filtering:** use the payload already stored with every chunk (`source`,
  `section`, `page_start`/`page_end`, `doc_id`) as Qdrant filters, so a sub-query such as
  "tiers in the NIST document" searches only that document or section. This narrows the
  candidate set and gives more precise, more accurate results.
- **Iterative retrieval:** when the evidence guard reports weak evidence, rewrite the query
  and retry before refusing.
- Fits behind the existing interfaces: a `QueryPlanner` step in front of the `Retriever`,
  with citations still verified per sub-answer.

## 2. Retrieval evaluation pipeline

Turn `eval/run_eval.py` into a continuous pipeline:

- Run on every change (CI) and fail the build when hit@k, MRR, refusal accuracy or
  citation-support rate regress beyond a tolerance.
- Grow the question sets per KB, including questions generated from the documents and
  real user questions with feedback (thumbs up/down, corrected sources).
- Add answer-level metrics (faithfulness, answer relevance, context precision/recall),
  optionally with an LLM judge.
- Track results over time to compare chunking, embedding, reranker and threshold changes.

## 3. Apache Tika for parsing

Add a `TikaParser` behind the existing `Parser` interface and map extensions to it in
`config/settings.yaml`. Tika handles 1000+ formats (PPTX, XLSX, HTML, email, RTF, ODT, ...)
and can run OCR through Tesseract for scanned PDFs, a current limitation. No pipeline
changes are needed, only a parser class and config entries.

## 4. NVIDIA Triton for model serving

Serve the embedding model, the cross-encoder and optionally the LLM from Triton Inference
Server instead of in-process fastembed:

- GPU acceleration and dynamic batching for high ingestion and query throughput.
- The API becomes stateless and light; models scale independently.
- Implemented as new `DenseEmbedder` / `Reranker` adapters that call Triton, selected by config.

## 5. On-premises deployment

The whole stack can run on-prem or air-gapped:

- Qdrant, the API and the UI are already containers; models already load from a local
  directory (`models/local/`).
- Point LiteLLM at a self-hosted LLM (vLLM, Triton, Ollama or an internal gateway) using
  an OpenAI-compatible base URL; no code change.
- Package with Helm/Kubernetes, add authentication (SSO/OIDC), per-KB access control and
  audit logging for enterprise use.
