# Decisions

Short log of choices that shape the system. Newest at the bottom.

## D1. No LangChain; small Protocols instead
Each pipeline step (`Parser`, `Chunker`, `DenseEmbedder`, `SparseEmbedder`, `VectorStore`,
`Reranker`, `EvidenceGuard`, `LLM`, `CitationVerifier`) is a Protocol in
`backend/app/core/interfaces.py`. `generation/answerer.py` wires the query steps and
`ingestion/pipeline.py` wires the ingestion steps, so each flow reads top to bottom.

## D2. "Config alone" means KBs, documents and extension mapping, not new formats
New KBs and documents are runtime API calls. `config/settings.yaml` maps file
extensions to existing parsers (for example `.markdown` and `.rst` map to the markdown and
text parsers), and sets chunking, models, retrieval sizes and thresholds. A new binary
format still needs a parser class; parsers are referenced by dotted path in config, so
adding one needs no change to the registry or pipeline.

## D3. Durable job queue in SQLite with one worker thread
FastAPI `BackgroundTasks` loses work on restart. Jobs live in SQLite and are claimed by an
in-process worker thread. On startup, jobs left `running` are put back to `queued`.
Celery/Redis would be overkill at this size.

## D4. One Qdrant collection per KB
It keeps deletion trivial and isolation obvious. Beyond a few hundred KBs, Qdrant
recommends one collection with a payload filter on `kb_id`; the `VectorStore` interface
hides this, so a switch is local to `qdrant_store.py`.

## D5. BM25 via fastembed `Qdrant/bm25` + Qdrant IDF modifier
The sparse model gives term frequencies and Qdrant applies IDF at query time. This is
BM25-style scoring rather than exact Lucene BM25, which is fine for hybrid retrieval.
Fusion is RRF, server-side, through the Query API with two prefetches.

## D6. Chunk text lives only in the Qdrant payload
SQLite holds KB, document and job metadata. Chunk text, page, section and source file
name are stored in the Qdrant payload, so a retrieval hit already has everything
needed to build a prompt and a citation.

## D7. PDF parsing with PyMuPDF
PyMuPDF exposes font sizes, which makes heading detection workable. Its license is AGPL,
which is acceptable for an assessment; `pypdf` is the permissive fallback (loses heading
detection, keeps page numbers).

## D8. Streaming first, verification at the end
Tokens stream as they arrive. After the answer completes, the verifier runs and a final
`citations` SSE event labels each `[n]` as `supported`, `unsupported` or `invalid`. The
UI renders the verified status. Buffering the whole answer would defeat streaming.

## D9. Citation verification reuses the cross-encoder
A citation is `invalid` if `n` is not a provided source. Otherwise the cited sentence
(with citation markers removed) is scored against the source chunk with the reranker's
cross-encoder; below `citation_min_score` it is `unsupported`. No second LLM call.

## D10. Refusal rule
Rerank scores are passed through a sigmoid. Refuse if the top score is below
`min_top_score`, or if fewer than `min_supporting` chunks reach `support_score`. The
model may also reply with the refusal sentence; that counts as a refusal only when the
reply carries no citation, so a partly answered question is not mistaken for one. The
thresholds are calibrated on the eval set (D13) and can be overridden per KB.

## D11. Contextual chunk header: document title + section
Chunks are embedded and reranked as `"<title> > <section>\n<text>"`, where the title comes
from the file name (`art_of_war.txt` -> `art of war`). Questions often name the work
("according to the Enchiridion"), and the cross-encoder otherwise preferred the book's
introduction, which contains that word, over the passage that answers. Humanities
hybrid+rerank: hit@1 0.73 -> 0.80, MRR 0.76 -> 0.82. Cost: a near-domain unanswerable that
names the document ("What ... does PEP 8 allow") scores higher (0.01 -> 0.54), still below
the threshold. The prompt and citations show the original text, not the header.

## D12. Final order is the reranker's own
Tested alternatives on both corpora: RRF of first-stage and rerank ranks (k=10, 60) and
linear blends (w=0.5, 0.7). None beat reranker-only on hit@1 or MRR; blends traded
hit@1 for hit@5 on one corpus. Kept the simplest.

## D13. Thresholds are calibrated, not guessed
`guard.min_top_score = 0.6` is the middle of the best balanced-accuracy plateau over both
corpora. Leave-one-corpus-out (tune on A, test on B) gives 0.93 and 1.0 balanced accuracy,
which is the generalisation evidence. `citations.min_score = 0.5`: correct claim/source
pairs score >= 0.98 and mismatched pairs <= 0.76; erring towards flagging, because a false
"unsupported" warning is cheaper than a silently wrong citation.

## D14. Models provisioned into `models/local/`, with fallbacks
The build network blocked the Hugging Face CDN. Models load from
`<cache>/local/<org>__<name>` when present, else the hub cache. `download_models.py`
falls back to Qdrant's GCS tarball for the dense model, and `export_reranker_onnx.sh`
exports the original cross-encoder weights (from Weaviate's reranker image) to ONNX,
verified to match torch logits. The API container runs the same script on start so it
fails fast with instructions rather than on the first request.

## D15. LLM failures are visible and specific
Errors reach the client as `answer generation failed: <Type>: <first line>` (SSE `error`
event or HTTP 502), and `/health` plus the UI report missing provider keys via
`litellm.validate_environment`. LiteLLM retries need `tenacity`, which is pinned
explicitly because its absence masked the real error.

## D16. Sync pipeline, async edges
Model inference and SQLite are blocking, so endpoints are plain `def` (FastAPI runs them
in its thread pool) and the SSE endpoint iterates the synchronous answer generator via
`iterate_in_threadpool`. No async rewrite of CPU-bound code.
