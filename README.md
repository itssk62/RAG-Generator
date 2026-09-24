# RAG Generator

Create knowledge bases at runtime, upload PDF / DOCX / TXT / MD files, and ask questions.
Answers are grounded in retrieved passages, cite them as `[n]`, and every citation is
verified. When the evidence is weak the system refuses instead of guessing.

Design: [HLD.md](HLD.md). Why things are the way they are: [DECISIONS.md](DECISIONS.md).

## Prerequisites

Install or download these before running:

| What | Version | Needed for | Notes |
|---|---|---|---|
| Docker Engine + Docker Compose v2 | Docker 24+, Compose 2.20+ | Running the app (recommended path) | `docker compose version` should work |
| An LLM API key | - | Generating answers | Any provider LiteLLM supports: OpenAI, Anthropic, Gemini, Azure OpenAI, or an OpenAI-compatible gateway |
| Python | 3.11 or 3.12 | Model download script, tests, eval, local (non-Docker) runs | Not needed inside the containers |
| Disk space | ~3 GB free | Docker images (~1.5 GB) + models (~300 MB) + data | |
| RAM | 4 GB free | Embedding and reranking run on CPU | No GPU needed |
| Network access | - | First run only: Docker Hub (or a mirror), PyPI, huggingface.co | After models are in `./models`, only the LLM provider is contacted |

The three ONNX models (downloaded automatically, about 300 MB in total):

| Model | Role |
|---|---|
| `BAAI/bge-small-en-v1.5` | Dense embeddings |
| `Qdrant/bm25` | BM25 sparse vectors |
| `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking and citation checks |

If huggingface.co's file CDN is blocked on your network, see
[Offline model provisioning](#offline-model-provisioning).

## Run the application (Docker Compose)

1. **Configure the LLM.** Copy the example env file and set a model and its key:

   ```bash
   cd RAG-Generator
   cp .env.example .env
   ```

   Pick one block for `.env`:

   ```bash
   # OpenAI
   RAG__MODELS__LLM=gpt-4o-mini
   OPENAI_API_KEY=sk-...

   # Anthropic
   RAG__MODELS__LLM=anthropic/claude-3-5-haiku-latest
   ANTHROPIC_API_KEY=...

   # Google Gemini
   RAG__MODELS__LLM=gemini/gemini-2.0-flash
   GEMINI_API_KEY=...

   # Azure OpenAI (model string is azure/<your deployment name>)
   RAG__MODELS__LLM=azure/my-gpt-4o-mini-deployment
   AZURE_API_KEY=...
   AZURE_API_BASE=https://<resource>.openai.azure.com
   AZURE_API_VERSION=2024-10-21

   # OpenAI-compatible gateway or proxy (LiteLLM proxy, vLLM, internal gateway)
   RAG__MODELS__LLM=openai/<model-name-on-the-gateway>
   OPENAI_API_KEY=...
   OPENAI_API_BASE=https://<gateway-host>/v1
   ```

2. **Download the models** (optional; the API container also does this on first start):

   ```bash
   python3 -m pip install fastembed==0.8.1
   python3 scripts/download_models.py        # writes to ./models, mounted into the API container
   ```

3. **Build and start** Qdrant, the API and the UI:

   ```bash
   docker compose up -d --build
   ```

   If the build fails with `failed to fetch anonymous token ... auth.docker.io` (BuildKit
   bypassing a registry mirror), build with the classic builder instead:

   ```bash
   DOCKER_BUILDKIT=0 docker build -f backend/Dockerfile -t rag-generator-api .
   DOCKER_BUILDKIT=0 docker build -f ui/Dockerfile -t rag-generator-ui .
   docker compose up -d --no-build
   ```

4. **Check it is healthy.** `missing_keys` must be empty:

   ```bash
   curl http://localhost:8000/health
   # {"status":"ok","qdrant":true,"llm":{"model":"gpt-4o-mini","missing_keys":[]}}
   ```

5. **Use it.**
   - UI: http://localhost:8501. Create a knowledge base in the sidebar, upload files on
     the Documents tab (status goes `queued` -> `processing` -> `ready`), then ask
     questions on the Ask tab.
   - API docs: http://localhost:8000/docs

6. **After changing `.env`**, recreate the API container so it picks up the new values:

   ```bash
   docker compose up -d --force-recreate api
   ```

   **To stop:** `docker compose down`. Data (KBs, documents, vectors) is kept in Docker
   volumes; `docker compose down -v` deletes it.

### Try it with the sample corpora

```bash
KB=$(curl -s -X POST localhost:8000/kbs -H 'content-type: application/json' \
       -d '{"name":"tech"}' | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
curl -s -X POST localhost:8000/kbs/$KB/documents \
  -F files=@eval/corpora/tech/nist_csf_2.0.pdf -F files=@eval/corpora/tech/pep8_style_guide.rst
curl -s localhost:8000/kbs/$KB/documents            # wait until both are "ready"
curl -s -X POST localhost:8000/kbs/$KB/query -H 'content-type: application/json' \
  -d '{"question":"What are the four CSF tiers?"}'
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `answer generation failed: AuthenticationError` | The key does not match the provider in `RAG__MODELS__LLM` (for example an Azure or gateway key used with plain `gpt-4o-mini`). Use the matching block from step 1 and recreate the API container. |
| `/health` shows `missing_keys` | The provider's key variable is not set in `.env`. |
| API container exits with `FAIL <model>` | Models could not be downloaded; see [Offline model provisioning](#offline-model-provisioning). |
| Document stuck in `failed` | The error is shown on the document; scanned PDFs without a text layer are not supported. |
| Every answer is refused | Evidence scored below `guard.min_top_score` (0.6); lower it per KB with `overrides`. |

## Local development (without Docker)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r backend/requirements.txt pytest httpx streamlit
python scripts/download_models.py
docker run -d -p 6333:6333 qdrant/qdrant:v1.19.1           # or set RAG__QDRANT__URL=:memory:
uvicorn app.main:app --app-dir backend --port 8000         # reads .env values from your shell env
API_URL=http://localhost:8000 streamlit run ui/app.py
```

Tests:

```bash
cd backend && pytest                          # 44 tests with fakes + Qdrant local mode, ~2 s
RAG_TEST_MODELS=1 pytest                      # + 3 tests with the real fastembed models
cd .. && pytest ui/test_components.py         # UI rendering tests (Streamlit AppTest)
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/kbs` | Create a KB `{name, description?, overrides?}` |
| `GET` / `DELETE` | `/kbs`, `/kbs/{kb}` | List, get, delete (drops its collection and files) |
| `POST` | `/kbs/{kb}/documents` | Multipart upload (`files`), returns document + job per file (202) |
| `GET` / `DELETE` | `/kbs/{kb}/documents[/{doc}]` | Document status (`queued`, `processing`, `ready`, `failed`), delete |
| `POST` | `/kbs/{kb}/documents/{doc}/reindex` | Re-ingest a document (replaces its chunks) |
| `GET` | `/jobs/{job}`, `/kbs/{kb}/jobs` | Ingestion job status and errors |
| `POST` | `/kbs/{kb}/query` | `{question}` -> answer, sources, citation checks, timings |
| `POST` | `/kbs/{kb}/query/stream` | Same over SSE: `token`* -> `citations` -> `done` (or `error`) |
| `GET` | `/config/formats`, `/health` | Supported extensions and upload limit; health |

## Configuration

Everything lives in [config/settings.yaml](config/settings.yaml); any key can be overridden
with `RAG__SECTION__KEY` env vars. `chunking`, `retrieval`, `guard` and `citations` can also
be overridden per KB at creation (`"overrides": {"guard": {"min_top_score": 0.8}}`),
validated strictly (unknown keys are rejected).

Supported formats are a config mapping from extension to parser class, so handling a new
extension with an existing parser is config only:

```yaml
parsers:
  .log: app.ingestion.parsers.text:TextParser
  .csv: {class: app.ingestion.parsers.text:TextParser, options: {encoding: latin-1}}
```

A genuinely new format needs one small class with `parse(path) -> list[Block]`, referenced
by the same mapping; the registry and pipeline do not change.

## Offline model provisioning

The API needs three fastembed models (dense `BAAI/bge-small-en-v1.5`, sparse `Qdrant/bm25`,
cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2`). They are loaded from
`models/local/<org>__<name>/` when present, otherwise from the Hugging Face cache.
`scripts/download_models.py` provisions them and falls back when the Hugging Face CDN is
blocked:

- dense: Qdrant's public GCS tarball (tokenizer config patched to 512 tokens)
- cross-encoder: `scripts/export_reranker_onnx.sh` exports the original
  `cross-encoder/ms-marco-MiniLM-L-6-v2` weights from Weaviate's reranker image to ONNX;
  the ONNX logits match torch exactly (8.7186 / -11.1401 on a sanity pair)
- BM25: only small stopword files, which the hub serves directly

## Evaluation

Two unrelated public-domain corpora, one shared config, no per-corpus tuning:

| corpus | documents | questions |
|---|---|---|
| tech | NIST CSF 2.0 (PDF, 32 pp.), PEP 8 (`.rst` via config mapping), PEP 257 (Markdown) | 15 answerable, 5 unanswerable |
| humanities | Sun Tzu, *The Art of War* with Giles' commentary (TXT), Epictetus, *Enchiridion* (DOCX) | 15 answerable, 5 unanswerable |

Each answerable question has a gold source, a verbatim evidence phrase and a reference
answer; unanswerable questions are tagged `near` (plausible, in-domain) or `far`.

```bash
python eval/fetch_corpora.py      # already done; corpora are committed under eval/corpora/
python eval/run_eval.py           # retrieval, refusal, citation verifier: no LLM key needed
python eval/run_eval.py --llm     # + end-to-end answers with the configured model
```

Latest results ([eval/results/results.md](eval/results/results.md)):

| | tech | humanities |
|---|---|---|
| hit@1 / hit@5 / MRR, dense only | 0.60 / 0.93 / 0.72 | 0.60 / 0.87 / 0.68 |
| hit@1 / hit@5 / MRR, BM25 only | 0.93 / 1.00 / 0.97 | 0.47 / 0.80 / 0.60 |
| hit@1 / hit@5 / MRR, hybrid RRF | 0.67 / 1.00 / 0.82 | 0.73 / 0.87 / 0.79 |
| hit@1 / hit@5 / MRR, hybrid + rerank (shipped) | 0.73 / 1.00 / 0.84 | 0.80 / 0.80 / 0.82 |
| refusal guard at 0.6: answer rate / refusal rate | 1.00 / 1.00 | 0.93 / 1.00 |
| threshold tuned on the *other* corpus: balanced accuracy | 1.00 | 0.93 |
| citation verifier AUC (correct vs mismatched claim/source) | 1.00 | 1.00 |

What this shows:

- **No single retriever wins everywhere.** BM25 is best on the tech corpus (exact terms
  like "79 characters") and worst on the humanities corpus (paraphrased questions about
  archaic prose); dense is the reverse. Hybrid is never the worst, and reranking gives the
  best hit@1 and MRR on both corpora.
- **The refusal threshold transfers.** Tuned on either corpus it scores 0.93 to 1.0
  balanced accuracy on the other. Off-domain questions score ~0.00; near-domain ones
  (0.28 to 0.54) are the hard cases and sit below 0.6.
- **Citation verification separates cleanly.** Correct pairs score >= 0.98, mismatched
  pairs <= 0.76, hence `citations.min_score: 0.5`.

Known gaps from the eval: the MiniLM cross-encoder demotes 3 humanities passages out of
the top 5 (Giles' bracketed commentary and 19th-century phrasing), and one answerable
humanities question is refused (top score 0.11). A larger reranker (e.g.
`BAAI/bge-reranker-base`) is a one-line config change where it can be downloaded. The
end-to-end LLM table was not produced in the build environment (no API key); run
`--llm` to add it.

## Layout

```
backend/app/
  core/interfaces.py     Protocols for every step + shared dataclasses
  ingestion/             parsers/ (registry, pdf, docx, markdown, text), chunker, pipeline, worker
  retrieval/             fastembed dense + BM25, Qdrant store (RRF fusion), retriever, cross-encoder
  generation/            prompt, LiteLLM client, evidence guard, citation verifier, answerer
  storage/               SQLite schema + repository (KBs, documents, durable job queue)
  api/                   FastAPI routers and schemas; main.py wires the app
backend/tests/           unit + integration (fakes, Qdrant local mode, LiteLLM mock, real models)
ui/                      Streamlit app + rendering components and tests
eval/                    corpora, question sets, fetch + run scripts, results
scripts/                 model provisioning
```

## Limitations

No auth or multi-tenancy; no OCR (scanned PDFs fail with a clear error); tables and images
are not extracted; DOCX has no page numbers (sections only); one worker thread ingests
documents sequentially.
