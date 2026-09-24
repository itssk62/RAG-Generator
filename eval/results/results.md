# Eval results

Config: guard.min_top_score=0.6, citations.min_score=0.5, top_n=5, models={'dense': 'BAAI/bge-small-en-v1.5', 'sparse': 'Qdrant/bm25', 'reranker': 'Xenova/ms-marco-MiniLM-L-6-v2', 'llm': 'gpt-4o-mini'}

## Retrieval (answerable questions)

| corpus | mode | hit@1 | hit@5 | MRR | misses |
|---|---|---|---|---|---|
| tech | dense | 0.6 | 0.933 | 0.717 | t01 |
| tech | sparse (BM25) | 0.933 | 1 | 0.967 | - |
| tech | hybrid (RRF) | 0.667 | 1 | 0.822 | - |
| tech | hybrid + rerank | 0.733 | 1 | 0.844 | - |
| humanities | dense | 0.6 | 0.867 | 0.683 | h01, h05 |
| humanities | sparse (BM25) | 0.467 | 0.8 | 0.604 | h04, h13, h14 |
| humanities | hybrid (RRF) | 0.733 | 0.867 | 0.788 | h05, h14 |
| humanities | hybrid + rerank | 0.8 | 0.8 | 0.821 | h05, h13, h14 |

## Refusal guard

| corpus | threshold | answer rate (answerable) | refusal rate (unanswerable) | balanced acc |
|---|---|---|---|---|
| tech | 0.6 (configured) | 1 | 1 | 1.0 |
| humanities | 0.6 (configured) | 0.933 | 1 | 0.967 |
| tuned on tech tested on humanities | 0.75 | 0.867 | 1 | 0.933 |
| tuned on humanities tested on tech | 0.6 | 1 | 1 | 1.0 |

Best joint threshold (middle of the best plateau): 0.6

Top rerank score per unanswerable question:

- tech: tu1 (near) 0.282, tu2 (near) 0.540, tu3 (near) 0.038, tu4 (far) 0.000, tu5 (far) 0.000; lowest answerable 0.976
- humanities: hu1 (near) 0.000, hu2 (near) 0.429, hu3 (near) 0.522, hu4 (far) 0.000, hu5 (far) 0.000; lowest answerable 0.108

## Citation verifier (claim vs source chunk)

| corpus | pairs | supported when correct | supported when mismatched | AUC | lowest correct | highest mismatched |
|---|---|---|---|---|---|---|
| tech | 15 | 1 | 0.133 | 1 | 0.9848 | 0.7596 |
| humanities | 15 | 1 | 0 | 1 | 0.98 | 0.0011 |

## Ingestion

- tech: {'nist_csf_2.0.pdf': 61, 'pep257_docstrings.md': 14, 'pep8_style_guide.rst': 58} chunks in 5.4 s
- humanities: {'art_of_war.txt': 340, 'enchiridion.docx': 86} chunks in 21.0 s
