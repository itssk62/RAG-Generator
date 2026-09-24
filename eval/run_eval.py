"""Evaluate the pipeline on two unrelated corpora with one shared config.

Runs the real app (HTTP API -> job queue -> parsers -> fastembed -> Qdrant local mode) and reports:
  1. retrieval  hit@1 / hit@5 / MRR for dense, sparse, hybrid and hybrid+rerank
  2. refusal    the guard on answerable vs unanswerable questions, a threshold sweep, and
                leave-one-corpus-out calibration (tune on one corpus, test on the other)
  3. citations  verifier separation of correct vs mismatched claim/source pairs
  4. end-to-end (only with --llm) answers, refusals, keyword recall, citation verification

Usage:  python eval/run_eval.py            # retrieval, refusal, verifier (no LLM key needed)
        python eval/run_eval.py --llm      # also full answers with the configured LiteLLM model
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean

import yaml

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import RetrievalSettings, Settings  # noqa: E402
from app.container import Container, build_container  # noqa: E402
from app.core.interfaces import Hit  # noqa: E402
from app.main import create_app  # noqa: E402

CORPORA = ["tech", "humanities"]
MODES = {
    "dense": {"mode": "dense", "rerank": False},
    "sparse (BM25)": {"mode": "sparse", "rerank": False},
    "hybrid (RRF)": {"mode": "hybrid", "rerank": False},
    "hybrid + rerank": {"mode": "hybrid", "rerank": True},
}
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "`": ""})


def norm(text: str) -> str:
    return " ".join(text.lower().translate(_QUOTES).split())


def is_gold(hit: Hit, q: dict) -> bool:
    return hit.chunk.source == q["source"] and norm(q["evidence"]) in norm(hit.chunk.text)


class NoLLM:
    def complete(self, messages):
        raise RuntimeError("run with --llm to generate answers")

    def stream(self, messages):
        raise RuntimeError("run with --llm to generate answers")


def load(corpus: str) -> dict:
    spec = yaml.safe_load((EVAL / f"questions_{corpus}.yaml").read_text())
    for q in spec["questions"]:
        q.setdefault("answerable", True)
    return spec


def ingest(client: TestClient, c: Container, spec: dict) -> tuple[str, dict]:
    kb = client.post("/kbs", json={"name": spec["corpus"]}).json()["id"]
    folder = EVAL / "corpora" / spec["corpus"]
    t0 = time.perf_counter()
    r = client.post(f"/kbs/{kb}/documents", files=[("files", (f, (folder / f).read_bytes())) for f in spec["files"]])
    r.raise_for_status()
    c.worker.drain()
    docs = client.get(f"/kbs/{kb}/documents").json()
    failed = [d for d in docs if d["status"] != "ready"]
    if failed:
        raise SystemExit(f"ingestion failed: {failed}")
    return kb, {
        "seconds": round(time.perf_counter() - t0, 1),
        "documents": {d["filename"]: d["chunk_count"] for d in docs},
    }


def evidence_check(c: Container, kb: str, questions: list[dict]) -> list[str]:
    """Every gold evidence phrase must exist in at least one indexed chunk."""
    points, offset = [], None
    while True:
        batch, offset = c.store.client.scroll(f"{c.store.prefix}{kb}", limit=256, offset=offset, with_payload=True)
        points += batch
        if offset is None:
            break
    texts = [(p.payload["source"], norm(p.payload["text"])) for p in points]
    return [q["id"] for q in questions if q["answerable"] and not any(s == q["source"] and norm(q["evidence"]) in t for s, t in texts)]


def retrieval(c: Container, kb: str, questions: list[dict], base: RetrievalSettings) -> dict:
    out = {}
    for name, overrides in MODES.items():
        cfg = base.model_copy(update=overrides)
        ranks = []
        for q in (q for q in questions if q["answerable"]):
            hits = c.retriever.candidates(kb, q["question"], cfg)
            if cfg.rerank:
                hits = c.retriever.rerank(q["question"], hits, len(hits))
            ranks.append(next((i for i, h in enumerate(hits, 1) if is_gold(h, q)), None))
        out[name] = {
            "hit@1": round(mean(r == 1 for r in ranks), 3),
            f"hit@{base.top_n}": round(mean(r is not None and r <= base.top_n for r in ranks), 3),
            "mrr": round(mean(1 / r if r else 0 for r in ranks), 3),
            "misses": [q["id"] for q, r in zip([q for q in questions if q["answerable"]], ranks) if not r or r > base.top_n],
        }
    return out


def top_scores(c: Container, kb: str, questions: list[dict], cfg: RetrievalSettings) -> list[dict]:
    rows = []
    for q in questions:
        hits = c.retriever.retrieve(kb, q["question"], cfg)
        rows.append({"id": q["id"], "answerable": q["answerable"], "kind": q.get("kind"), "top": max((h.rerank_score or 0 for h in hits), default=0.0)})
    return rows


def refusal_at(rows: list[dict], threshold: float) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = mean(r["top"] >= threshold for r in ans)
    refused = mean(r["top"] < threshold for r in una)
    return {"threshold": threshold, "answer_rate": round(answered, 3), "refusal_rate": round(refused, 3), "balanced_acc": round((answered + refused) / 2, 3)}


def best_threshold(rows: list[dict]) -> float:
    scored = [refusal_at(rows, t) for t in THRESHOLDS]
    best = max(s["balanced_acc"] for s in scored)
    plateau = [s["threshold"] for s in scored if s["balanced_acc"] == best]
    return plateau[len(plateau) // 2]  # middle of the best plateau: least sensitive choice


def verifier_probe(c: Container, kb: str, questions: list[dict], cfg: RetrievalSettings, min_score: float) -> dict:
    gold = {}
    for q in (q for q in questions if q["answerable"]):
        hits = c.retriever.rerank(q["question"], c.retriever.candidates(kb, q["question"], cfg), cfg.fused_k)
        match = next((h for h in hits if is_gold(h, q)), None)
        if match:
            gold[q["id"]] = (q, match.chunk.text)
    ids = list(gold)
    pos = [c.retriever.reranker.score(gold[i][0]["answer"], [gold[i][1]])[0] for i in ids]
    neg = [c.retriever.reranker.score(gold[i][0]["answer"], [gold[ids[(k + 1) % len(ids)]][1]])[0] for k, i in enumerate(ids)]
    auc = mean(p > n for p in pos for n in neg)
    return {
        "pairs": len(ids),
        "supported_when_correct": round(mean(s >= min_score for s in pos), 3),
        "supported_when_mismatched": round(mean(s >= min_score for s in neg), 3),
        "auc": round(auc, 3),
        "lowest_correct": round(min(pos), 4),
        "highest_mismatched": round(max(neg), 4),
    }


def end_to_end(c: Container, kb: str, questions: list[dict], settings: Settings) -> dict:
    cfg = settings.kb_config()
    rows = []
    for q in questions:
        a = c.answerer.answer(kb, q["question"], cfg)
        checks = a.verification.checks if a.verification else []
        text = norm(a.text)
        rows.append(
            {
                "id": q["id"],
                "answerable": q["answerable"],
                "refused": a.refused,
                "keyword_recall": mean(norm(k) in text for k in q.get("keywords", [])) if q["answerable"] and q.get("keywords") else None,
                "citations": len(checks),
                "supported": sum(ch.status == "supported" for ch in checks),
                "invalid": sum(ch.status == "invalid" for ch in checks),
                "verified": bool(a.verification and a.verification.ok),
                "ms": sum(a.timings_ms.values()),
                "answer": a.text,
            }
        )
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    total_cites = sum(r["citations"] for r in answered)
    return {
        "answer_rate": round(mean(not r["refused"] for r in ans), 3),
        "refusal_rate_unanswerable": round(mean(r["refused"] for r in una), 3),
        "keyword_recall": round(mean(r["keyword_recall"] for r in answered if r["keyword_recall"] is not None), 3) if answered else None,
        "citation_support_rate": round(sum(r["supported"] for r in answered) / total_cites, 3) if total_cites else None,
        "invalid_citations": sum(r["invalid"] for r in answered),
        "fully_verified_answers": round(mean(r["verified"] for r in answered), 3) if answered else None,
        "median_latency_ms": sorted(r["ms"] for r in rows)[len(rows) // 2],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also generate answers with the configured LLM")
    parser.add_argument("--out", default=str(EVAL / "results"))
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="rag-eval-")
    settings = Settings(data_dir=tmp, qdrant={"url": ":memory:"})
    c = build_container(settings, llm=None if args.llm else NoLLM())
    rcfg = settings.retrieval
    results: dict = {"config": {"retrieval": rcfg.model_dump(), "guard": settings.guard.model_dump(), "citations": settings.citations.model_dump(), "models": {k: v for k, v in settings.models.model_dump().items() if k in ("dense", "sparse", "reranker", "llm")}}, "corpora": {}}
    scores: dict[str, list[dict]] = {}

    with TestClient(create_app(c, start_worker=False)) as client:
        for corpus in CORPORA:
            spec = load(corpus)
            qs = spec["questions"]
            kb, ingest_stats = ingest(client, c, spec)
            print(f"[{corpus}] ingested {ingest_stats}")
            missing = evidence_check(c, kb, qs)
            if missing:
                print(f"[{corpus}] WARNING evidence not found in any chunk: {missing}")
            scores[corpus] = top_scores(c, kb, qs, rcfg)
            results["corpora"][corpus] = {
                "description": spec["description"],
                "questions": {"answerable": sum(q["answerable"] for q in qs), "unanswerable": sum(not q["answerable"] for q in qs)},
                "ingest": ingest_stats,
                "evidence_missing": missing,
                "retrieval": retrieval(c, kb, qs, rcfg),
                "refusal_at_configured": refusal_at(scores[corpus], settings.guard.min_top_score),
                "top_scores": scores[corpus],
                "citation_verifier": verifier_probe(c, kb, qs, rcfg, settings.citations.min_score),
            }
            if args.llm:
                print(f"[{corpus}] generating answers with {settings.models.llm} ...")
                results["corpora"][corpus]["end_to_end"] = end_to_end(c, kb, qs, settings)

    a, b = CORPORA
    results["calibration"] = {
        "sweep": {corpus: [refusal_at(scores[corpus], t) for t in THRESHOLDS] for corpus in CORPORA},
        "best_joint": best_threshold(scores[a] + scores[b]),
        f"tuned_on_{a}_tested_on_{b}": refusal_at(scores[b], best_threshold(scores[a])),
        f"tuned_on_{b}_tested_on_{a}": refusal_at(scores[a], best_threshold(scores[b])),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "results.md").write_text(report(results))
    print(report(results))
    return 0


def report(r: dict) -> str:
    lines = ["# Eval results", "", f"Config: guard.min_top_score={r['config']['guard']['min_top_score']}, citations.min_score={r['config']['citations']['min_score']}, "
             f"top_n={r['config']['retrieval']['top_n']}, models={r['config']['models']}", ""]
    top_n = r["config"]["retrieval"]["top_n"]
    lines += ["## Retrieval (answerable questions)", "", f"| corpus | mode | hit@1 | hit@{top_n} | MRR | misses |", "|---|---|---|---|---|---|"]
    for corpus, res in r["corpora"].items():
        for mode, m in res["retrieval"].items():
            lines.append(f"| {corpus} | {mode} | {m['hit@1']} | {m[f'hit@{top_n}']} | {m['mrr']} | {', '.join(m['misses']) or '-'} |")
    lines += ["", "## Refusal guard", "", "| corpus | threshold | answer rate (answerable) | refusal rate (unanswerable) | balanced acc |", "|---|---|---|---|---|"]
    for corpus, res in r["corpora"].items():
        m = res["refusal_at_configured"]
        lines.append(f"| {corpus} | {m['threshold']} (configured) | {m['answer_rate']} | {m['refusal_rate']} | {m['balanced_acc']} |")
    cal = r["calibration"]
    for key, m in cal.items():
        if key.startswith("tuned_on"):
            lines.append(f"| {key.replace('_', ' ')} | {m['threshold']} | {m['answer_rate']} | {m['refusal_rate']} | {m['balanced_acc']} |")
    lines += ["", f"Best joint threshold (middle of the best plateau): {cal['best_joint']}", ""]
    lines += ["Top rerank score per unanswerable question:", ""]
    for corpus, res in r["corpora"].items():
        una = ", ".join(f"{s['id']} ({s['kind']}) {s['top']:.3f}" for s in res["top_scores"] if not s["answerable"])
        ans_min = min(s["top"] for s in res["top_scores"] if s["answerable"])
        lines.append(f"- {corpus}: {una}; lowest answerable {ans_min:.3f}")
    lines += ["", "## Citation verifier (claim vs source chunk)", "",
              "| corpus | pairs | supported when correct | supported when mismatched | AUC | lowest correct | highest mismatched |", "|---|---|---|---|---|---|---|"]
    for corpus, res in r["corpora"].items():
        v = res["citation_verifier"]
        lines.append(f"| {corpus} | {v['pairs']} | {v['supported_when_correct']} | {v['supported_when_mismatched']} | {v['auc']} | {v['lowest_correct']} | {v['highest_mismatched']} |")
    if any("end_to_end" in res for res in r["corpora"].values()):
        lines += ["", "## End to end (LLM)", "", "| corpus | answer rate | refusal rate (unanswerable) | keyword recall | citation support | invalid | fully verified | median ms |", "|---|---|---|---|---|---|---|---|"]
        for corpus, res in r["corpora"].items():
            e = res.get("end_to_end")
            if e:
                lines.append(f"| {corpus} | {e['answer_rate']} | {e['refusal_rate_unanswerable']} | {e['keyword_recall']} | {e['citation_support_rate']} | {e['invalid_citations']} | {e['fully_verified_answers']} | {e['median_latency_ms']} |")
    lines += ["", "## Ingestion", ""]
    for corpus, res in r["corpora"].items():
        lines.append(f"- {corpus}: {res['ingest']['documents']} chunks in {res['ingest']['seconds']} s")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
