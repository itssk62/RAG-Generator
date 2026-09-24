"""Make sure the three fastembed models are available before the API starts.

For each model: use `<cache>/local/<org>__<name>` if present, else let fastembed fetch it from
Hugging Face into the cache, else try a known mirror. Exits non-zero with instructions when a
model cannot be provisioned, so a container fails fast instead of on the first request.

Env: RAG__MODELS__CACHE_DIR (default ./models), RAG__MODELS__DENSE, RAG__MODELS__SPARSE, RAG__MODELS__RERANKER
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

CACHE = Path(os.environ.get("RAG__MODELS__CACHE_DIR", "./models"))
DENSE = os.environ.get("RAG__MODELS__DENSE", "BAAI/bge-small-en-v1.5")
SPARSE = os.environ.get("RAG__MODELS__SPARSE", "Qdrant/bm25")
RERANKER = os.environ.get("RAG__MODELS__RERANKER", "Xenova/ms-marco-MiniLM-L-6-v2")

# Tarball mirrors published by Qdrant (used when the Hugging Face CDN is unreachable).
# `max_length` patches tokenizer_config.json, which these older tarballs leave without a limit.
MIRRORS = {
    "BAAI/bge-small-en-v1.5": {"url": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz", "max_length": 512},
}


def local_dir(model: str) -> Path:
    return CACHE / "local" / model.replace("/", "__")


def from_mirror(model: str) -> bool:
    mirror = MIRRORS.get(model)
    if not mirror:
        return False
    print(f"  fetching {model} from {mirror['url']}")
    with urllib.request.urlopen(mirror["url"], timeout=300) as r:
        data = r.read()
    target = local_dir(model)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if member.isfile() and ".." not in name.parts:
                (target / name.name).write_bytes(tar.extractfile(member).read())
    tok = target / "tokenizer_config.json"
    if "max_length" in mirror and tok.exists():
        cfg = json.loads(tok.read_text())
        if not 0 < int(cfg.get("model_max_length") or 0) <= 100_000:  # HF writes 1e30 for "unset"
            cfg["model_max_length"] = mirror["max_length"]
        tok.write_text(json.dumps(cfg, indent=2))
    return True


def ensure(model: str, loader) -> bool:
    if local_dir(model).is_dir():
        try:
            loader(model, str(local_dir(model)))
            print(f"ok   {model} (local: {local_dir(model)})")
            return True
        except Exception as e:
            print(f"warn {model}: local copy at {local_dir(model)} does not load ({type(e).__name__}: {e})")
    try:
        loader(model)
        print(f"ok   {model} (cache: {CACHE})")
        return True
    except Exception as e:  # network or hub errors
        print(f"warn {model}: hub download failed ({type(e).__name__})")
    try:
        if from_mirror(model):
            loader(model, str(local_dir(model)))
            print(f"ok   {model} (mirror -> {local_dir(model)})")
            return True
    except Exception as e:
        print(f"warn {model}: mirror failed ({type(e).__name__}: {e})")
    print(
        f"FAIL {model}: place the ONNX model files in {local_dir(model)} "
        "(see README, 'Offline model provisioning') or fix access to huggingface.co."
    )
    return False


def main() -> int:
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    def dense(m, path=None):
        TextEmbedding(m, cache_dir=str(CACHE), **({"specific_model_path": path} if path else {}))

    def sparse(m, path=None):
        SparseTextEmbedding(m, cache_dir=str(CACHE), **({"specific_model_path": path} if path else {}))

    def rerank(m, path=None):
        TextCrossEncoder(m, cache_dir=str(CACHE), **({"specific_model_path": path} if path else {}))

    CACHE.mkdir(parents=True, exist_ok=True)
    results = [ensure(DENSE, dense), ensure(SPARSE, sparse), ensure(RERANKER, rerank)]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
