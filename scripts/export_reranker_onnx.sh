#!/usr/bin/env bash
# Offline fallback for the cross-encoder when huggingface.co's CDN is unreachable.
# Weaviate's reranker image bundles the original cross-encoder/ms-marco-MiniLM-L-6-v2 weights
# (safetensors) plus torch; export them to ONNX where fastembed expects them.
# ONNX output matches the torch logits (e.g. 8.7186 / -11.1401 on a sanity pair).
set -euo pipefail

IMAGE=semitechnologies/reranker-transformers:cross-encoder-ms-marco-MiniLM-L-6-v2-1.1.2
OUT="$(cd "$(dirname "$0")/.." && pwd)/models/local/Xenova__ms-marco-MiniLM-L-6-v2"
mkdir -p "$OUT/onnx" && chmod 777 "$OUT" "$OUT/onnx"

docker run --rm -v "$OUT:/out" --entrypoint sh "$IMAGE" -c '
pip install -q onnx==1.16.2 && python - <<EOF
import shutil, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
p = "/app/models/model"
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForSequenceClassification.from_pretrained(p).eval()
names = ["input_ids", "attention_mask", "token_type_ids"]
enc = tok(["query"], ["a passage"], return_tensors="pt")
torch.onnx.export(
    model, tuple(enc[k] for k in names), "/out/onnx/model.onnx",
    input_names=names, output_names=["logits"], opset_version=14,
    dynamic_axes={**{k: {0: "batch", 1: "seq"} for k in names}, "logits": {0: "batch"}},
)
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]:
    shutil.copy(f"{p}/{f}", f"/out/{f}")
EOF'
echo "exported to $OUT"
