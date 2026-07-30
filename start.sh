#!/bin/sh
set -eu

exec python -m reg_rag.web_app \
  --host "${REG_RAG_HOST:-0.0.0.0}" \
  --port "${PORT:-7860}"
