FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/app/src \
    COMPETITION_PUBLIC_MODE=1 \
    REG_RAG_HOST=0.0.0.0 \
    REG_RAG_PORT=7860 \
    PORT=7860 \
    REG_RAG_RECORDS=data/synthetic/records.jsonl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --requirement requirements.txt

COPY src/ src/
COPY configs/shadow_router_v1.yaml configs/shadow_router_v1.yaml
COPY configs/system/public_demo.yaml configs/system/public_demo.yaml
COPY data/synthetic/records.jsonl data/synthetic/records.jsonl
COPY data/public_reference/ data/public_reference/
COPY data/official_benchmark/full500_semantic_v2.sqlite data/official_benchmark/full500_semantic_v2.sqlite
COPY data/public_demo/new_document_demo.docx data/public_demo/new_document_demo.docx
COPY data/demo_sources/ data/demo_sources/
COPY public_sources/ public_sources/
COPY artifacts/rag_002a/document_family_catalog.csv artifacts/rag_002a/document_family_catalog.csv
COPY artifacts/rag_002a/table_template_catalog.csv artifacts/rag_002a/table_template_catalog.csv
COPY artifacts/rag_public_hardening/public_metrics.json artifacts/rag_public_hardening/public_metrics.json
COPY start.sh start.sh

RUN chmod +x /app/start.sh \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json,os,urllib.request; port=os.environ.get('PORT','7860'); data=json.load(urllib.request.urlopen(f'http://0.0.0.0:{port}/health', timeout=4)); assert data.get('status') == 'PASS'"

CMD ["/app/start.sh"]
