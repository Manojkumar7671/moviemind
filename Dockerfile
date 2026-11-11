FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PORT=7860

# 1. Install runtime dependencies only (torch excluded from runtime image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy scripts, source code, and tracked Wikidata cache for catalog building
COPY scripts/ scripts/
COPY src/ src/
COPY artifacts/wikidata_cache.json artifacts/wikidata_cache.json

# 3. Build world catalog and precompute derived TF-IDF artifacts
# Downloads IMDb TSVs, builds global_catalog.parquet and derived artifacts, then cleans raw TSVs
RUN python scripts/build_catalog.py --clean-raw && \
    python scripts/build_global_tfidf.py && \
    rm -rf /tmp/title.*.tsv.gz /tmp/*.tmp && \
    (test -s artifacts/global_catalog.parquet || (echo "FATAL: artifacts/global_catalog.parquet is missing or empty" && exit 1)) && \
    (test -s artifacts/global_tfidf.npz || (echo "FATAL: artifacts/global_tfidf.npz is missing or empty" && exit 1)) && \
    (test -s artifacts/global_vectorizer.pkl || (echo "FATAL: artifacts/global_vectorizer.pkl is missing or empty" && exit 1)) && \
    (test -s artifacts/global_lookups.json || (echo "FATAL: artifacts/global_lookups.json is missing or empty" && exit 1))

# 4. Copy rest of the application code, precomputed artifacts, evaluations, and data
COPY data/ data/
COPY artifacts/ artifacts/
COPY app/ app/
COPY evals/ evals/

# 5. Non-root user setup (Hugging Face Spaces default UID 1000)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python3 -c "import urllib.request, os; p = os.getenv('PORT', '7860'); urllib.request.urlopen(f'http://localhost:{p}/healthz')" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
