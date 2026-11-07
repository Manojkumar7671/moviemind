FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PORT=7860

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user (Hugging Face Spaces default UID 1000)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy application code, precomputed artifacts, evaluations, and data
COPY --chown=appuser:appuser data/ data/
COPY --chown=appuser:appuser artifacts/ artifacts/
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser evals/ evals/
COPY --chown=appuser:appuser docs/ docs/
COPY --chown=appuser:appuser scripts/ scripts/
COPY --chown=appuser:appuser pytest.ini .
COPY --chown=appuser:appuser tests/ tests/

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python3 -c "import urllib.request, os; p = os.getenv('PORT', '7860'); urllib.request.urlopen(f'http://localhost:{p}/healthz')" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
