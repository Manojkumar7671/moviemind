FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, precomputed artifacts, evaluations, and data
COPY data/ data/
COPY artifacts/ artifacts/
COPY src/ src/
COPY app/ app/
COPY evals/ evals/
COPY docs/ docs/
COPY pytest.ini .
COPY tests/ tests/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
