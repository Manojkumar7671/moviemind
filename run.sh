#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

CMD="${1:-serve}"

case "$CMD" in
    serve)
        echo "Starting MovieMind App at http://localhost:8000 ..."
        uvicorn app.main:app --host 0.0.0.0 --port 8000
        ;;
    test)
        echo "Running test suite..."
        pytest -v
        ;;
    pipeline)
        echo "Running full MovieMind offline pipeline..."
        python prepare.py
        python baselines.py
        python retrieval.py
        python ranker.py
        python diversity.py
        python ab_sim.py
        python evaluate.py
        ;;
    *)
        echo "Usage: ./run.sh [serve|test|pipeline]"
        exit 1
        ;;
esac
