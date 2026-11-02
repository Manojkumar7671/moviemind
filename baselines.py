import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.baselines import run_popularity_baseline

if __name__ == "__main__":
    run_popularity_baseline()
