import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.ranker import train_and_rank

if __name__ == "__main__":
    train_and_rank()
