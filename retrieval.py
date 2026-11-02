import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.retrieval import train_and_export_retrieval

if __name__ == "__main__":
    train_and_export_retrieval()
