import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.memory import run_memory_demo

if __name__ == "__main__":
    run_memory_demo()
