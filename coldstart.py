import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.coldstart import run_coldstart_demo

if __name__ == "__main__":
    run_coldstart_demo()
