import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.ab_sim import run_ab_simulation

if __name__ == "__main__":
    run_ab_simulation()
