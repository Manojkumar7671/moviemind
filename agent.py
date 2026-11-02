import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.agent import run_agent_demo

if __name__ == "__main__":
    run_agent_demo()
