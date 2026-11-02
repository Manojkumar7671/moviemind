import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.prepare import prepare_data

if __name__ == "__main__":
    prepare_data()
