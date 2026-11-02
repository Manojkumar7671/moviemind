import os
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_PATH = DATA_DIR / "ml-1m.zip"
EXTRACT_DIR = DATA_DIR / "ml-1m"

def download_movielens_1m(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (EXTRACT_DIR / "ratings.dat").exists() or force:
        print(f"Downloading MovieLens 1M from {URL}...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"Extracted to {EXTRACT_DIR}")
    else:
        print(f"MovieLens 1M already exists at {EXTRACT_DIR}")
    return EXTRACT_DIR

if __name__ == "__main__":
    download_movielens_1m()
