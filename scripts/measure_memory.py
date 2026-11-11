#!/usr/bin/env python3
"""scripts/measure_memory.py

Measures MovieMind uvicorn startup RSS and provides a per-component breakdown
of memory consumption (MovieLens artifacts, torch import, models, FAISS index,
World catalog parquet, TF-IDF matrix, and candidate caches).
"""

import gc
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"


def get_pid_rss_mb(pid: int) -> float:
    """Measure Resident Set Size (RSS) in megabytes for a given PID on macOS/Linux."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).decode().strip()
        return round(int(out) / 1024.0, 2)
    except Exception as e:
        print(f"Error measuring PID {pid} RSS: {e}", file=sys.stderr)
        return 0.0


def measure_uvicorn_startup(port: int = 8765, timeout_s: float = 25.0) -> float:
    """Start uvicorn app.main:app in a subprocess, wait for /healthz, and measure RSS."""
    venv_python = ROOT_DIR / ".venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [python_exe, "-m", "uvicorn", "app.main:app", "--port", str(port), "--host", "127.0.0.1"]
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    rss_mb = 0.0
    start_time = time.time()
    healthy = False

    try:
        while time.time() - start_time < timeout_s:
            time.sleep(0.3)
            try:
                url = f"http://127.0.0.1:{port}/healthz"
                with urllib.request.urlopen(url, timeout=1.0) as res:
                    if res.status == 200:
                        healthy = True
                        break
            except Exception:
                pass

        if not healthy:
            raise TimeoutError(f"Uvicorn failed to respond on port {port} within {timeout_s} seconds.")

        # Allow process memory to settle
        time.sleep(0.5)
        rss_mb = get_pid_rss_mb(proc.pid)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    return rss_mb


def measure_component_breakdown() -> list:
    """Measure RSS delta around individual subsystem loads in an isolated subprocess."""
    script = """
import os, sys, gc, subprocess
from pathlib import Path

ROOT_DIR = Path.cwd()
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"

def get_current_rss_mb():
    gc.collect()
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()
    return int(out) / 1024.0

results = []

# Baseline
base_rss = get_current_rss_mb()
results.append(("Python runtime baseline", base_rss))

# 1. MovieLens artifacts
r0 = get_current_rss_mb()
import pandas as pd
import numpy as np
movies_df = pd.read_parquet(DATA_DIR / "movies.parquet")
item_embs = np.load(ARTIFACTS_DIR / "item_embeddings.npy")
item_ids = np.load(ARTIFACTS_DIR / "item_ids.npy")
r1 = get_current_rss_mb()
results.append(("MovieLens artifacts (movies.parquet, item embeddings)", r1 - r0))

# 2. FAISS index
r0 = get_current_rss_mb()
import faiss
faiss_idx = faiss.read_index(str(ARTIFACTS_DIR / "faiss_index.bin"))
r1 = get_current_rss_mb()
results.append(("FAISS index (faiss_index.bin)", r1 - r0))

# 3. MMR candidates cache
r0 = get_current_rss_mb()
df_mmr = pd.read_parquet(ARTIFACTS_DIR / "mmr_candidates.parquet")
mmr_cache = {int(r["user_id"]): list(r["candidate_ids"]) for _, r in df_mmr.iterrows()}
r1 = get_current_rss_mb()
results.append(("MMR candidates (mmr_candidates.parquet)", r1 - r0))

# 4. World catalog parquet
r0 = get_current_rss_mb()
gdf = pd.read_parquet(ARTIFACTS_DIR / "global_catalog.parquet")
r1 = get_current_rss_mb()
results.append(("World catalog parquet (global_catalog.parquet)", r1 - r0))

# 5. TF-IDF matrix
r0 = get_current_rss_mb()
from sklearn.feature_extraction.text import TfidfVectorizer
feature_texts = []
for _, r in gdf.iterrows():
    genres = " ".join([f"g_{g.strip().lower()}" for g in r["genres"].split("|") if g.strip()])
    langs = " ".join([f"l_{str(l).strip().lower()}" for l in r["original_languages"] if str(l).strip()])
    countries = " ".join([f"c_{str(c).strip().lower()}" for c in r["origin_countries"] if str(c).strip()])
    tokens = f"{genres} {langs} {countries}".strip()
    feature_texts.append(tokens if tokens else "empty")
vec = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
tfidf_matrix = vec.fit_transform(feature_texts)
r1 = get_current_rss_mb()
results.append(("TF-IDF matrix & vectorizer", r1 - r0))

# 6. PyTorch import
r0 = get_current_rss_mb()
import torch
r1 = get_current_rss_mb()
results.append(("torch import (overhead)", r1 - r0))

# 7. Two-tower & ranker models
r0 = get_current_rss_mb()
tt_pt = torch.load(ARTIFACTS_DIR / "two_tower.pt", map_location="cpu")
wd_pt = torch.load(ARTIFACTS_DIR / "wide_and_deep.pt", map_location="cpu")
r1 = get_current_rss_mb()
results.append(("Two-Tower / Wide&Deep model weights (.pt)", r1 - r0))

import json
print(json.dumps(results))
"""
    venv_python = ROOT_DIR / ".venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    out = subprocess.check_output([python_exe, "-c", script], cwd=str(ROOT_DIR)).decode().strip()
    return json.loads(out)


def main():
    print("=================================================================")
    print("MovieMind Memory Profiler (scripts/measure_memory.py)")
    print("=================================================================")
    print("\nMeasuring live uvicorn app.main:app startup RSS...")
    startup_rss = measure_uvicorn_startup()
    print(f"-> Live Startup RSS: {startup_rss:.2f} MB")

    print("\nProfiling component memory breakdown in isolated process...")
    breakdown = measure_component_breakdown()

    print("\n-----------------------------------------------------------------")
    print(f"| {'Component':<50} | {'RSS Delta (MB)':>14} |")
    print("-----------------------------------------------------------------")
    for name, delta in breakdown:
        print(f"| {name:<50} | {delta:>14.2f} |")
    print("-----------------------------------------------------------------")
    print(f"| {'Total uvicorn startup RSS (measured live)':<50} | {startup_rss:>14.2f} |")
    print("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()
