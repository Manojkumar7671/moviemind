#!/usr/bin/env python3
"""Fetch TMDB poster paths for the World Cinema catalog using IMDb IDs."""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT_DIR / "artifacts" / "global_catalog.parquet"
OUTPUT_PATH = ROOT_DIR / "app" / "static" / "world_posters.json"


def get_tmdb_token() -> str:
    load_dotenv(ROOT_DIR / ".env")
    token = os.environ.get("TMDB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TMDB_TOKEN not found in environment or .env")
    return token


def fetch_poster_for_imdb(imdb_id: str, token: str, timeout: float = 10.0, retries: int = 3) -> tuple[str, str | None]:
    url = f"https://api.themoviedb.org/3/find/{imdb_id}?external_source=imdb_id"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("movie_results") or []
                if results:
                    poster_path = results[0].get("poster_path")
                    if poster_path:
                        return imdb_id, poster_path
                return imdb_id, None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = float(e.headers.get("Retry-After", 1.5))
                time.sleep(min(retry_after, 5.0))
            elif e.code == 404:
                return imdb_id, None
            elif attempt < retries - 1:
                time.sleep(0.5 * (2 ** attempt))
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5 * (2 ** attempt))
    return imdb_id, None


def main():
    token = get_tmdb_token()
    df = pd.read_parquet(CATALOG_PATH, columns=["imdb_id"])
    imdb_ids = [str(x).strip() for x in df["imdb_id"].dropna().unique() if str(x).strip()]

    posters: dict[str, str] = {}
    if OUTPUT_PATH.exists():
        try:
            posters = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            posters = {}

    todo = [iid for iid in imdb_ids if iid not in posters]
    total = len(imdb_ids)
    print(f"Total IMDb IDs: {total}, already cached: {len(posters)}, to fetch: {len(todo)}", flush=True)

    lock = threading.Lock()
    done_count = 0
    found_count = len(posters)
    start_t = time.time()

    def save_checkpoint():
        tmp = OUTPUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(posters, separators=(",", ":")), encoding="utf-8")
        tmp.replace(OUTPUT_PATH)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_poster_for_imdb, iid, token): iid for iid in todo}
        for fut in as_completed(futures):
            iid, poster_path = fut.result()
            with lock:
                done_count += 1
                if poster_path:
                    posters[iid] = poster_path
                    found_count += 1
                if done_count % 500 == 0 or done_count == len(todo):
                    elapsed = time.time() - start_t
                    rate = done_count / elapsed if elapsed > 0 else 0.0
                    print(
                        f"Progress: {done_count}/{len(todo)} fetched ({found_count} posters found, {rate:.1f}/s)",
                        flush=True,
                    )
                if done_count % 2500 == 0:
                    save_checkpoint()

    save_checkpoint()
    print(f"Done! Saved {len(posters)} world posters to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
