"""Build MovieMind World Cinema catalog from IMDb non-commercial datasets and Wikidata.

Filters: titleType=movie, numVotes>=2000, valid startYear.
Origin and language metadata sourced from Wikidata (CC0).
All countries and languages are stored as full lists (no primary country/language column).
Output: artifacts/global_catalog.parquet (<50 MB).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils import clean_title

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_catalog")

IMDB_BASE_URL = "https://datasets.imdbws.com"
FILES = ["title.ratings.tsv.gz", "title.basics.tsv.gz"]

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CACHE_PATH = ARTIFACTS_DIR / "wikidata_cache.json"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "global_catalog.parquet"
DEFAULT_MIN_VOTES = int(os.environ.get("MIN_VOTES", 2000))


def download_file(filename: str, dest_dir: Path, max_retries: int = 3, timeout_s: float = 60.0) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 1000:
        logger.info(f"Found existing {filename} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    url = f"{IMDB_BASE_URL}/{filename}"
    temp_dest = dest_dir / f"{filename}.tmp"
    logger.info(f"Downloading {url} -> {dest} (timeout={timeout_s}s, max_retries={max_retries}) ...")

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MovieMind/1.0 (Academic/Non-Commercial; +https://github.com/manojkumarmathangi/moviemind)"},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(temp_dest, "wb") as out:
                chunk_size = 1024 * 1024
                downloaded = 0
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (20 * 1024 * 1024) == 0:
                        logger.info(f"  .. downloaded {downloaded / 1e6:.1f} MB")

            temp_dest.replace(dest)
            logger.info(f"Successfully downloaded {filename} ({downloaded / 1e6:.1f} MB)")
            return dest
        except Exception as err:
            last_err = err
            if temp_dest.exists():
                temp_dest.unlink(missing_ok=True)
            if attempt < max_retries:
                wait_s = 2 ** attempt
                logger.warning(
                    f"Download attempt {attempt}/{max_retries} failed for {url}: {err}. "
                    f"Retrying in {wait_s}s..."
                )
                time.sleep(wait_s)
            else:
                logger.error(f"All {max_retries} attempts failed downloading {url}: {err}")

    raise RuntimeError(
        f"Failed to download required IMDb dataset from '{url}' after {max_retries} retries: {last_err}"
    )


def build_catalog(
    tmp_dir: Path = Path("/tmp"),
    output_path: Path = DEFAULT_OUTPUT_PATH,
    min_votes: int = DEFAULT_MIN_VOTES,
    clean_raw: bool = False,
) -> Path:
    # 1. Download or locate IMDb source files
    ratings_path = download_file("title.ratings.tsv.gz", tmp_dir)
    basics_path = download_file("title.basics.tsv.gz", tmp_dir)

    # 2. Filter ratings (numVotes >= min_votes)
    logger.info(f"Filtering ratings (numVotes >= {min_votes}) ...")
    ratings: Dict[str, tuple[float, int]] = {}
    with gzip.open(ratings_path, "rt", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                votes = int(parts[2]) if parts[2].isdigit() else 0
                if votes >= min_votes:
                    ratings[parts[0]] = (float(parts[1]), votes)
    logger.info(f"Kept {len(ratings):,} titles with >= {min_votes} votes.")

    # 3. Filter basics (titleType=movie, valid startYear)
    logger.info("Filtering title.basics ...")
    movies: Dict[str, Dict[str, Any]] = {}
    with gzip.open(basics_path, "rt", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 9:
                tconst, ttype, primary_title, _, _, start_year, _, _, genres = parts[:9]
                if tconst in ratings and ttype == "movie" and start_year != r"\N" and start_year.isdigit():
                    norm_genres = "" if genres == r"\N" else genres.replace(",", "|")
                    movies[tconst] = {
                        "title": clean_title(primary_title),
                        "year": int(start_year),
                        "genres": norm_genres,
                        "rating": ratings[tconst][0],
                        "votes": ratings[tconst][1],
                    }
    logger.info(f"Kept {len(movies):,} feature films.")

    # 4. Load Wikidata cache (CC0 origin countries and languages)
    wikidata_data: Dict[str, Dict[str, Any]] = {}
    if CACHE_PATH.exists():
        logger.info(f"Loading Wikidata metadata from {CACHE_PATH} ...")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            wikidata_data = json.load(f)
    else:
        logger.warning(f"Wikidata cache not found at {CACHE_PATH}!")

    # 5. Build final catalog records with all countries and languages
    records = []
    for mid, m in movies.items():
        wdata = wikidata_data.get(mid, {})
        countries = list(wdata.get("countries", []))
        country_names = list(wdata.get("country_labels", []))
        languages = list(wdata.get("languages", []))
        language_names = list(wdata.get("language_labels", []))

        records.append({
            "id": mid,
            "imdb_id": mid,
            "title": m["title"],
            "year": m["year"],
            "genres": m["genres"],
            "rating": m["rating"],
            "votes": m["votes"],
            "origin_countries": countries,
            "origin_country_names": country_names,
            "original_languages": languages,
            "original_language_names": language_names,
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Catalog build failed: output file {output_path} is missing or empty.")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[catalog] Successfully saved {len(df):,} movies to {output_path} ({size_mb:.2f} MB)")
    print(f"[catalog] Year range: {df['year'].min()} - {df['year'].max()}")

    # Coverage statistics
    total = len(df)
    n_countries = (df["origin_countries"].apply(len) > 0).sum()
    n_country_names = (df["origin_country_names"].apply(len) > 0).sum()
    n_languages = (df["original_languages"].apply(len) > 0).sum()
    n_lang_names = (df["original_language_names"].apply(len) > 0).sum()
    n_empty_origin = (df["origin_countries"].apply(len) == 0).sum()
    n_empty_lang = (df["original_languages"].apply(len) == 0).sum()

    print("\n=== Catalog Coverage Report ===")
    print(f"Total films:                     {total:,}")
    print(f"origin_countries (>= 1):         {n_countries:,} ({n_countries / total * 100:.2f}%)")
    print(f"origin_country_names (>= 1):     {n_country_names:,} ({n_country_names / total * 100:.2f}%)")
    print(f"original_languages (>= 1):       {n_languages:,} ({n_languages / total * 100:.2f}%)")
    print(f"original_language_names (>= 1): {n_lang_names:,} ({n_lang_names / total * 100:.2f}%)")
    print(f"Empty origin_countries:          {n_empty_origin:,} ({n_empty_origin / total * 100:.2f}%)")
    print(f"Empty original_languages:        {n_empty_lang:,} ({n_empty_lang / total * 100:.2f}%)")
    print("===============================\n")

    if clean_raw:
        logger.info(f"Cleaning raw downloaded TSVs from {tmp_dir} ...")
        for f in [ratings_path, basics_path]:
            try:
                if f.exists():
                    f.unlink()
                    logger.info(f"Removed temporary raw file: {f}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {f}: {e}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MovieMind World Cinema catalog.")
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp"), help="Directory for temporary raw TSVs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output parquet path.")
    parser.add_argument("--min-votes", type=int, default=DEFAULT_MIN_VOTES, help="Minimum votes threshold.")
    parser.add_argument("--clean-raw", action="store_true", help="Delete downloaded TSVs after building.")
    args = parser.parse_args()
    build_catalog(
        tmp_dir=args.tmp_dir,
        output_path=args.output,
        min_votes=args.min_votes,
        clean_raw=args.clean_raw,
    )
