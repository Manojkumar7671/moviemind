"""Build MovieMind World Cinema catalog from IMDb non-commercial datasets and Wikidata.

Filters: titleType=movie, numVotes>=2000, valid startYear.
Origin and language metadata sourced from Wikidata (CC0).
Output: artifacts/global_catalog.parquet (<50 MB).
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set

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


def download_file(filename: str, dest_dir: Path) -> Path:
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 1000:
        logger.info(f"Found existing {filename} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    url = f"{IMDB_BASE_URL}/{filename}"
    logger.info(f"Downloading {url} -> {dest} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "MovieMind/1.0 (Academic/Non-Commercial)"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
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
    logger.info(f"Downloaded {filename} ({downloaded / 1e6:.1f} MB)")
    return dest


def build_catalog(
    tmp_dir: Path = Path("/tmp"),
    output_path: Path = DEFAULT_OUTPUT_PATH,
    min_votes: int = 2000,
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

    # 5. Build final catalog records with honest Unknowns
    records = []
    for mid, m in movies.items():
        wdata = wikidata_data.get(mid, {})
        countries = list(wdata.get("countries", []))
        languages = list(wdata.get("languages", []))
        country_labels = list(wdata.get("country_labels", []))
        language_labels = list(wdata.get("language_labels", []))

        origin_primary = countries[0] if countries else "Unknown"
        language_primary = languages[0] if languages else "Unknown"

        records.append({
            "id": mid,
            "title": m["title"],
            "year": m["year"],
            "genres": m["genres"],
            "rating": m["rating"],
            "votes": m["votes"],
            "origin_countries": countries,
            "original_languages": languages,
            "origin_primary": origin_primary,
            "language_primary": language_primary,
            "country_labels": country_labels,
            "language_labels": language_labels,
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[catalog] Successfully saved {len(df):,} movies to {output_path} ({size_mb:.2f} MB)")
    print(f"[catalog] Year range: {df['year'].min()} - {df['year'].max()}")

    # Coverage statistics
    total = len(df)
    n_countries = (df["origin_countries"].apply(len) > 0).sum()
    n_languages = (df["original_languages"].apply(len) > 0).sum()
    n_unknown_origin = (df["origin_primary"] == "Unknown").sum()
    n_unknown_lang = (df["language_primary"] == "Unknown").sum()

    print("\n=== Catalog Coverage Report ===")
    print(f"Total films:               {total:,}")
    print(f"origin_countries >= 1:     {n_countries:,} ({n_countries / total * 100:.2f}%)")
    print(f"original_languages >= 1:   {n_languages:,} ({n_languages / total * 100:.2f}%)")
    print(f"origin_primary Unknown:    {n_unknown_origin:,} ({n_unknown_origin / total * 100:.2f}%)")
    print(f"language_primary Unknown:  {n_unknown_lang:,} ({n_unknown_lang / total * 100:.2f}%)")
    print("\nTop 10 origin_primary countries:")
    print(df["origin_primary"].value_counts().head(10).to_string())
    print("===============================\n")

    return output_path


if __name__ == "__main__":
    build_catalog()
