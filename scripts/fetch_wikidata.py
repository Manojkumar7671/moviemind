#!/usr/bin/env python3
"""Fetch country of origin and original language from Wikidata for global catalog films.

Polite batch SPARQL queries with retry logic, exponential backoff, and resumable cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_wikidata")

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "MovieMind/1.0 (portfolio project; github.com/Manojkumar7671)"
BATCH_SIZE = 150
SLEEP_BETWEEN_BATCHES_SEC = 1.5
TIMEOUT_SEC = 60
MAX_RETRIES = 5

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CACHE_PATH = ARTIFACTS_DIR / "wikidata_cache.json"
CATALOG_PATH = ARTIFACTS_DIR / "global_catalog.parquet"

KNOWN_TEST_IDS = [
    "tt0068646",  # The Godfather (1972)
    "tt0120338",  # Titanic (1997)
    "tt8178634",  # RRR (2022)
    "tt6751668",  # Parasite (2019)
    "tt0245429",  # Spirited Away (2001)
    "tt0317248",  # City of God (2002)
    "tt0047478",  # Seven Samurai (1954)
    "tt0110912",  # Pulp Fiction (1994)
    "tt0137523",  # Fight Club (1999)
    "tt0109830",  # Forrest Gump (1994)
    "tt0076759",  # Star Wars (1977)
    "tt0111161",  # The Shawshank Redemption (1994)
    "tt0108052",  # Schindler's List (1993)
    "tt0050083",  # 12 Angry Men (1957)
    "tt0468569",  # The Dark Knight (2008)
    "tt0167260",  # LOTR: Return of the King (2003)
    "tt0060196",  # The Good, the Bad and the Ugly (1966)
    "tt0114369",  # Se7en (1995)
    "tt0032553",  # The Great Dictator (1940)
    "tt0118715",  # The Big Lebowski (1998)
]


def load_cache() -> Dict[str, Dict[str, Any]]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read cache at {CACHE_PATH}: {e}. Starting fresh.")
    return {}


def save_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CACHE_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    temp_path.replace(CACHE_PATH)


def build_sparql_query(imdb_ids: List[str]) -> str:
    values_clause = " ".join(f'"{i}"' for i in imdb_ids)
    return f"""SELECT ?imdb ?countryCode ?countryLabel ?langCode ?langCode3 ?langLabel WHERE {{
  VALUES ?imdb {{ {values_clause} }}
  ?film wdt:P345 ?imdb .
  OPTIONAL {{ ?film wdt:P495 ?country . OPTIONAL {{ ?country wdt:P297 ?countryCode }} }}
  OPTIONAL {{ ?film wdt:P364 ?lang . OPTIONAL {{ ?lang wdt:P218 ?langCode }} OPTIONAL {{ ?lang wdt:P220 ?langCode3 }} }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


def execute_sparql_batch(imdb_ids: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Execute SPARQL query for a batch with polite retry and backoff."""
    query = build_sparql_query(imdb_ids)
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
    }
    encoded_data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    req = urllib.request.Request(WIKIDATA_SPARQL_URL, data=encoded_data, headers=headers)

    results_by_id: Dict[str, Dict[str, List[str]]] = {
        i: {"countries": [], "languages": [], "country_labels": [], "language_labels": []}
        for i in imdb_ids
    }

    response_json = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                response_json = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            retry_after = e.headers.get("Retry-After")
            wait_time = float(retry_after) if retry_after else (2 ** attempt) * 2.0
            logger.warning(f"HTTPError {e.code} on batch attempt {attempt}/{MAX_RETRIES}. Sleeping {wait_time:.1f}s.")
            if attempt == MAX_RETRIES:
                logger.error(f"Batch failed after {MAX_RETRIES} attempts.")
                return results_by_id
            time.sleep(wait_time)
        except Exception as e:
            wait_time = (2 ** attempt) * 1.5
            logger.warning(f"Network error {e} on attempt {attempt}/{MAX_RETRIES}. Sleeping {wait_time:.1f}s.")
            if attempt == MAX_RETRIES:
                logger.error(f"Batch failed after {MAX_RETRIES} attempts: {e}")
                return results_by_id
            time.sleep(wait_time)

    if not response_json:
        return results_by_id

    for b in response_json.get("results", {}).get("bindings", []):
        imdb = b.get("imdb", {}).get("value")
        if not imdb or imdb not in results_by_id:
            continue
        c_code = b.get("countryCode", {}).get("value")
        c_label = b.get("countryLabel", {}).get("value")
        country = c_code or c_label

        l_code = b.get("langCode", {}).get("value") or b.get("langCode3", {}).get("value")
        l_label = b.get("langLabel", {}).get("value")
        lang = l_code or l_label

        if country and country not in results_by_id[imdb]["countries"]:
            results_by_id[imdb]["countries"].append(country)
            if c_label and c_label not in results_by_id[imdb]["country_labels"]:
                results_by_id[imdb]["country_labels"].append(c_label)

        if lang and lang not in results_by_id[imdb]["languages"]:
            results_by_id[imdb]["languages"].append(lang)
            if l_label and l_label not in results_by_id[imdb]["language_labels"]:
                results_by_id[imdb]["language_labels"].append(l_label)

    return results_by_id


def run_test_20() -> None:
    logger.info("Testing Wikidata fetcher on 20 known IDs...")
    results = execute_sparql_batch(KNOWN_TEST_IDS)
    print("\n--- 20 Known IDs Test Results ---")
    for imdb in KNOWN_TEST_IDS:
        c = results[imdb]["countries"]
        l = results[imdb]["languages"]
        c_lbl = results[imdb]["country_labels"]
        l_lbl = results[imdb]["language_labels"]
        print(f"{imdb}: countries={c} ({c_lbl}), languages={l} ({l_lbl})")
    print("---------------------------------\n")


def fetch_all(force: bool = False) -> None:
    cache = {} if force else load_cache()
    cached_ids: Set[str] = set(cache.keys())
    logger.info(f"Loaded {len(cached_ids)} cached films.")

    if not CATALOG_PATH.exists():
        logger.error(f"Catalog not found at {CATALOG_PATH}")
        sys.exit(1)

    df = pd.read_parquet(CATALOG_PATH)
    all_ids: List[str] = df["id"].tolist()
    total_films = len(all_ids)
    logger.info(f"Catalog contains {total_films} films.")

    ids_to_fetch = [i for i in all_ids if i not in cached_ids]
    logger.info(f"{len(ids_to_fetch)} films remain to fetch ({len(cached_ids)} already cached).")

    total_batches = (len(ids_to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE if ids_to_fetch else 0
    failures = 0

    for batch_idx in range(total_batches):
        batch_ids = ids_to_fetch[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        batch_results = execute_sparql_batch(batch_ids)

        # Check if the entire batch came back empty
        batch_found = any(len(v["countries"]) > 0 or len(v["languages"]) > 0 for v in batch_results.values())
        if not batch_found:
            failures += 1

        for imdb, data in batch_results.items():
            cache[imdb] = data

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            save_cache(cache)
            done = len(cache)
            logger.info(
                f"Batch {batch_idx + 1}/{total_batches} done. "
                f"Total cached: {done}/{total_films} "
                f"({done / total_films * 100:.1f}%). "
                f"Empty batches: {failures}."
            )

        time.sleep(SLEEP_BETWEEN_BATCHES_SEC)

    save_cache(cache)
    cache_size_mb = CACHE_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Cache successfully saved to {CACHE_PATH} ({cache_size_mb:.2f} MB).")

    # Coverage statistics over the full catalog
    n_with_country = sum(1 for i in all_ids if len(cache.get(i, {}).get("countries", [])) > 0)
    n_with_lang = sum(1 for i in all_ids if len(cache.get(i, {}).get("languages", [])) > 0)
    n_with_both = sum(
        1
        for i in all_ids
        if len(cache.get(i, {}).get("countries", [])) > 0 and len(cache.get(i, {}).get("languages", [])) > 0
    )

    print("\n=== Final Wikidata Coverage Report ===")
    print(f"Total catalog films: {total_films}")
    print(f"Films with >= 1 country:  {n_with_country} ({n_with_country / total_films * 100:.2f}%)")
    print(f"Films with >= 1 language: {n_with_lang} ({n_with_lang / total_films * 100:.2f}%)")
    print(f"Films with country & lang:{n_with_both} ({n_with_both / total_films * 100:.2f}%)")
    print(f"Cache size: {cache_size_mb:.2f} MB")
    print("======================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch origin metadata from Wikidata.")
    parser.add_argument("--test-only", action="store_true", help="Only run test on 20 known IDs")
    parser.add_argument("--force", action="store_true", help="Ignore existing cache and refetch all")
    args = parser.parse_args()

    run_test_20()

    if not args.test_only:
        fetch_all(force=args.force)


if __name__ == "__main__":
    main()
