"""Build MovieMind World Cinema catalog from IMDb non-commercial datasets.

Downloads title.basics, title.ratings, and title.akas from datasets.imdbws.com.
Filters: titleType=movie, numVotes>=2000, valid startYear.
Output: artifacts/global_catalog.parquet (<50 MB).
"""

import gzip
import os
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils import clean_title

IMDB_BASE_URL = "https://datasets.imdbws.com"
FILES = ["title.ratings.tsv.gz", "title.basics.tsv.gz", "title.akas.tsv.gz"]

def download_file(filename: str, dest_dir: Path) -> Path:
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[imdb] Found existing {filename} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    url = f"{IMDB_BASE_URL}/{filename}"
    print(f"[imdb] Downloading {url} -> {dest} ...")
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
                print(f"  .. downloaded {downloaded / 1e6:.1f} MB")
    print(f"[imdb] Downloaded {filename} ({downloaded / 1e6:.1f} MB)")
    return dest

def build_catalog(
    tmp_dir: Path = Path("/tmp"),
    output_path: Path = Path("artifacts/global_catalog.parquet"),
    min_votes: int = 2000,
) -> Path:
    # 1. Download source files
    ratings_path = download_file("title.ratings.tsv.gz", tmp_dir)
    basics_path = download_file("title.basics.tsv.gz", tmp_dir)
    akas_path = download_file("title.akas.tsv.gz", tmp_dir)

    # 2. Filter ratings (numVotes >= min_votes)
    print(f"[catalog] Filtering ratings (numVotes >= {min_votes}) ...")
    ratings = {}
    with gzip.open(ratings_path, "rt", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                votes = int(parts[2]) if parts[2].isdigit() else 0
                if votes >= min_votes:
                    ratings[parts[0]] = (float(parts[1]), votes)
    print(f"[catalog] Kept {len(ratings):,} titles with >= {min_votes} votes.")

    # 3. Filter basics (titleType=movie, valid startYear)
    print("[catalog] Filtering title.basics ...")
    movies = {}
    with gzip.open(basics_path, "rt", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 9:
                tconst, ttype, primary_title, orig_title, _, start_year, _, _, genres = parts[:9]
                if tconst in ratings and ttype == "movie" and start_year != r"\N" and start_year.isdigit():
                    norm_genres = "" if genres == r"\N" else genres.replace(",", "|")
                    movies[tconst] = {
                        "title": clean_title(primary_title),
                        "orig_title": orig_title if orig_title != r"\N" else primary_title,
                        "year": int(start_year),
                        "genres": norm_genres,
                        "rating": ratings[tconst][0],
                        "votes": ratings[tconst][1],
                    }
    movie_ids = set(movies.keys())
    print(f"[catalog] Kept {len(movies):,} feature films.")

    # 4. Stream akas for languages and approximate origin region
    print("[catalog] Parsing title.akas for languages and origin region ...")
    origin_regions = {}
    orig_title_strs = {}
    movie_languages = defaultdict(set)
    candidate_regions = defaultdict(list)

    with gzip.open(akas_path, "rt", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 8:
                mid = parts[0]
                if mid in movie_ids:
                    title_str = parts[2].strip().lower()
                    region = parts[3]
                    language = parts[4]
                    attribs = parts[6].lower()
                    is_orig = parts[7]

                    if language != r"\N" and language:
                        movie_languages[mid].add(language)

                    if is_orig == "1":
                        orig_title_strs[mid] = title_str
                        if region != r"\N" and region:
                            origin_regions[mid] = region

                    if region != r"\N" and region:
                        candidate_regions[mid].append((region, title_str, attribs))

    # Resolve origin region for movies where isOriginalTitle had no direct region
    for mid, mdata in movies.items():
        if mid not in origin_regions:
            orig_t = orig_title_strs.get(mid) or mdata["orig_title"].strip().lower()
            # 1. Look for matching regional release title
            found = False
            for reg, t_str, att in candidate_regions[mid]:
                if t_str == orig_t:
                    origin_regions[mid] = reg
                    found = True
                    break
            # 2. Look for premiere attribute
            if not found:
                for reg, _, att in candidate_regions[mid]:
                    if "première" in att or "premiere" in att or "original" in att:
                        origin_regions[mid] = reg
                        found = True
                        break
            # 3. Fallback to most frequent candidate region
            if not found and candidate_regions[mid]:
                regs = [r[0] for r in candidate_regions[mid]]
                origin_regions[mid] = Counter(regs).most_common(1)[0][0]

    # 5. Build final catalog records
    records = []
    for mid, m in movies.items():
        langs = ",".join(sorted(movie_languages.get(mid, [])))
        reg = origin_regions.get(mid, "Unknown")
        records.append({
            "id": mid,
            "title": m["title"],
            "year": m["year"],
            "genres": m["genres"],
            "rating": m["rating"],
            "votes": m["votes"],
            "languages": langs,
            "approx_origin_region": reg,
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[catalog] Successfully saved {len(df):,} movies to {output_path} ({size_mb:.2f} MB)")
    print(f"[catalog] Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"[catalog] Top 5 regions: {df['approx_origin_region'].value_counts().head(5).to_dict()}")
    return output_path

if __name__ == "__main__":
    build_catalog()
