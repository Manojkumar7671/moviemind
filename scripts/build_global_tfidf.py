#!/usr/bin/env python3
"""scripts/build_global_tfidf.py

Precomputes:
1. artifacts/global_tfidf.npz (scipy.sparse float32 CSR matrix)
2. artifacts/global_vectorizer.pkl (fitted TfidfVectorizer)
3. artifacts/global_lookups.json (regions, languages, genres, country_primary_lang, min/max year)
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def build_global_artifacts():
    catalog_path = ARTIFACTS_DIR / "global_catalog.parquet"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing {catalog_path}")

    df = pd.read_parquet(catalog_path)
    print(f"Loaded global catalog: {len(df)} rows")

    # 1. Feature texts
    feature_texts = []
    for _, r in df.iterrows():
        genres = " ".join([f"g_{g.strip().lower()}" for g in r["genres"].split("|") if g.strip()])
        langs = " ".join([f"l_{str(l).strip().lower()}" for l in r["original_languages"] if str(l).strip()])
        countries = " ".join([f"c_{str(c).strip().lower()}" for c in r["origin_countries"] if str(c).strip()])
        tokens = f"{genres} {langs} {countries}".strip()
        feature_texts.append(tokens if tokens else "empty")

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(feature_texts).astype(np.float32)

    # Save TF-IDF matrix & vectorizer
    sp.save_npz(ARTIFACTS_DIR / "global_tfidf.npz", tfidf_matrix)
    with open(ARTIFACTS_DIR / "global_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved global_tfidf.npz ({tfidf_matrix.shape}) and global_vectorizer.pkl")

    # 2. Lookups
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    region_counts = Counter()
    region_names = {}
    region_name_to_code = {}
    code_to_region_name = {}

    for _, row in df.iterrows():
        cs = row["origin_countries"]
        clbls = row["origin_country_names"]
        for c, lbl in zip(cs, clbls):
            if c:
                code = str(c).upper()
                region_counts[code] += 1
                region_name_to_code[code.lower()] = code
                if lbl:
                    lbl_str = str(lbl).strip()
                    region_names[code] = lbl_str
                    region_name_to_code[lbl_str.lower()] = code
                    code_to_region_name.setdefault(code, lbl_str)

    available_regions = [
        {"code": code, "name": region_names.get(code, code), "count": count}
        for code, count in region_counts.most_common()
    ]

    lang_counts = Counter()
    lang_names = {}
    lang_name_to_code = {}
    code_to_lang_name = {}

    for _, row in df.iterrows():
        ls = row["original_languages"]
        llbls = row["original_language_names"]
        for l, lbl in zip(ls, llbls):
            if l:
                code = str(l).lower()
                lang_counts[code] += 1
                lang_name_to_code[code] = code
                if lbl:
                    lbl_str = str(lbl).strip()
                    lang_names[code] = lbl_str
                    lang_name_to_code[lbl_str.lower()] = code
                    code_to_lang_name.setdefault(code, lbl_str)

    available_languages = [
        {"code": code, "name": lang_names.get(code, code), "count": count}
        for code, count in lang_counts.most_common()
    ]

    genre_counts = Counter()
    for _, row in df.iterrows():
        for g in row["genres"].split("|"):
            if g.strip():
                genre_counts[g.strip()] += 1
    available_genres = [g for g, _ in genre_counts.most_common()]

    sole_counts = defaultdict(Counter)
    for _, r in df.iterrows():
        cs = [str(c).upper() for c in r.get("origin_countries", []) if c]
        ls = [str(l).lower() for l in r.get("original_languages", []) if l]
        if len(cs) == 1 and ls:
            sole_counts[cs[0]][ls[0]] += 1
    country_primary_lang = {
        c: lc.most_common(1)[0][0] for c, lc in sole_counts.items() if lc
    }

    lookups = {
        "min_year": min_year,
        "max_year": max_year,
        "available_regions": available_regions,
        "available_languages": available_languages,
        "available_genres": available_genres,
        "country_primary_lang": country_primary_lang,
        "region_name_to_code": region_name_to_code,
        "code_to_region_name": code_to_region_name,
        "lang_name_to_code": lang_name_to_code,
        "code_to_lang_name": code_to_lang_name,
    }

    with open(ARTIFACTS_DIR / "global_lookups.json", "w") as f:
        json.dump(lookups, f)
    print(f"Saved global_lookups.json")

    # Precompute normalized title index sorted by votes descending for fast startup & fuzzy lookup
    import re as _re
    df_by_votes = df.sort_values(by="votes", ascending=False)
    seen_titles = set()
    title_entries = []
    for _, r in df_by_votes.iterrows():
        gid = str(r["id"])
        ct = _re.sub(r"\s+\(\d{4}\)$", "", str(r["title"]).strip()).lower().strip()
        variants = [ct]
        if ", the" in ct:
            variants.append("the " + ct.replace(", the", "").strip())
        if ", a" in ct:
            variants.append("a " + ct.replace(", a", "").strip())
        for v in variants:
            if v and v not in seen_titles:
                seen_titles.add(v)
                title_entries.append([v, gid])
    with open(ARTIFACTS_DIR / "global_title_index.json", "w") as f:
        json.dump(title_entries, f)
    print(f"Saved global_title_index.json ({len(title_entries)} entries)")

    # Validate output files exist and are non-empty
    tfidf_file = ARTIFACTS_DIR / "global_tfidf.npz"
    vec_file = ARTIFACTS_DIR / "global_vectorizer.pkl"
    lookups_file = ARTIFACTS_DIR / "global_lookups.json"
    title_idx_file = ARTIFACTS_DIR / "global_title_index.json"
    for out_f in [tfidf_file, vec_file, lookups_file, title_idx_file]:
        if not out_f.exists() or out_f.stat().st_size == 0:
            raise RuntimeError(f"Derived artifact build failed: {out_f} is missing or empty.")
    print("All derived artifacts successfully verified.")


if __name__ == "__main__":
    build_global_artifacts()
