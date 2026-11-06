import os
from pathlib import Path
import pandas as pd
import pytest

CATALOG_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "global_catalog.parquet"

def test_global_catalog_schema_and_size():
    assert CATALOG_PATH.exists(), f"Missing {CATALOG_PATH}"
    size_mb = CATALOG_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < 50.0, f"Catalog file size {size_mb:.2f} MB exceeds 50 MB threshold"

    df = pd.read_parquet(CATALOG_PATH)
    assert len(df) > 10000, f"Expected >10,000 global movies, got {len(df)}"

    expected_cols = ["id", "title", "year", "genres", "rating", "votes", "languages", "approx_origin_region"]
    assert list(df.columns) == expected_cols

    assert (df["votes"] >= 2000).all(), "All catalog movies must have >= 2000 votes"
    assert (df["year"] > 1880).all(), "All catalog movies must have a valid year"
    assert df["title"].str.strip().ne("").all(), "Titles must not be empty"

def test_global_content_recommender_cold_start():
    from src.global_recommender import GlobalContentRecommender
    rec = GlobalContentRecommender(CATALOG_PATH)
    recs = rec.recommend(n=10)
    assert len(recs) == 10
    for r in recs:
        assert "movie_id" in r
        assert "title" in r
        assert "approx_origin_region" in r
        assert "rating" in r
        assert "match_pct" in r

def test_global_recommender_likes_shift_profile():
    from src.global_recommender import GlobalContentRecommender
    rec = GlobalContentRecommender(CATALOG_PATH)

    # Pick a movie: e.g. Seven Samurai 'tt0047478' (JP, Action/Drama)
    seven_samurai_id = "tt0047478"
    recs_unconditioned = rec.recommend(n=10)
    recs_liked = rec.recommend(liked_ids=[seven_samurai_id], n=10)

    # The liked item should not be in recommendations
    assert seven_samurai_id not in [r["movie_id"] for r in recs_liked]

    # The liked profile should elevate Japanese or Action/Drama titles
    regions_liked = [r["approx_origin_region"] for r in recs_liked]
    assert "JP" in regions_liked or any("Action" in r["genres"] or "Drama" in r["genres"] for r in recs_liked)

def test_global_recommender_filtering():
    from src.global_recommender import GlobalContentRecommender
    rec = GlobalContentRecommender(CATALOG_PATH)

    # Filter by region: FR
    fr_recs = rec.recommend(region="FR", n=5)
    assert len(fr_recs) == 5
    for r in fr_recs:
        assert r["approx_origin_region"] == "FR"

    # Filter by year range: 1980 - 1989
    eighties_recs = rec.recommend(year_min=1980, year_max=1989, n=5)
    assert len(eighties_recs) == 5
    for r in eighties_recs:
        assert 1980 <= r["year"] <= 1989
