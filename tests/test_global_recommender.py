import os
from pathlib import Path
import json
import pandas as pd
import pytest

CATALOG_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "global_catalog.parquet"


def test_global_catalog_schema_and_size():
    assert CATALOG_PATH.exists(), f"Missing {CATALOG_PATH}"
    size_mb = CATALOG_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < 50.0, f"Catalog file size {size_mb:.2f} MB exceeds 50 MB threshold"

    df = pd.read_parquet(CATALOG_PATH)
    assert len(df) > 10000, f"Expected >10,000 global movies, got {len(df)}"

    expected_cols = [
        "id",
        "imdb_id",
        "title",
        "year",
        "genres",
        "rating",
        "votes",
        "origin_countries",
        "origin_country_names",
        "original_languages",
        "original_language_names",
    ]
    assert list(df.columns) == expected_cols

    # Verify no legacy approx columns or single primary country columns exist
    legacy_col_substrings = ["approx" + "_origin", "origin" + "_primary"]
    for bad_col in legacy_col_substrings:
        assert bad_col not in df.columns

    assert (df["votes"] >= 2000).all(), "All catalog movies must have >= 2000 votes"
    assert (df["year"] > 1880).all(), "All catalog movies must have a valid year"
    assert df["title"].str.strip().ne("").all(), "Titles must not be empty"


def test_global_catalog_known_film_origins():
    df = pd.read_parquet(CATALOG_PATH)

    # 1. The Godfather (1972): origin_countries contains both IT and US
    gf = df[(df["title"] == "The Godfather") & (df["year"] == 1972)].iloc[0]
    assert "IT" in gf["origin_countries"]
    assert "US" in gf["origin_countries"]
    assert "Italian" in gf["origin_country_names"] or "Italy" in gf["origin_country_names"]
    assert "United States" in gf["origin_country_names"]

    # 2. RRR (2022): origin_countries == ["IN"], original_languages contains "te"
    rrr = df[(df["title"] == "RRR") & (df["year"] == 2022)].iloc[0]
    assert list(rrr["origin_countries"]) == ["IN"]
    assert "te" in rrr["original_languages"]

    # 3. Parasite (2019): contains KR and ko
    parasite = df[(df["title"] == "Parasite") & (df["year"] == 2019)].iloc[0]
    assert "KR" in parasite["origin_countries"]
    assert "ko" in parasite["original_languages"]

    # 4. Titanic (1997): contains US
    titanic = df[(df["title"] == "Titanic") & (df["year"] == 1997)].iloc[0]
    assert "US" in titanic["origin_countries"]

    # 5. Spirited Away -> JP
    spirited = df[df["title"] == "Spirited Away"].iloc[0]
    assert "JP" in spirited["origin_countries"]

    # 6. City of God (2002) -> BR
    city = df[(df["title"] == "City of God") & (df["year"] == 2002)].iloc[0]
    assert "BR" in city["origin_countries"]


def test_global_catalog_coverage_thresholds():
    df = pd.read_parquet(CATALOG_PATH)
    total = len(df)
    country_coverage = (df["origin_countries"].apply(len) > 0).sum() / total
    lang_coverage = (df["original_languages"].apply(len) > 0).sum() / total

    # Rule: country coverage >= 96%, language coverage >= 92%
    assert country_coverage >= 0.96, f"Expected country coverage >= 96%, got {country_coverage * 100:.2f}%"
    assert lang_coverage >= 0.92, f"Expected language coverage >= 92%, got {lang_coverage * 100:.2f}%"


def test_global_content_recommender_cold_start():
    from src.global_recommender import GlobalContentRecommender
    rec = GlobalContentRecommender(CATALOG_PATH)
    recs = rec.recommend(n=10)
    assert len(recs) == 10
    for r in recs:
        assert "movie_id" in r
        assert "title" in r
        assert "origin" in r
        assert "origin_countries" in r
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
    countries_liked = [c for r in recs_liked for c in r["origin_countries"]]
    assert "JP" in countries_liked or any("Action" in r["genres"] or "Drama" in r["genres"] for r in recs_liked)


def test_global_recommender_filtering():
    from src.global_recommender import GlobalContentRecommender
    rec = GlobalContentRecommender(CATALOG_PATH)

    # Filter by region: JP returns only films whose origin_countries contains JP
    jp_recs = rec.recommend(region="JP", n=10)
    assert len(jp_recs) == 10
    for r in jp_recs:
        assert "JP" in r["origin_countries"]
        assert len(r["origin_countries"]) > 0

    # Filter by language: te returns only Telugu originals
    te_recs = rec.recommend(language="te", n=5)
    assert len(te_recs) == 5
    for r in te_recs:
        assert "te" in r["original_languages"]
        assert len(r["original_languages"]) > 0

    # The Godfather appears under region=US and region=IT
    us_recs = rec.recommend(region="US", n=50)
    it_recs = rec.recommend(region="IT", n=50)
    us_ids = [r["movie_id"] for r in us_recs]
    it_ids = [r["movie_id"] for r in it_recs]
    godfather_id = "tt0068646"
    assert godfather_id in us_ids
    assert godfather_id in it_ids

    # An empty-list film never appears under an active filter
    for r in jp_recs + te_recs:
        assert len(r["origin_countries"]) > 0 or len(r["original_languages"]) > 0


def test_global_api_endpoints():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    # 1. Meta endpoint returns regions and languages with code, name, and count
    res_meta = client.get("/api/global/meta")
    assert res_meta.status_code == 200
    meta = res_meta.json()
    assert "regions" in meta and len(meta["regions"]) > 10
    assert "languages" in meta and len(meta["languages"]) > 10
    assert "genres" in meta and len(meta["genres"]) > 5
    assert meta["year_min"] > 1880
    assert meta["year_max"] >= 2020

    # Verify counts and display names are included in meta
    first_reg = meta["regions"][0]
    assert "code" in first_reg and "name" in first_reg and "count" in first_reg
    assert first_reg["code"] == "US"
    assert first_reg["count"] > 10000

    # 2. Recommendations endpoint
    res_rec = client.get("/api/global/recommend?n=6&region=FR")
    assert res_rec.status_code == 200
    data = res_rec.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 6
    for r in data["recommendations"]:
        assert "FR" in r["origin_countries"]
        assert "match_pct" in r
        assert "explanation" in r

    # 3. Feedback and why-this with string IMDb ID
    fb_res = client.post("/api/feedback", json={
        "user_id": "test_user_global",
        "movie_id": "tt0047478",
        "feedback_type": "like"
    })
    assert fb_res.status_code == 200
    assert fb_res.json()["status"] in ("success", "noop")

    why_res = client.get("/api/why-this?movie_id=tt0047478&user_id=test_user_global")
    assert why_res.status_code == 200
    why_data = why_res.json()
    assert "1954" in why_data["text"] or "Japan" in why_data["text"] or "Seven Samurai" in why_data["text"]


def test_chat_filter_movies_with_region_and_language():
    from src.agent import MovieMindAgent
    agent = MovieMindAgent()

    # 1. Year range comes from data
    assert agent.data_min_year <= 1920
    assert agent.data_max_year >= 2020

    # 2. Filter by region (code and name) and genre
    jp_movies = agent.filter_movies(region="JP", genre="Drama", limit=5)
    assert len(jp_movies) > 0
    for m in jp_movies:
        assert "JP" in m["origin_countries"]
        assert any("Drama" in g for g in m["genres"])

    japan_movies = agent.filter_movies(region="Japan", genre="Drama", limit=5)
    assert len(japan_movies) > 0
    for m in japan_movies:
        assert "JP" in m["origin_countries"]

    # 3. Filter by language (code or name) and exact year
    fr_movies = agent.filter_movies(language="fr", year=1995, limit=5)
    assert len(fr_movies) > 0
    for m in fr_movies:
        assert m["year"] == 1995
        assert "fr" in [l.lower() for l in m["original_languages"]]

    te_movies = agent.filter_movies(language="Telugu", limit=5)
    assert len(te_movies) > 0
    for m in te_movies:
        assert "te" in [l.lower() for l in m["original_languages"]]

    # 4. Chat router routes regional queries to filter_movies
    chat_res = agent.route_message("filter Japanese movies from 1956")
    assert chat_res["tool"] == "filter_movies"
    assert chat_res["args"]["region"] == "JP"
    assert chat_res["args"]["year"] == 1956
    assert any("The Burmese Harp" in r["title"] or "Burmese Harp" in r["title"] for r in chat_res["result"])


def test_global_content_eval_metrics():
    eval_file = Path(__file__).resolve().parent.parent / "artifacts" / "global_eval_results.json"
    docs_file = Path(__file__).resolve().parent.parent / "docs" / "global_eval.md"

    if not eval_file.exists():
        from src.global_eval import run_global_evaluation
        run_global_evaluation(n_samples=50, seed=42)

    assert eval_file.exists(), "Missing artifacts/global_eval_results.json"
    assert docs_file.exists(), "Missing docs/global_eval.md"

    with open(eval_file) as f:
        data = json.load(f)

    # Verify honesty disclaimer and limitations are present
    assert "collaborative" in data["honesty_disclaimer"].lower()
    assert "sanity check" in data["limitations"].lower()

    # Verify protocol metrics
    assert data["n_samples"] == 500
    assert data["seed"] == 42
    assert data["recommender_hit20"] > 0.0, "Expected recommender Hit@20 > 0%"
    assert data["recommender_hit20"] >= data["random_hit20"]
    assert data["recommender_hit20"] >= data["popularity_hit20"]
    assert data["lift_over_random"] >= 0.0
    assert data["lift_over_popularity"] >= 0.0

    # Verify /api/eval includes world cinema eval
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    res = client.get("/api/eval")
    assert res.status_code == 200
    eval_json = res.json()
    assert "world_cinema_eval" in eval_json
    assert eval_json["world_cinema_eval"] is not None
    assert "honesty_disclaimer" in eval_json["world_cinema_eval"]


def test_global_catalog_missing_fallback(monkeypatch):
    import app.main as main_mod
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main_mod, "global_recommender", None)
    client = TestClient(main_mod.app)

    res_meta = client.get("/api/global/meta")
    assert res_meta.status_code == 503
    assert "World catalog not built" in res_meta.json()["detail"]

    res_rec = client.get("/api/global/recommend")
    assert res_rec.status_code == 503
    assert "World catalog not built" in res_rec.json()["detail"]
