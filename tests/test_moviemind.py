import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.baselines import ALL_GENRES, evaluate_recs
from src.coldstart import ColdStartRecommender
from src.diversity import mmr_rerank

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# 1. Test: Split has no leakage
def test_split_has_no_leakage():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    assert len(val) == 6040, f"Expected 6,040 val interactions, got {len(val)}"
    assert len(test) == 6040, f"Expected 6,040 test interactions, got {len(test)}"
    assert len(train) + len(val) + len(test) == 1000209

    # Max train timestamp < val timestamp < test timestamp
    max_train_ts = train.groupby("user_id")["timestamp"].max().to_dict()
    val_ts = dict(zip(val["user_id"], val["timestamp"]))
    test_ts = dict(zip(test["user_id"], test["timestamp"]))

    for uid in test_ts:
        t_train = max_train_ts[uid]
        t_val = val_ts[uid]
        t_test = test_ts[uid]
        assert t_train <= t_val, f"Leakage: user {uid} train timestamp {t_train} > val {t_val}"
        assert t_val <= t_test, f"Leakage: user {uid} val timestamp {t_val} > test {t_test}"

# 2. Test: Metrics calculation is exact
def test_metrics_correctness():
    # Synthetic recommendations and targets
    recs = {
        1: [10, 20, 30, 40, 50],
        2: [99, 88, 77, 66, 55],
    }
    targets = {
        1: 20,  # Rank 2 (1-based)
        2: 999, # Not in top-5
    }
    train_counts = {10: 100, 20: 50, 30: 25, 40: 10, 50: 5, 99: 10, 88: 10, 77: 10, 66: 10, 55: 10}
    # Mock genre vectors
    g_vecs = {m: np.zeros(len(ALL_GENRES)) for m in [10, 20, 30, 40, 50, 99, 88, 77, 66, 55]}
    g_vecs[10][0] = 1.0  # Action
    g_vecs[20][1] = 1.0  # Adventure
    g_vecs[30][2] = 1.0  # Animation
    g_vecs[40][3] = 1.0
    g_vecs[50][4] = 1.0

    m = evaluate_recs(recs, targets, train_counts, total_catalog_size=100, genre_vectors=g_vecs, ks=(5,))

    # Recall@5: user 1 has 1.0, user 2 has 0.0 -> mean 0.5
    assert m["Recall@5"] == 0.5
    # NDCG@5: user 1 has 1 / log2(2 + 1) = 1 / 1.5849625 ~ 0.63093, user 2 has 0 -> mean ~ 0.31546
    expected_ndcg = 0.5 * (1.0 / math.log2(3))
    assert abs(m["NDCG@5"] - expected_ndcg) < 1e-4
    # Coverage@5: 10 unique items out of 100 catalog -> 0.10
    assert m["Coverage@5"] == 0.10

# 3. Test: MMR diversity >= no-MMR
def test_mmr_diversity_greater_or_equal():
    movies_df = pd.read_parquet(DATA_DIR / "movies.parquet")
    from src.baselines import get_genre_matrix
    genre_vectors = get_genre_matrix(movies_df)

    # Let item_ids be 10 movies, some with identical genres
    item_ids = list(movies_df["movie_id"].head(20).values)
    scores = [1.0 - i * 0.03 for i in range(len(item_ids))]

    # Pure relevance (no MMR penalty)
    pure_recs = mmr_rerank(item_ids, scores, genre_vectors, top_k=10, lmbda=1.0)
    # Diversified MMR
    div_recs = mmr_rerank(item_ids, scores, genre_vectors, top_k=10, lmbda=0.3)

    # Compute ILD of both
    def calc_ild(item_list):
        vecs = np.array([genre_vectors.get(m, np.zeros(len(ALL_GENRES))) for m in item_list])
        sims = np.dot(vecs, vecs.T)
        n = len(item_list)
        return float(np.mean(1.0 - sims[np.triu_indices(n, k=1)]))

    ild_pure = calc_ild(pure_recs)
    ild_div = calc_ild(div_recs)
    assert ild_div >= ild_pure - 1e-6, f"Expected ILD(div) >= ILD(pure), got {ild_div} vs {ild_pure}"

# 4. Test: Cold-start returns exactly n items with proper schema
def test_coldstart_returns_n_items():
    cs = ColdStartRecommender(DATA_DIR, ARTIFACTS_DIR)
    for n in (3, 5, 10):
        recs = cs.recommend(["Action", "Sci-Fi", "Thriller"], top_n=n)
        assert len(recs) == n, f"Expected {n} recommendations, got {len(recs)}"
        for r in recs:
            assert "movie_id" in r and isinstance(r["movie_id"], int)
            assert "title" in r and isinstance(r["title"], str)
            assert "genres" in r and isinstance(r["genres"], (list, tuple))
            assert "year" in r and isinstance(r["year"], int)
            assert "score" in r and isinstance(r["score"], float)
            assert not r["title"].endswith(f"({r['year']})")

# 5. Test: API smoke test
def test_api_smoke_test():
    client = TestClient(app)

    # Health
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Genres
    r = client.get("/api/genres")
    assert r.status_code == 200
    assert len(r.json()["genres"]) == 18

    # Onboarding
    r = client.post("/api/onboarding", json={"user_id": "test_pytest", "genres": ["Action", "Comedy"]})
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 6

    # Chat
    r = client.post("/api/chat", json={"user_id": "test_pytest", "message": "Search for Star Wars"})
    assert r.status_code == 200
    assert "Star Wars" in r.json()["response"]

    # Feedback
    r = client.post("/api/feedback", json={"user_id": "test_pytest", "movie_id": 260, "feedback_type": "like"})
    assert r.status_code == 200
    assert r.json()["status"] in ("success", "noop")

    # AB
    r = client.get("/api/ab")
    assert r.status_code == 200
    assert "relative_ctr_lift_pct" in r.json()

    # Frontend routes
    r = client.get("/")
    assert r.status_code == 200
    assert "MovieMind" in r.text
    assert "onboardingSummaryBar" in r.text

    r = client.get("/ab")
    assert r.status_code == 200
    assert "A/B Test" in r.text or "A/B" in r.text

    # Static assets
    assert client.get("/static/style.css").status_code == 200
    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "Relative match" in js_resp.text
    assert "editGenres" in js_resp.text
    assert client.get("/static/fonts/Anton-Regular.ttf").status_code == 200

# 6. Test: explain() excludes target movie from its own similar-to-liked list
def test_explain_excludes_self():
    import uuid
    from src.agent import MovieMindAgent
    agent = MovieMindAgent()
    uid = f"test_self_{uuid.uuid4().hex[:8]}"

    # Like movie 2571 (The Matrix)
    agent.record_feedback(uid, 2571, "like")

    # Explain movie 2571 for this user
    res = agent.explain(2571, user_id=uid)
    assert res["movie_id"] == 2571
    # Target 2571 must NOT appear in its own similar_liked_movies
    sim_titles = [m["title"] for m in res["similar_liked_movies"]]
    assert "The Matrix" not in sim_titles
    assert "Matrix, The" not in sim_titles
    assert len(res["similar_liked_movies"]) == 0
    # Must cite genre match only
    assert "similar to" not in res["explanation"]
    assert "matches your interest in" in res["explanation"]

# 7. Test: Onboarding rejects >5 genres
def test_onboarding_genre_cap():
    client = TestClient(app)
    # 6 genres should fail
    r = client.post("/api/onboarding", json={
        "user_id": "test_cap_user",
        "genres": ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Horror"]
    })
    assert r.status_code == 400
    assert "5 genres" in r.json()["detail"]

    # 5 genres should succeed
    r2 = client.post("/api/onboarding", json={
        "user_id": "test_cap_user",
        "genres": ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller"]
    })
    assert r2.status_code == 200
    assert len(r2.json()["selected_genres"]) == 5

# 8. Test: Duplicate like returns already liked
def test_duplicate_like_returns_already_liked():
    import uuid
    from src.agent import MovieMindAgent
    agent = MovieMindAgent()
    uid = f"test_dup_{uuid.uuid4().hex[:8]}"

    # First like
    r1 = agent.record_feedback(uid, 1, "like")
    assert r1["status"] == "success"
    assert "Recorded like" in r1["message"]

    # Duplicate like
    r2 = agent.record_feedback(uid, 1, "like")
    assert r2["status"] == "noop"
    assert "already liked" in r2["message"]

# 9. Test: Phase 2 API recommendations schema & latency breakdown
def test_api_recommend_schema_and_latency():
    client = TestClient(app)
    r = client.post("/api/recommend", json={"user_id": "demo_user", "genres": ["Action", "Sci-Fi"], "n": 4})
    assert r.status_code == 200
    data = r.json()

    assert "recommendations" in data
    assert len(data["recommendations"]) == 4

    card = data["recommendations"][0]
    assert "movie_id" in card
    assert "title" in card and not card["title"].endswith(f"({card['year']})")
    assert "year" in card and isinstance(card["year"], int)
    assert "genres" in card and isinstance(card["genres"], list)
    assert "match_pct" in card and 50 <= card["match_pct"] <= 100
    assert "score_parts" in card
    for k in ("retrieval", "ranker", "diversity"):
        assert k in card["score_parts"]
    assert "explanation" in card
    assert "matched_genres" in card["explanation"]
    assert "similar_liked_movies" in card["explanation"]

    # Latencies (recommend makes zero LLM calls)
    assert "latency_ms" in data
    assert data["latency_ms"]["llm"] == 0.0
    for stage in ("retrieval", "ranker", "mmr", "llm", "total"):
        assert stage in data["latency_ms"]
        assert isinstance(data["latency_ms"][stage], (int, float))

    # Trace
    assert "trace" in data and len(data["trace"]) >= 4

# 10. Test: Why-this on-demand endpoint and caching
def test_why_this_endpoint_and_caching():
    client = TestClient(app)
    # /api/recommend has zero LLM calls
    rec_r = client.post("/api/recommend", json={"user_id": "test_why_user", "genres": ["Action"], "n": 3})
    assert rec_r.status_code == 200
    assert rec_r.json()["latency_ms"]["llm"] == 0.0

    # /api/why-this generates explanation
    r = client.get("/api/why-this?movie_id=2571&user_id=test_why_user")
    assert r.status_code == 200
    d1 = r.json()
    assert d1["movie_id"] == 2571
    assert "text" in d1 and len(d1["text"]) > 10

    # Second call returns cached result
    r2 = client.get("/api/why-this?movie_id=2571&user_id=test_why_user")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["text"] == d1["text"]

# 10. Test: Phase 2 Taste Map & Eval endpoints
def test_api_taste_map_and_eval():
    client = TestClient(app)

    # Taste map
    r_tm = client.get("/api/taste-map?user_id=demo_user")
    assert r_tm.status_code == 200
    tm_data = r_tm.json()
    assert "points" in tm_data and len(tm_data["points"]) > 0
    assert "user_point" in tm_data
    assert "x" in tm_data["user_point"] and "y" in tm_data["user_point"]

    # Eval endpoint
    r_ev = client.get("/api/eval")
    assert r_ev.status_code == 200
    ev_data = r_ev.json()
    assert "ablation_table" in ev_data
    assert len(ev_data["ablation_table"]) == 4
    assert "ab_test" in ev_data
    assert ev_data["ab_test"]["badge"] == "SIMULATED"
    assert "relative_ctr_lift_pct" in ev_data["ab_test"]

# 11. Test: match_pct derived from real ranker score (min-max across list), not rank
def test_match_pct_derived_from_ranker_score():
    client = TestClient(app)
    r = client.post("/api/recommend", json={"user_id": "demo_user", "genres": ["Action", "Sci-Fi"], "n": 6})
    assert r.status_code == 200
    cards = r.json()["recommendations"]
    assert len(cards) == 6

    rk_scores = [c["score_parts"]["ranker"] for c in cards]
    match_pcts = [c["match_pct"] for c in cards]

    # Highest ranker score maps to highest match_pct (98) and lowest to 75
    assert max(match_pcts) == 98
    assert min(match_pcts) == 75

    # Any two items with identical ranker score have identical match_pct regardless of rank position
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if abs(rk_scores[i] - rk_scores[j]) < 1e-4:
                assert match_pcts[i] == match_pcts[j]

# 12. Test: Normalize titles "X, The/A/An" -> "The X"
def test_title_normalization_articles():
    from src.utils import clean_title
    from src.agent import MovieMindAgent
    assert clean_title("Matrix, The (1999)") == "The Matrix"
    assert clean_title("Godfather, The (1972)") == "The Godfather"
    assert clean_title("Clockwork Orange, A (1971)") == "A Clockwork Orange"
    assert clean_title("Affair to Remember, An (1957)") == "An Affair to Remember"
    assert clean_title("Toy Story (1995)") == "Toy Story"

    agent = MovieMindAgent()
    results = agent.search_movies("Matrix", limit=3)
    assert any(r["title"] == "The Matrix" for r in results)
    assert not any(r["title"].endswith(", The") for r in results)

# 13. Test: Off-topic input vs Movie catalog messages
def test_off_topic_queries_return_movie_only_message():
    from src.agent import MovieMindAgent
    agent = MovieMindAgent()

    off_topic_queries = [
        "what is 2 + 2?",
        "calculate 50 * 12",
        "who was George Washington?",
        "how do I make pizza dough?",
        "write python code to reverse a list",
    ]
    for q in off_topic_queries:
        res = agent.route_message(q)
        assert res["tool"] is None
        assert res["response"] == "I only help with movies. Try a title, genre, or year."

    # Movie title queries that are not in catalog must keep the catalog message
    movie_miss = agent.route_message("search for Avatar The Way of Water 2022")
    assert movie_miss["tool"] == "search_movies"
    assert "not in the MovieLens 1M catalog" in movie_miss["response"]

# 14. Test: Healthz endpoint
def test_healthz_endpoint():
    client = TestClient(app)
    res = client.get("/healthz")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "llm_configured" in data
    assert "world_catalog_loaded" in data
    assert isinstance(data["llm_configured"], bool)
    assert isinstance(data["world_catalog_loaded"], bool)

# 15. Test: Zero LLM calls on /api/recommend and /api/global/recommend
def test_zero_llm_calls_on_recommend_endpoints(monkeypatch):
    import app.main as main_mod
    llm_call_count = 0

    class MockCompletions:
        def create(self, *args, **kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            raise AssertionError("LLM must not be called during recommendation requests!")

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    monkeypatch.setattr(main_mod.agent, "client", MockClient())

    client = TestClient(main_mod.app)

    # 1. POST /api/recommend
    res1 = client.post("/api/recommend", json={"user_id": "test_user", "genres": ["Action", "Sci-Fi"], "n": 6})
    assert res1.status_code == 200
    assert len(res1.json()["recommendations"]) == 6

    # 2. GET /api/recommend
    res2 = client.get("/api/recommend?user_id=test_user&n=6")
    assert res2.status_code == 200
    assert len(res2.json()["recommendations"]) == 6

    # 3. GET /api/global/recommend
    res3 = client.get("/api/global/recommend?user_id=test_user&n=6")
    assert res3.status_code == 200
    assert len(res3.json()["recommendations"]) == 6

    # Assert exactly 0 calls to LLM occurred
    assert llm_call_count == 0, f"Expected 0 LLM calls, but got {llm_call_count}"

# 16. Test: API input validation and 4xx status codes
def test_api_input_validation():
    client = TestClient(app)

    # 1. POST /api/recommend with > 5 genres
    res = client.post("/api/recommend", json={"user_id": "u1", "genres": ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Thriller"], "n": 6})
    assert res.status_code == 400
    assert "Cannot select more than 5 genres" in res.json()["detail"]

    # 2. POST /api/recommend with invalid n
    res = client.post("/api/recommend", json={"user_id": "u1", "n": 0})
    assert res.status_code == 400
    res = client.post("/api/recommend", json={"user_id": "u1", "n": 100})
    assert res.status_code == 400

    # 3. POST /api/chat with empty or oversize message
    res = client.post("/api/chat", json={"message": "   "})
    assert res.status_code == 400
    res = client.post("/api/chat", json={"message": "a" * 501})
    assert res.status_code == 400

    # 4. POST /api/onboarding with > 5 genres
    res = client.post("/api/onboarding", json={"genres": ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Thriller"]})
    assert res.status_code == 400

    # 5. GET /api/global/recommend with year_min > year_max
    res = client.get("/api/global/recommend?year_min=2020&year_max=1990")
    assert res.status_code == 400
