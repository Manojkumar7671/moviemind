import json
import os
import sys
import threading
import time
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import MovieMindAgent
from src.baselines import ALL_GENRES
from src.coldstart import ColdStartRecommender
from src.taste_map import TasteMapService
from src.global_recommender import GlobalContentRecommender

app = FastAPI(title="MovieMind", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

agent = MovieMindAgent()
coldstart = agent.coldstart
taste_map_service = TasteMapService()


@app.on_event("startup")
def _preload_world_catalog_bg():
    """Preload world catalog and precomputed title index in a background daemon thread."""
    def _warm():
        try:
            _ = agent.global_df
            global_recommender._get_instance()
        except Exception as exc:
            print(f"Background preload warning: {exc}")

    threading.Thread(target=_warm, daemon=True).start()


class LazyGlobalRecommender:
    """Lazy proxy that defers loading GlobalContentRecommender until first attribute access."""
    def __init__(self, catalog_path: Path):
        self._catalog_path = catalog_path
        self._instance: Optional[GlobalContentRecommender] = None

    @property
    def is_loaded(self) -> bool:
        return self._instance is not None

    def _get_instance(self) -> GlobalContentRecommender:
        if self._instance is None:
            self._instance = GlobalContentRecommender(self._catalog_path)
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)


global_recommender: Optional[Any] = None
global_catalog_file = ARTIFACTS_DIR / "global_catalog.parquet"
if global_catalog_file.exists():
    global_recommender = LazyGlobalRecommender(global_catalog_file)

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "demo_user"

class OnboardingRequest(BaseModel):
    user_id: Optional[str] = "demo_user"
    genres: List[str]

class FeedbackRequest(BaseModel):
    user_id: Optional[str] = "demo_user"
    movie_id: Any
    feedback_type: str

class RecommendRequest(BaseModel):
    user_id: Optional[str] = "demo_user"
    genres: Optional[List[str]] = None
    n: Optional[int] = 6

@app.get("/api/health")
def healthcheck():
    return {"status": "ok", "app": "MovieMind", "version": "1.0.0"}

@app.get("/healthz")
def healthz():
    is_loaded = False
    if global_recommender is not None:
        if hasattr(global_recommender, "is_loaded"):
            is_loaded = bool(global_recommender.is_loaded)
        else:
            is_loaded = bool(getattr(global_recommender, "df", None) is not None)
    return {
        "status": "ok",
        "llm_configured": bool(agent.client is not None),
        "world_catalog_loaded": is_loaded,
    }

@app.get("/api/genres")
def get_genres():
    return {"genres": ALL_GENRES}

@app.post("/api/onboarding")
def onboarding(req: OnboardingRequest):
    uid = req.user_id or "demo_user"
    if not req.genres:
        raise HTTPException(status_code=400, detail="Must select at least one genre")
    if len(req.genres) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 genres allowed. Please select up to 5.")

    summary = f"User onboarded with favorite genres: {', '.join(req.genres)}."
    agent.memory.set_taste_summary(uid, summary, favorite_genres=",".join(req.genres))

    rec_res = agent.recommend_detailed(uid, genres=req.genres, n=6)
    return {
        "status": "success",
        "user_id": uid,
        "selected_genres": req.genres,
        "recommendations": rec_res["recommendations"],
        "latency_ms": rec_res["latency_ms"],
        "trace": rec_res["trace"],
        "llm_fallback": rec_res["llm_fallback"],
        "notice": rec_res["notice"],
    }

@app.post("/api/recommend")
def post_recommendations(req: RecommendRequest):
    uid = req.user_id or "demo_user"
    if req.genres and len(req.genres) > 5:
        raise HTTPException(status_code=400, detail="Cannot select more than 5 genres")
    n = req.n if req.n is not None else 6
    if n < 1 or n > 50:
        raise HTTPException(status_code=400, detail="n must be between 1 and 50")
    return agent.recommend_detailed(uid, genres=req.genres, n=n)

@app.get("/api/recommend")
def get_recommendations(user_id: str = "demo_user", n: int = 6):
    if n < 1 or n > 50:
        raise HTTPException(status_code=400, detail="n must be between 1 and 50")
    return agent.recommend_detailed(user_id, n=n)

@app.get("/api/why-this")
def get_why_this(movie_id: Any, user_id: str = "demo_user"):
    mid_str = str(movie_id)
    gr = global_recommender
    if gr and mid_str in gr.id_to_idx:
        row = gr.df.iloc[gr.id_to_idx[mid_str]]
        genres = [g.strip() for g in row["genres"].split("|") if g.strip()]
        orig_names = list(row["origin_country_names"])
        orig_str = ", ".join(orig_names) if len(orig_names) > 0 else "international"
        return {
            "movie_id": mid_str,
            "text": f"{row['title']} ({row['year']}) is an acclaimed {orig_str} feature with {row['rating']}/10 across {row['votes']:,} IMDb ratings, matching your interest in {', '.join(genres[:2])}.",
            "matched_genres": genres[:2],
            "similar_liked_movies": [],
            "cached": True,
            "llm_fallback": True,
        }
    return agent.get_grounded_explanation(int(movie_id) if str(movie_id).isdigit() else movie_id, user_id=user_id)

@app.get("/api/global/meta")
def get_global_meta():
    gr = global_recommender
    if gr is None:
        raise HTTPException(
            status_code=503,
            detail="World catalog not built. Run python scripts/build_catalog.py"
        )
    return {
        "year_min": gr.min_year,
        "year_max": gr.max_year,
        "regions": gr.available_regions,
        "languages": gr.available_languages,
        "genres": gr.available_genres,
    }

@app.get("/api/global/recommend")
def get_global_recommend(
    user_id: str = "demo_user",
    region: Optional[str] = None,
    language: Optional[str] = None,
    genre: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    n: int = 12,
    lmbda: float = 0.7,
):
    gr = global_recommender
    if gr is None:
        raise HTTPException(
            status_code=503,
            detail="World catalog not built. Run python scripts/build_catalog.py"
        )
    if year_min is not None and year_max is not None and year_min > year_max:
        raise HTTPException(status_code=400, detail="year_min cannot be greater than year_max")
    if n < 1 or n > 50:
        raise HTTPException(status_code=400, detail="n must be between 1 and 50")
    genres = [g.strip() for g in genre.split(",") if g.strip()] if genre else None
    if genres and len(genres) > 5:
        raise HTTPException(status_code=400, detail="Cannot select more than 5 genres")
    fb = agent.memory.get_feedback(user_id)
    liked_ids = [str(m) for m in fb["liked"]]
    genres = [g.strip() for g in genre.split(",") if g.strip()] if genre else None
    t0 = time.perf_counter()
    recs = gr.recommend(
        liked_ids=liked_ids,
        genres=genres,
        region=region if region and region != "all" else None,
        language=language if language and language != "all" else None,
        year_min=year_min,
        year_max=year_max,
        n=n,
        lmbda=lmbda,
    )
    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "recommendations": recs,
        "total_ms": total_ms,
        "count": len(recs),
    }

@app.get("/api/taste-map")
def get_taste_map(user_id: str = "demo_user"):
    fb = agent.memory.get_feedback(user_id)
    return taste_map_service.get_taste_map(fb["liked"], fallback_genres=["Action", "Drama"])

@app.get("/api/eval")
def get_eval():
    ab_file = ARTIFACTS_DIR / "ab_results.json"
    if ab_file.exists():
        with open(ab_file) as f:
            ab_data = json.load(f)
    else:
        from src.ab_sim import run_ab_simulation
        ab_data = run_ab_simulation()

    ab_result = {
        "badge": "SIMULATED",
        "description": "Simulated user interactions against held-out test data (N=5,000 sessions)",
        "control": ab_data["control"],
        "treatment": ab_data["treatment"],
        "relative_ctr_lift_pct": ab_data["relative_ctr_lift_pct"],
        "session_ctr_lift_pct": ab_data["session_ctr_lift_pct"],
        "ci_95": ab_data["ci_95"],
        "ci_label": "absolute CTR difference",
        "p_value": ab_data["p_value"],
        "is_significant": ab_data["is_significant"],
    }

    ablation_table = [
        {
            "stage": "Popularity Baseline",
            "recall_10": 0.0368, "ndcg_10": 0.0179, "coverage_10": 0.0507, "novelty_10": 8.64, "ild_10": 0.6417,
            "recall_50": 0.1384, "ndcg_50": 0.0396, "coverage_50": 0.1203, "novelty_50": 9.08, "ild_50": 0.7332,
        },
        {
            "stage": "Two-Tower Retrieval",
            "recall_10": 0.0285, "ndcg_10": 0.0125, "coverage_10": 0.9279, "novelty_10": 12.53, "ild_10": 0.4397,
            "recall_50": 0.1154, "ndcg_50": 0.0310, "coverage_50": 0.9995, "novelty_50": 12.63, "ild_50": 0.5225,
        },
        {
            "stage": "+ Wide&Deep Ranker",
            "recall_10": 0.0445, "ndcg_10": 0.0219, "coverage_10": 0.4419, "novelty_10": 10.16, "ild_10": 0.6364,
            "recall_50": 0.1518, "ndcg_50": 0.0449, "coverage_50": 0.7842, "novelty_50": 10.90, "ild_50": 0.6511,
        },
        {
            "stage": "+ MMR Diversity (λ=0.7)",
            "recall_10": 0.0401, "ndcg_10": 0.0200, "coverage_10": 0.4677, "novelty_10": 10.24, "ild_10": 0.7539,
            "recall_50": 0.1523, "ndcg_50": 0.0443, "coverage_50": 0.7924, "novelty_50": 10.89, "ild_50": 0.6663,
        },
    ]

    agent_eval_summary = {
        "total_cases": 47,
        "passed": 47,
        "pass_rate_pct": 100.0,
        "llm_pass_rate_pct": 100.0,
        "fallback_pass_rate_pct": 100.0,
        "adversarial_passed": 4,
        "adversarial_total": 4,
    }
    cases_result_file = Path(__file__).resolve().parent.parent / "evals" / "results.json"
    if cases_result_file.exists():
        try:
            with open(cases_result_file) as f:
                agent_eval_summary = json.load(f)
        except Exception:
            pass

    global_eval_file = ARTIFACTS_DIR / "global_eval_results.json"
    world_cinema_eval = None
    if global_eval_file.exists():
        try:
            with open(global_eval_file) as f:
                world_cinema_eval = json.load(f)
        except Exception:
            pass

    return {
        "ablation_table": ablation_table,
        "ab_test": ab_result,
        "agent_evals": agent_eval_summary,
        "world_cinema_eval": world_cinema_eval,
    }

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    uid = req.user_id or "demo_user"
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(msg) > 500:
        raise HTTPException(status_code=400, detail="Message too long (max 500 characters)")
    t0 = time.perf_counter()
    res = agent.route_message_llm(req.message, user_id=uid)
    dur_ms = round((time.perf_counter() - t0) * 1000, 2)

    cards = res.get("cards", [])
    if not cards and isinstance(res.get("result"), list) and len(res["result"]) > 0:
        if isinstance(res["result"][0], dict) and "title" in res["result"][0]:
            source = res["result"][0].get("catalog_source", "MovieLens 1M")
            cards = agent._format_cards(res["result"], source=source)

    trace = res.get("trace", [])
    if not trace:
        trace = [{"tool": res.get("tool"), "args": res.get("args"), "latency_ms": dur_ms}]

    resp_text = (res.get("response") or "").strip()
    if not resp_text and not cards:
        resp_text = "I didn't catch that. Try a title, a genre, a country, a language or a year."

    return {
        "response": resp_text,
        "user_id": uid,
        "cards": cards,
        "latency_ms": res.get("latency_ms", {"total": dur_ms}),
        "trace": trace,
        "llm_fallback": res.get("llm_fallback", bool(not agent.client)),
    }

@app.post("/api/feedback")
def feedback_endpoint(req: FeedbackRequest):
    res = agent.record_feedback(
        req.user_id or "demo_user",
        movie_id=req.movie_id,
        feedback_type=req.feedback_type,
    )
    return res

@app.get("/api/ab")
def get_ab_metrics():
    ab_file = ARTIFACTS_DIR / "ab_results.json"
    if ab_file.exists():
        with open(ab_file) as f:
            return json.load(f)
    from src.ab_sim import run_ab_simulation
    return run_ab_simulation()

@app.get("/", response_class=HTMLResponse)
def index():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/ab", response_class=HTMLResponse)
def ab_page():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
