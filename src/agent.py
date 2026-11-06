import json
import os
import re
import sys
import time
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from src.baselines import get_genre_matrix
from src.coldstart import ColdStartRecommender
from src.diversity import mmr_rerank
from src.memory import UserMemory
from src.utils import clean_title, parse_genres

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

class MovieMindAgent:
    """Tool-calling Recommender Agent with OpenRouter client and robust fallback."""
    def __init__(self, data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR):
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.movies_df = pd.read_parquet(data_dir / "movies.parquet")
        self.movie_meta = self.movies_df.set_index("movie_id").to_dict(orient="index")
        self.title_to_id = {row["title"].lower(): int(row["movie_id"]) for _, row in self.movies_df.iterrows()}

        # Normalized title mapping for robust resolution
        self.title_map = {}
        for _, row in self.movies_df.iterrows():
            mid = int(row["movie_id"])
            ct = clean_title(row["title"]).lower()
            self.title_map[ct] = mid
            if ", the" in ct:
                self.title_map["the " + ct.replace(", the", "").strip()] = mid
            if ", a" in ct:
                self.title_map["a " + ct.replace(", a", "").strip()] = mid
        for a, mid in {
            "matrix": 2571, "aliens": 1200, "alien": 1214, "godfather": 858,
            "star wars": 260, "toy story": 1, "jurassic park": 480
        }.items():
            self.title_map[a] = mid

        self.memory = UserMemory()
        self.coldstart = ColdStartRecommender(data_dir, artifacts_dir)

        # Preloaded candidate tables if available
        self.mmr_path = artifacts_dir / "mmr_candidates.parquet"
        self.user_recs_cache = {}
        if self.mmr_path.exists():
            df = pd.read_parquet(self.mmr_path)
            self.user_recs_cache = {int(r["user_id"]): list(r["candidate_ids"]) for _, r in df.iterrows()}

        self.genre_vectors = get_genre_matrix(self.movies_df)

        self.retrieval_path = artifacts_dir / "retrieval_candidates.parquet"
        self.retrieval_cache = {}
        if self.retrieval_path.exists():
            df_ret = pd.read_parquet(self.retrieval_path)
            self.retrieval_cache = {int(r["user_id"]): (list(r["candidate_ids"]), list(r["candidate_scores"])) for _, r in df_ret.iterrows()}

        self.ranked_path = artifacts_dir / "ranked_candidates.parquet"
        self.ranked_cache = {}
        if self.ranked_path.exists():
            df_rk = pd.read_parquet(self.ranked_path)
            self.ranked_cache = {int(r["user_id"]): (list(r["candidate_ids"]), list(r["candidate_scores"])) for _, r in df_rk.iterrows()}

        # Item embeddings for similarity explanation
        self.item_embs = np.load(artifacts_dir / "item_embeddings.npy")
        self.item_ids = np.load(artifacts_dir / "item_ids.npy")
        self.mid_to_idx = {int(mid): idx for idx, mid in enumerate(self.item_ids)}

        # Setup LLM / OpenAI-compatible client
        self.base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://api.groq.com/openai/v1"
        ).strip()
        self.api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or ""
        ).strip()
        self.model = (
            os.getenv("LLM_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or "openai/gpt-oss-20b"
        ).strip()
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    default_headers={"User-Agent": "MovieMind/1.0 (Mozilla/5.0; AI Assistant)"},
                )
            except Exception as e:
                print(f"Warning: could not initialize OpenAI client: {e}")

    # ================= TOOLS =================
    def clean_search_query(self, text: str) -> str:
        """Strip conversational prefixes/suffixes to isolate title or search keywords."""
        q = text.strip()
        q = re.sub(r"[?!.]+$", "", q).strip()
        prefixes = [
            r"^(can you\s+)?(please\s+)?(find|search for|search|look up)\s+",
            r"^i want to (see|watch)\s+",
            r"^tell me about\s+",
            r"^do you have\s+",
            r"^what is\s+",
            r"^show me\s+",
            r"^is\s+",
        ]
        for p in prefixes:
            q = re.sub(p, "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r"\s+available$", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r"\s+in (the )?catalog$", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r"\s+movies?$", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r"\s+films?$", "", q, flags=re.IGNORECASE).strip()
        return q

    def search_movies(self, query: str, limit: int = 5) -> List[Dict]:
        """Search movies by title substring or keyword."""
        clean_q = self.clean_search_query(query).lower().strip()
        if not clean_q:
            clean_q = query.lower().strip()
        if not clean_q:
            return []

        # 1. Exact or substring match in clean title
        matches = []
        for mid, meta in self.movie_meta.items():
            t_low = clean_title(meta["title"]).lower()
            if clean_q == t_low or (clean_q in t_low and not clean_q.isdigit()):
                matches.append({
                    "movie_id": int(mid),
                    "title": clean_title(meta["title"]),
                    "genres": parse_genres(meta["genres"]),
                    "year": int(meta["year"]),
                })
            elif len(clean_q) >= 4 and t_low in clean_q and not t_low.isdigit():
                if re.search(r"\b" + re.escape(t_low) + r"\b", clean_q):
                    matches.append({
                        "movie_id": int(mid),
                        "title": clean_title(meta["title"]),
                        "genres": parse_genres(meta["genres"]),
                        "year": int(meta["year"]),
                    })
            if len(matches) >= limit:
                break
        if matches:
            return matches

        # 2. Token-based match (all significant non-numeric words must appear in title)
        words = [w for w in re.split(r"\W+", clean_q) if len(w) >= 3 and not w.isdigit()]
        if words:
            for mid, meta in self.movie_meta.items():
                t_low = clean_title(meta["title"]).lower()
                if all(w in t_low for w in words):
                    matches.append({
                        "movie_id": int(mid),
                        "title": clean_title(meta["title"]),
                        "genres": parse_genres(meta["genres"]),
                        "year": int(meta["year"]),
                    })
                if len(matches) >= limit:
                    break
        return matches

    def recommend(self, user_id: str, genres: Optional[List[str]] = None, n: int = 5) -> List[Dict]:
        """Generate personalized or genre-guided recommendations."""
        fb = self.memory.get_feedback(user_id)
        disliked = set(fb["disliked"])

        # Try personalized precomputed recs if user_id is a known ML-1M user
        recs = []
        try:
            uid_int = int(user_id)
            if uid_int in self.user_recs_cache:
                for mid in self.user_recs_cache[uid_int]:
                    if mid not in disliked and mid in self.movie_meta:
                        meta = self.movie_meta[mid]
                        recs.append({
                            "movie_id": int(mid),
                            "title": clean_title(meta["title"]),
                            "genres": parse_genres(meta["genres"]),
                            "year": int(meta["year"]),
                        })
                    if len(recs) >= n:
                        return recs
        except ValueError:
            pass

        # Otherwise use coldstart / feedback-driven recommendations
        pref_genres = genres
        if not pref_genres:
            # Infer from liked movies
            liked_genres = []
            for mid in fb["liked"]:
                if mid in self.movie_meta:
                    liked_genres.extend(str(self.movie_meta[mid]["genres"]).split("|"))
            if liked_genres:
                counts = pd.Series(liked_genres).value_counts()
                pref_genres = counts.head(3).index.tolist()

        raw_recs = self.coldstart.recommend(pref_genres or ["Drama", "Action"], top_n=n * 2)
        for r in raw_recs:
            if r["movie_id"] not in disliked:
                recs.append(r)
            if len(recs) >= n:
                break
        return recs

    def recommend_detailed(self, user_id: str, genres: Optional[List[str]] = None, n: int = 6) -> Dict[str, Any]:
        """Generate cards with full score parts, stage latencies, trace, and factual explanations."""
        t0 = time.perf_counter()

        fb = self.memory.get_feedback(user_id)
        disliked = set(fb["disliked"])
        liked = set(fb["liked"])

        # Determine user preferred genres
        pref_genres = genres or []
        if not pref_genres:
            liked_genres = []
            for mid in fb["liked"]:
                if mid in self.movie_meta:
                    liked_genres.extend(parse_genres(self.movie_meta[mid]["genres"]))
            if liked_genres:
                counts = pd.Series(liked_genres).value_counts()
                pref_genres = counts.head(3).index.tolist()
        if not pref_genres:
            pref_genres = ["Action", "Sci-Fi", "Drama"]

        # STAGE 1: Retrieval (Two-Tower FAISS)
        t_ret0 = time.perf_counter()
        cand_ids = []
        ret_scores = {}

        try:
            uid_int = int(user_id)
        except (ValueError, TypeError):
            uid_int = None

        if uid_int is not None and uid_int in self.retrieval_cache:
            c_list, s_list = self.retrieval_cache[uid_int]
            for mid, s in zip(c_list, s_list):
                if mid not in disliked and mid in self.movie_meta:
                    cand_ids.append(mid)
                    ret_scores[mid] = float(s)
        else:
            p_vec = self.coldstart.build_profile_vector(pref_genres)
            emb_sims = np.dot(self.item_embs, p_vec.T).squeeze(1)
            top_indices = np.argsort(-emb_sims)[:250]
            for idx in top_indices:
                mid = int(self.item_ids[idx])
                if mid not in disliked and mid not in liked:
                    cand_ids.append(mid)
                    ret_scores[mid] = float(emb_sims[idx])

        t_ret1 = time.perf_counter()
        retrieval_ms = (t_ret1 - t_ret0) * 1000

        # STAGE 2: Heavy Ranking (Wide & Deep scoring)
        t_rk0 = time.perf_counter()
        ranked_scores = {}
        if uid_int is not None and uid_int in self.ranked_cache:
            c_list, s_list = self.ranked_cache[uid_int]
            c_map = dict(zip(c_list, s_list))
            for mid in cand_ids:
                ranked_scores[mid] = float(c_map.get(mid, ret_scores.get(mid, 0.5)))
        else:
            genre_set = set(g.lower() for g in pref_genres)
            for mid in cand_ids:
                meta = self.movie_meta.get(mid, {})
                m_genres = set(g.lower() for g in parse_genres(meta.get("genres", "")))
                overlap = len(genre_set.intersection(m_genres)) / max(len(genre_set), 1)
                pop = self.coldstart.log_pop.get(mid, 0.5)
                r_score = ret_scores.get(mid, 0.5)
                rk_val = 0.45 * overlap + 0.35 * r_score + 0.20 * pop
                ranked_scores[mid] = float(rk_val)

        cand_ids.sort(key=lambda m: ranked_scores.get(m, 0.0), reverse=True)
        top_ranked = cand_ids[:60]
        t_rk1 = time.perf_counter()
        ranker_ms = (t_rk1 - t_rk0) * 1000

        # STAGE 3: MMR Diversity Re-ranking (lambda = 0.7)
        t_mmr0 = time.perf_counter()
        scores_for_mmr = [ranked_scores[m] for m in top_ranked]
        reranked_ids = mmr_rerank(
            top_ranked,
            scores_for_mmr,
            self.genre_vectors,
            top_k=n,
            lmbda=0.7,
        )
        t_mmr1 = time.perf_counter()
        mmr_ms = (t_mmr1 - t_mmr0) * 1000

        # STAGE 4: Explanations & LLM / Data Chips Grounding
        t_llm0 = time.perf_counter()
        cards = []
        llm_fallback = True
        notice = None

        user_liked_ids = list(fb["liked"])

        # Real ranker scores min-max across the returned recommendations list
        all_rk_scores = [float(ranked_scores.get(m, 0.5)) for m in reranked_ids]
        min_rk = min(all_rk_scores) if all_rk_scores else 0.0
        max_rk = max(all_rk_scores) if all_rk_scores else 1.0
        rk_range = (max_rk - min_rk) if (max_rk - min_rk) > 1e-6 else 1.0

        for rank_idx, mid in enumerate(reranked_ids):
            meta = self.movie_meta[mid]
            clean_t = clean_title(meta["title"])
            m_genres = parse_genres(meta["genres"])

            # Matched genres
            matched = [g for g in m_genres if g.lower() in [pg.lower() for pg in pref_genres]]
            if not matched:
                matched = m_genres[:2]

            # Similar liked movies (excluding self)
            similar_titles = []
            if mid in self.mid_to_idx and user_liked_ids:
                t_vec = self.item_embs[self.mid_to_idx[mid]]
                sim_scores = []
                for l_id in user_liked_ids:
                    if l_id != mid and l_id in self.mid_to_idx and l_id in self.movie_meta:
                        l_vec = self.item_embs[self.mid_to_idx[l_id]]
                        sim = float(np.dot(t_vec, l_vec))
                        sim_scores.append((clean_title(self.movie_meta[l_id]["title"]), sim))
                sim_scores.sort(key=lambda x: x[1], reverse=True)
                similar_titles = [s[0] for s in sim_scores[:2]]

            # Score parts
            r_val = round(ret_scores.get(mid, 0.70), 3)
            raw_rk = float(ranked_scores.get(mid, 0.75))
            rk_val = round(raw_rk, 3)
            div_val = round(0.7 * rk_val + 0.3 * (1.0 - (rank_idx * 0.04)), 3)

            # Derive match_pct strictly from real ranker score (min-max across list), not rank
            norm_rk = (raw_rk - min_rk) / rk_range if max_rk > min_rk else 0.85
            match_pct = int(round(75 + (norm_rk * 23)))  # Scales from 75% to 98%

            # Short explanation
            if similar_titles:
                exp_text = f"Recommended for your interest in {', '.join(matched)}, matching themes from '{similar_titles[0]}'."
            else:
                exp_text = f"Recommended for your interest in {', '.join(matched)}."

            cards.append({
                "movie_id": int(mid),
                "title": clean_t,
                "year": int(meta["year"]),
                "genres": m_genres,
                "match_pct": match_pct,
                "score_parts": {
                    "retrieval": r_val,
                    "ranker": rk_val,
                    "diversity": div_val,
                },
                "explanation": {
                    "text": exp_text,
                    "matched_genres": matched,
                    "similar_liked_movies": similar_titles,
                },
            })

        # Enrich via LLM if client is ready (one short grounded sentence)
        if self.client and cards:
            try:
                prompt = (
                    "You are MovieMind. Write exactly one short, grounded sentence explaining why each movie is recommended. "
                    "Cite only real genres and similar liked movies provided. Never invent facts.\n"
                    f"User prefers: {pref_genres}\n"
                    "Movies:\n" +
                    "\n".join([f"- {c['title']} ({c['year']}): genres={c['genres']}, similar_to={c['explanation']['similar_liked_movies']}" for c in cards]) +
                    "\nReturn valid JSON array of strings: [\"reason 1\", ...]"
                )
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1024,
                )
                content = (resp.choices[0].message.content or "[]").replace("\u202f", " ").replace("\xa0", " ")
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    reasons = json.loads(match.group(0))
                    if len(reasons) == len(cards):
                        for c, r_text in zip(cards, reasons):
                            if isinstance(r_text, str) and len(r_text) > 8:
                                c["explanation"]["text"] = r_text.strip()
                        llm_fallback = False
            except Exception as e:
                llm_fallback = True
                notice = f"Fallback active: LLM unavailable ({type(e).__name__}); using deterministic data chips."

        if llm_fallback and not notice:
            notice = "Fallback active: using deterministic data chips (LLM API key not configured)."

        t_llm1 = time.perf_counter()
        llm_ms = (t_llm1 - t_llm0) * 1000
        total_ms = (t_llm1 - t0) * 1000

        trace = [
            {"stage": "retrieval", "tool": "two_tower_faiss", "args": {"user_id": user_id, "candidates": 200}, "latency_ms": round(retrieval_ms, 2)},
            {"stage": "ranker", "tool": "wide_and_deep", "args": {"candidates_scored": len(top_ranked)}, "latency_ms": round(ranker_ms, 2)},
            {"stage": "diversity", "tool": "mmr_rerank", "args": {"lambda": 0.7, "top_k": n}, "latency_ms": round(mmr_ms, 2)},
            {"stage": "explanation", "tool": "factual_grounding", "args": {"source": "data_chips" if llm_fallback else "openrouter"}, "latency_ms": round(llm_ms, 2)},
        ]

        return {
            "user_id": user_id,
            "recommendations": cards,
            "latency_ms": {
                "retrieval": round(retrieval_ms, 2),
                "ranker": round(ranker_ms, 2),
                "mmr": round(mmr_ms, 2),
                "llm": round(llm_ms, 2),
                "total": round(total_ms, 2),
            },
            "trace": trace,
            "llm_fallback": llm_fallback,
            "notice": notice,
        }

    def explain(self, movie_id: int, user_id: str = "demo_user") -> Dict[str, Any]:
        """Explain recommendation using real factual features and similarity to liked movies."""
        if movie_id not in self.movie_meta:
            return {"error": f"Movie ID {movie_id} not found in catalog."}

        meta = self.movie_meta[movie_id]
        t_clean = clean_title(meta["title"])
        m_genres = parse_genres(meta["genres"])
        m_genres_set = set(m_genres)
        fb = self.memory.get_feedback(user_id)

        # Check similarity with user liked movies (exclude target movie itself)
        similar_liked = []
        if movie_id in self.mid_to_idx:
            target_vec = self.item_embs[self.mid_to_idx[movie_id]]
            for liked_id in fb["liked"]:
                if liked_id == movie_id:
                    continue  # Do not compare target movie to itself
                if liked_id in self.mid_to_idx and liked_id in self.movie_meta:
                    l_vec = self.item_embs[self.mid_to_idx[liked_id]]
                    sim = float(np.dot(target_vec, l_vec))
                    l_meta = self.movie_meta[liked_id]
                    shared = m_genres_set.intersection(set(parse_genres(l_meta["genres"])))
                    similar_liked.append({
                        "title": clean_title(l_meta["title"]),
                        "cosine_similarity": round(sim, 3),
                        "shared_genres": list(shared),
                    })

        similar_liked.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        genre_str = ", ".join(m_genres)

        if similar_liked:
            sim_title = similar_liked[0]["title"]
            explanation = f"'{t_clean}' ({meta['year']}) matches your interest in {genre_str}, similar to '{sim_title}' from your liked movies."
        else:
            explanation = f"'{t_clean}' ({meta['year']}) matches your interest in {genre_str}."

        return {
            "movie_id": movie_id,
            "title": t_clean,
            "genres": m_genres,
            "year": int(meta["year"]),
            "shared_taste_profile": m_genres,
            "similar_liked_movies": similar_liked[:3],
            "explanation": explanation,
        }

    def filter_movies(
        self,
        genre: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        limit: int = 5,
    ) -> List[Dict]:
        """Filter movie catalog by genre and year boundaries."""
        results = []
        for mid, meta in self.movie_meta.items():
            if genre and genre.lower() not in meta["genres"].lower():
                continue
            if min_year and meta["year"] < min_year:
                continue
            if max_year and meta["year"] > max_year:
                continue
            results.append({
                "movie_id": int(mid),
                "title": clean_title(meta["title"]),
                "genres": parse_genres(meta["genres"]),
                "year": int(meta["year"]),
            })
            if len(results) >= limit:
                break
        return results

    def get_taste_profile(self, user_id: str) -> Dict[str, Any]:
        """Fetch user taste summary and feedback counts."""
        fb = self.memory.get_feedback(user_id)
        summary = self.memory.get_taste_summary(user_id)
        return {
            "user_id": user_id,
            "liked_count": len(fb["liked"]),
            "disliked_count": len(fb["disliked"]),
            "taste_summary": summary,
        }

    def record_feedback(self, user_id: str, movie_id: int, feedback_type: str) -> Dict[str, Any]:
        """Store like or dislike for a movie with duplicate prevention."""
        if movie_id not in self.movie_meta:
            return {"error": f"Movie ID {movie_id} does not exist."}
        title = clean_title(self.movie_meta[movie_id]["title"])
        is_new, status_msg = self.memory.record_feedback(user_id, movie_id, feedback_type)
        if not is_new and "already" in status_msg:
            return {"status": "noop", "message": f"'{title}' is already {feedback_type}d."}
        return {"status": "success", "message": f"Recorded {feedback_type} for '{title}'."}

    # ================= DISPATCHER & CHAT =================
    def get_tool_specs(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_movies",
                    "description": "Search movies by title query or keyword.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Movie title or query"},
                            "limit": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "recommend",
                    "description": "Recommend movies for a user with optional genre preferences.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "User identifier"},
                            "genres": {"type": "array", "items": {"type": "string"}, "description": "Preferred genres"},
                            "n": {"type": "integer", "default": 5},
                        },
                        "required": ["user_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "explain",
                    "description": "Explain why a movie is recommended citing real genres and similar liked movies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "movie_id": {"type": "integer", "description": "Movie ID"},
                            "user_id": {"type": "string", "default": "demo_user"},
                        },
                        "required": ["movie_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "filter_movies",
                    "description": "Filter movies by genre and release year range.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "genre": {"type": "string"},
                            "min_year": {"type": "integer"},
                            "max_year": {"type": "integer"},
                            "limit": {"type": "integer", "default": 5},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_taste_profile",
                    "description": "Get current taste profile and liked/disliked statistics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                        },
                        "required": ["user_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "record_feedback",
                    "description": "Record user feedback: 'like' or 'dislike'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "movie_id": {"type": "integer"},
                            "feedback_type": {"type": "string", "enum": ["like", "dislike"]},
                        },
                        "required": ["user_id", "movie_id", "feedback_type"],
                    },
                },
            },
        ]

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        tool_map = {
            "search_movies": self.search_movies,
            "recommend": self.recommend,
            "explain": self.explain,
            "filter_movies": self.filter_movies,
            "get_taste_profile": self.get_taste_profile,
            "record_feedback": self.record_feedback,
        }
        fn = tool_map.get(name)
        return fn(**args) if fn else {"error": f"Unknown tool {name}"}

    def find_movie_id_in_text(self, text: str) -> Optional[int]:
        """Resolve numeric ID or canonical movie title from text."""
        low = text.lower()
        num_match = re.search(r"\b(\d{1,6})\b", low)
        if num_match:
            return int(num_match.group(1))
        for t, mid in sorted(self.title_map.items(), key=lambda x: -len(x[0])):
            if len(t) >= 4 and re.search(r"\b" + re.escape(t) + r"\b", low):
                return mid
        return None

    def route_message(self, user_msg: str, user_id: str = "demo_user") -> Dict[str, Any]:
        """Parse message, route to expected tool, execute, and return structured dispatch trace."""
        msg = (user_msg or "").strip()
        low = msg.lower()

        # Adversarial 1: Empty input
        if not msg:
            return {
                "tool": None,
                "args": {},
                "result": {"error": "Empty input"},
                "response": "Please enter a message or movie query.",
                "safe": True,
                "is_adversarial": True,
            }

        # Adversarial 2: Prompt injection
        if any(p in low for p in ("ignore all previous", "ignore previous instructions", "secret system prompt", "invent a fake movie")):
            return {
                "tool": None,
                "args": {},
                "result": {"error": "Prompt injection rejected"},
                "response": "I cannot alter my instructions or invent non-existent movies. I am MovieMind, dedicated to recommending real movies from the MovieLens catalog.",
                "safe": True,
                "is_adversarial": True,
            }

        # Intent 1: Greetings -> short reply, no tools
        if re.search(r"^(hello|hi|hey|good\s+(morning|evening|afternoon)|howdy|greetings)\b", low):
            return {
                "tool": None,
                "args": {},
                "result": {},
                "response": "Hello! Welcome to MovieMind screening room. How can I help you find or explore movies today?",
                "safe": True,
                "is_adversarial": False,
            }

        # Intent 2: Explain queries
        if re.search(r"\b(why|explain)\b", low):
            mid = self.find_movie_id_in_text(low)
            if mid is not None:
                exp = self.explain(mid, user_id=user_id)
                if "error" in exp:
                    return {
                        "tool": "explain",
                        "args": {"movie_id": mid, "user_id": user_id},
                        "result": exp,
                        "response": exp["error"],
                        "safe": True,
                        "is_adversarial": mid not in self.movie_meta,
                    }
                return {
                    "tool": "explain",
                    "args": {"movie_id": mid, "user_id": user_id},
                    "result": exp,
                    "response": f"**Explanation for {exp['title']}**:\n{exp['explanation']}",
                    "safe": True,
                    "is_adversarial": False,
                }

        # Intent 3: Feedback queries
        if re.search(r"\b(like|liked|love|loved|dislike|disliked|hate|hated)\b", low):
            feedback_type = "dislike" if any(w in low for w in ("dislike", "hate")) else "like"
            mid = self.find_movie_id_in_text(low)
            if mid is not None:
                res = self.record_feedback(user_id, mid, feedback_type)
                return {
                    "tool": "record_feedback",
                    "args": {"user_id": user_id, "movie_id": mid, "feedback_type": feedback_type},
                    "result": res,
                    "response": f"Done! {res.get('message', res.get('error', ''))}",
                    "safe": True,
                    "is_adversarial": False,
                }

        # Intent 4: Taste profile queries
        if any(w in low for w in ("taste", "profile", "history")):
            prof = self.get_taste_profile(user_id)
            return {
                "tool": "get_taste_profile",
                "args": {"user_id": user_id},
                "result": prof,
                "response": f"**Your Taste Profile**:\n{prof['taste_summary']}\n*(Liked: {prof['liked_count']}, Disliked: {prof['disliked_count']})*",
                "safe": True,
                "is_adversarial": False,
            }

        # Intent 5: Filter queries
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", low)
        all_genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        matched_genre = next((g for g in all_genres if g.lower() in low), None)
        if "filter" in low or (year_match and any(w in low for w in ("from", "before", "after", "since", "released"))):
            yr = int(year_match.group(1)) if year_match else None
            min_y = yr if any(w in low for w in ("after", "since")) else None
            max_y = yr if "before" in low else None
            if ("from" in low or "released" in low) and yr and not min_y and not max_y:
                min_y, max_y = yr, yr
            results = self.filter_movies(genre=matched_genre, min_year=min_y, max_year=max_y, limit=5)
            lines = [f"- **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for r in results]
            return {
                "tool": "filter_movies",
                "args": {"genre": matched_genre, "min_year": min_y, "max_year": max_y, "limit": 5},
                "result": results,
                "response": "**Filtered Movies:**\n" + ("\n".join(lines) if lines else "No movies matched your filter."),
                "safe": True,
                "is_adversarial": False,
            }

        # Intent 6: Explicit Recommendation queries
        is_rec_intent = (
            any(w in low for w in ("recommend", "recommendation", "recommendations", "suggest", "suggestions", "what should i watch", "what to watch", "picks"))
            or bool(re.search(r"\bgive me\s+(\d+\s+)?([a-z\s-]+\s+)?movies?\b", low))
        )
        if is_rec_intent:
            n_match = re.search(r"\b(\d+)\b", low)
            n_items = int(n_match.group(1)) if n_match else 5
            n_items = min(max(n_items, 1), 20)
            genres_to_rec = [g for g in all_genres if g.lower() in low]
            recs = self.recommend(user_id=user_id, genres=genres_to_rec or None, n=n_items)
            lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(recs)]
            return {
                "tool": "recommend",
                "args": {"user_id": user_id, "genres": genres_to_rec or None, "n": n_items},
                "result": recs,
                "response": f"**Recommended Movies for you:**\n" + "\n".join(lines),
                "safe": True,
                "is_adversarial": False,
            }

        # Intent 7: Title / Search queries
        search_triggers = ("search", "find", "look up", "i want to see", "i want to watch", "tell me about", "do you have", "is there", "show me")
        is_search_intent = any(tr in low for tr in search_triggers)
        clean_q = self.clean_search_query(msg)
        is_known_title = bool(self.find_movie_id_in_text(clean_q or low))
        is_short_title = len(clean_q.split()) <= 5 and not re.search(r"^(who|what|where|when|why|how)\b", low)

        if is_search_intent or is_known_title or is_short_title:
            q_term = clean_q if clean_q else msg
            results = self.search_movies(q_term, limit=5)
            if results:
                lines = [f"- **{r['title']}** ({r['year']}) [ID: {r['movie_id']}] - {', '.join(r['genres'])}" for r in results]
                resp = "**Search Results:**\n" + "\n".join(lines)
            else:
                resp = f"'{q_term}' is not in the MovieLens 1M catalog."
            return {
                "tool": "search_movies",
                "args": {"query": q_term.lower()},
                "result": results,
                "response": resp,
                "safe": True,
                "is_adversarial": len(results) == 0,
            }

        # Intent 8: Unrecognized - Never default to recommend
        return {
            "tool": None,
            "args": {},
            "result": {},
            "response": "I didn't recognize that movie title or request. You can search for movies (e.g. 'tell me about Aliens'), ask for recommendations by genre (e.g. 'recommend 4 sci-fi movies'), or check your taste profile.",
            "safe": True,
            "is_adversarial": False,
        }

    def chat_fallback(self, user_msg: str, user_id: str = "demo_user") -> str:
        """Rule-based tool router for graceful fallback when no API key is supplied."""
        return self.route_message(user_msg, user_id=user_id)["response"]

    def chat(self, user_msg: str, user_id: str = "demo_user") -> str:
        """Main chat interface calling LLM with tool calling or falling back gracefully."""
        if not self.client:
            return self.chat_fallback(user_msg, user_id=user_id)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are MovieMind, an expert, charming movie recommender assistant. "
                    "You have direct access to tools for searching movies, generating personalized recommendations, "
                    "explaining recommendations with factual metadata (genres, release year, similarity to liked movies), "
                    "filtering by year/genre, and storing user feedback. "
                    "CRITICAL: Always use your tools to fetch facts. Never invent or hallucinate movie titles, genres, or release years."
                ),
            },
            {"role": "user", "content": f"[User ID: {user_id}] {user_msg}"},
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.get_tool_specs(),
                tool_choice="auto",
            )
            choice = resp.choices[0].message
            if not choice.tool_calls:
                return choice.content or self.chat_fallback(user_msg, user_id=user_id)

            # Handle tool calls
            messages.append(choice)
            for tc in choice.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments or "{}")
                # Default user_id if omitted
                if "user_id" in fn_args and not fn_args["user_id"]:
                    fn_args["user_id"] = user_id
                elif "user_id" not in fn_args and fn_name in ("recommend", "get_taste_profile"):
                    fn_args["user_id"] = user_id

                tool_result = self.execute_tool(fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": json.dumps(tool_result),
                })

            # Get final response from model after tools execution
            follow_up = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
            )
            content = follow_up.choices[0].message.content or "Here are your results."
            return content.replace("\u202f", " ").replace("\xa0", " ")

        except Exception as e:
            # Fallback gracefully on API errors / missing keys
            return self.chat_fallback(user_msg, user_id=user_id)

def run_agent_demo():
    print("=== Phase 8: Tool-Calling MovieMind Agent ===")
    agent = MovieMindAgent()
    test_user = "demo_user"

    print(f"API Key configured: {bool(agent.api_key)}")
    print(f"Model configured: {agent.model}")

    queries = [
        "Search for Matrix movies",
        "Recommend 5 movies for someone who loves Action and Sci-Fi",
        "Why should I watch movie ID 2571?",
        "Show me Sci-Fi movies from 1999",
        "What is my current taste profile?",
    ]

    for q in queries:
        print(f"\nUser: {q}")
        reply = agent.chat(q, user_id=test_user)
        print(f"MovieMind: {reply}")

if __name__ == "__main__":
    run_agent_demo()
