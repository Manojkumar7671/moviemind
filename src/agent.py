import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from src.coldstart import ColdStartRecommender
from src.memory import UserMemory

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

        self.memory = UserMemory()
        self.coldstart = ColdStartRecommender(data_dir, artifacts_dir)

        # Preloaded candidate tables if available
        self.mmr_path = artifacts_dir / "mmr_candidates.parquet"
        self.user_recs_cache = {}
        if self.mmr_path.exists():
            df = pd.read_parquet(self.mmr_path)
            self.user_recs_cache = {int(r["user_id"]): list(r["candidate_ids"]) for _, r in df.iterrows()}

        # Item embeddings for similarity explanation
        self.item_embs = np.load(artifacts_dir / "item_embeddings.npy")
        self.item_ids = np.load(artifacts_dir / "item_ids.npy")
        self.mid_to_idx = {int(mid): idx for idx, mid in enumerate(self.item_ids)}

        # Setup OpenRouter / OpenAI client
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
            except Exception as e:
                print(f"Warning: could not initialize OpenAI client: {e}")

    # ================= TOOLS =================
    def search_movies(self, query: str, limit: int = 5) -> List[Dict]:
        """Search movies by title substring or keyword."""
        q = query.lower().strip()
        tokens = [t for t in re.split(r"\W+", q) if t and t not in ("movie", "movies", "film", "films")]
        if not tokens:
            tokens = [q]

        matches = []
        for mid, meta in self.movie_meta.items():
            t_low = meta["title"].lower()
            g_low = meta["genres"].lower()
            if q in t_low or q in g_low or any(tok in t_low for tok in tokens):
                matches.append({
                    "movie_id": int(mid),
                    "title": meta["title"],
                    "genres": meta["genres"],
                    "year": int(meta["year"]),
                })
            if len(matches) >= limit * 2:
                break
        return matches[:limit]

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
                            "title": meta["title"],
                            "genres": meta["genres"],
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

    def explain(self, movie_id: int, user_id: str = "demo_user") -> Dict[str, Any]:
        """Explain recommendation using real factual features and similarity to liked movies."""
        if movie_id not in self.movie_meta:
            return {"error": f"Movie ID {movie_id} not found in catalog."}

        meta = self.movie_meta[movie_id]
        m_genres = set(str(meta["genres"]).split("|"))
        fb = self.memory.get_feedback(user_id)

        # Check similarity with user liked movies
        similar_liked = []
        if movie_id in self.mid_to_idx:
            target_vec = self.item_embs[self.mid_to_idx[movie_id]]
            for liked_id in fb["liked"]:
                if liked_id in self.mid_to_idx and liked_id in self.movie_meta:
                    l_vec = self.item_embs[self.mid_to_idx[liked_id]]
                    sim = float(np.dot(target_vec, l_vec))
                    l_meta = self.movie_meta[liked_id]
                    shared = m_genres.intersection(set(str(l_meta["genres"]).split("|")))
                    similar_liked.append({
                        "title": l_meta["title"],
                        "cosine_similarity": round(sim, 3),
                        "shared_genres": list(shared),
                    })

        similar_liked.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return {
            "movie_id": movie_id,
            "title": meta["title"],
            "genres": meta["genres"],
            "year": int(meta["year"]),
            "shared_taste_profile": list(m_genres),
            "similar_liked_movies": similar_liked[:3],
            "explanation": (
                f"'{meta['title']}' ({meta['year']}) matches your preference for {meta['genres']}."
                + (f" It is structurally similar to '{similar_liked[0]['title']}' you liked." if similar_liked else "")
            ),
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
                "title": meta["title"],
                "genres": meta["genres"],
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
        """Store like or dislike for a movie."""
        if movie_id not in self.movie_meta:
            return {"error": f"Movie ID {movie_id} does not exist."}
        self.memory.record_feedback(user_id, movie_id, feedback_type)
        title = self.movie_meta[movie_id]["title"]
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

    def chat_fallback(self, user_msg: str, user_id: str = "demo_user") -> str:
        """Rule-based tool router for graceful fallback when no API key is supplied."""
        low = user_msg.lower()

        # Check for feedback (e.g. "I liked The Matrix" or "like 2571")
        if "like " in low or "liked " in low or "dislike " in low or "disliked " in low:
            feedback_type = "dislike" if ("dislike" in low or "hate" in low) else "like"
            # Check for movie ID
            id_match = re.search(r"\b(\d{1,5})\b", low)
            mid = None
            if id_match:
                mid = int(id_match.group(1))
            else:
                # Search for title in catalog
                for title, tid in self.title_to_id.items():
                    clean_title = re.sub(r"\s*\(\d{4}\)$", "", title)
                    if clean_title in low:
                        mid = tid
                        break
            if mid and mid in self.movie_meta:
                res = self.record_feedback(user_id, mid, feedback_type)
                return f"Done! {res['message']}"

        # Check for profile
        if "taste" in low or "profile" in low or "history" in low:
            prof = self.get_taste_profile(user_id)
            return f"**Your Taste Profile**:\n{prof['taste_summary']}\n*(Liked: {prof['liked_count']}, Disliked: {prof['disliked_count']})*"

        # Check for explanation (e.g. "why recommend 1?" or "explain 2571")
        if "why" in low or "explain" in low:
            id_match = re.search(r"\b(\d{1,5})\b", low)
            if id_match:
                mid = int(id_match.group(1))
                exp = self.explain(mid, user_id=user_id)
                return f"**Explanation for {exp['title']}**:\n{exp['explanation']}"

        # Check for filter (e.g. "movies from 1995" or "sci-fi movies after 1990")
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", low)
        genre_found = next((g for g in self.movies_df["genres"].unique() if g.lower() in low), None)
        if "filter" in low or (year_match and ("before" in low or "after" in low or "from" in low)):
            yr = int(year_match.group(1)) if year_match else None
            min_y = yr if ("after" in low or "since" in low) else None
            max_y = yr if ("before" in low) else None
            if "from" in low and yr:
                min_y, max_y = yr, yr
            results = self.filter_movies(genre=genre_found, min_year=min_y, max_year=max_y, limit=5)
            lines = [f"- **{r['title']}** ({r['year']}) - [{r['genres']}]" for r in results]
            return "**Filtered Movies:**\n" + ("\n".join(lines) if lines else "No movies matched your filter.")

        # Check for search
        if "search" in low or "find" in low:
            query = re.sub(r"^(search for|find|search)\s+", "", low).strip()
            results = self.search_movies(query, limit=5)
            lines = [f"- **{r['title']}** ({r['year']}) [ID: {r['movie_id']}] - {r['genres']}" for r in results]
            return "**Search Results:**\n" + ("\n".join(lines) if lines else f"No movies found matching '{query}'.")

        # Default: recommend
        genres_to_rec = [g for g in ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Animation", "Romance"] if g.lower() in low]
        recs = self.recommend(user_id=user_id, genres=genres_to_rec or None, n=5)
        lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{r['genres']}]" for i, r in enumerate(recs)]
        return f"**Recommended Movies for you:**\n" + "\n".join(lines)

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
            )
            return follow_up.choices[0].message.content or "Here are your results."

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
