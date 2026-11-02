import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from src.utils import clean_title

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "memory.db"

class UserMemory:
    """SQLite store of per-user liked/disliked movies and taste summaries."""
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_db()

    def get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    user_id TEXT NOT NULL,
                    movie_id INTEGER NOT NULL,
                    feedback_type TEXT NOT NULL CHECK(feedback_type IN ('like', 'dislike')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, movie_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    taste_summary TEXT,
                    favorite_genres TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def record_feedback(self, user_id: str, movie_id: int, feedback_type: str) -> Tuple[bool, str]:
        if feedback_type not in ("like", "dislike"):
            raise ValueError(f"Invalid feedback_type: {feedback_type}")
        with self.get_conn() as conn:
            cur = conn.execute(
                "SELECT feedback_type FROM feedback WHERE user_id = ? AND movie_id = ?",
                (str(user_id), int(movie_id)),
            )
            existing = cur.fetchone()
            if existing:
                if existing["feedback_type"] == feedback_type:
                    return False, f"already {feedback_type}d"
                # Updated from like to dislike or vice versa
                conn.execute(
                    "UPDATE feedback SET feedback_type = ?, created_at = ? WHERE user_id = ? AND movie_id = ?",
                    (feedback_type, datetime.utcnow(), str(user_id), int(movie_id)),
                )
                conn.commit()
                self.refresh_taste_summary(str(user_id))
                return True, f"updated to {feedback_type}"

            conn.execute(
                "INSERT INTO feedback (user_id, movie_id, feedback_type, created_at) VALUES (?, ?, ?, ?)",
                (str(user_id), int(movie_id), feedback_type, datetime.utcnow()),
            )
            conn.commit()
        self.refresh_taste_summary(str(user_id))
        return True, f"recorded {feedback_type}"

    def get_feedback(self, user_id: str) -> Dict[str, List[int]]:
        with self.get_conn() as conn:
            cur = conn.execute(
                "SELECT movie_id, feedback_type FROM feedback WHERE user_id = ? ORDER BY created_at DESC",
                (str(user_id),),
            )
            rows = cur.fetchall()
        liked = [r["movie_id"] for r in rows if r["feedback_type"] == "like"]
        disliked = [r["movie_id"] for r in rows if r["feedback_type"] == "dislike"]
        return {"liked": liked, "disliked": disliked}

    def get_taste_summary(self, user_id: str) -> str:
        with self.get_conn() as conn:
            cur = conn.execute("SELECT taste_summary FROM user_profiles WHERE user_id = ?", (str(user_id),))
            row = cur.fetchone()
            if row and row["taste_summary"]:
                return row["taste_summary"]
        return "No taste summary yet. Like or dislike some movies or pick favorite genres to build one."

    def set_taste_summary(self, user_id: str, summary: str, favorite_genres: str = ""):
        with self.get_conn() as conn:
            conn.execute("""
                INSERT INTO user_profiles (user_id, taste_summary, favorite_genres, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    taste_summary = excluded.taste_summary,
                    favorite_genres = CASE WHEN excluded.favorite_genres != '' THEN excluded.favorite_genres ELSE user_profiles.favorite_genres END,
                    updated_at = excluded.updated_at
            """, (str(user_id), summary, favorite_genres, datetime.utcnow()))
            conn.commit()

    def refresh_taste_summary(self, user_id: str, movies_df: Optional[pd.DataFrame] = None) -> str:
        """Derives taste profile based on liked/disliked movie genres."""
        fb = self.get_feedback(user_id)
        if not fb["liked"] and not fb["disliked"]:
            return self.get_taste_summary(user_id)

        if movies_df is None:
            movies_df = pd.read_parquet(DATA_DIR / "movies.parquet")
        meta = movies_df.set_index("movie_id").to_dict(orient="index")

        liked_titles = [clean_title(meta[m]["title"]) for m in fb["liked"] if m in meta]
        disliked_titles = [clean_title(meta[m]["title"]) for m in fb["disliked"] if m in meta]

        liked_genres = {}
        for m in fb["liked"]:
            if m in meta:
                for g in str(meta[m]["genres"]).split("|"):
                    liked_genres[g] = liked_genres.get(g, 0) + 1

        top_genres = sorted(liked_genres.items(), key=lambda x: x[1], reverse=True)[:3]
        top_genre_str = ", ".join([g for g, _ in top_genres]) if top_genres else "General"

        summary = (
            f"User prefers {top_genre_str}. "
            f"Liked: {', '.join(liked_titles[:5]) if liked_titles else 'None'}. "
            f"Disliked: {', '.join(disliked_titles[:3]) if disliked_titles else 'None'}."
        )
        self.set_taste_summary(user_id, summary, favorite_genres=top_genre_str)
        return summary

def run_memory_demo():
    print("=== Phase 7: SQLite Memory & Taste Summary Store ===")
    mem = UserMemory()
    test_user = "demo_user"

    # Record some likes and dislikes
    mem.record_feedback(test_user, 1, "like")    # Toy Story (Animation|Children's|Comedy)
    mem.record_feedback(test_user, 260, "like")  # Star Wars: Ep IV (Action|Adventure|Sci-Fi)
    mem.record_feedback(test_user, 2571, "like") # Matrix, The (Action|Sci-Fi|Thriller)
    mem.record_feedback(test_user, 2710, "dislike") # Blair Witch Project (Horror)

    fb = mem.get_feedback(test_user)
    print(f"User '{test_user}' Feedback:")
    print(f"  Liked movie IDs   : {fb['liked']}")
    print(f"  Disliked movie IDs: {fb['disliked']}")

    summary = mem.get_taste_summary(test_user)
    print(f"Taste Summary:\n  {summary}")

if __name__ == "__main__":
    run_memory_demo()
