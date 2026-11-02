import re
from pathlib import Path
import pandas as pd
from src.download import download_movielens_1m

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def extract_year(title: str) -> int:
    match = re.search(r"\((\d{4})\)", title)
    return int(match.group(1)) if match else 1990

def prepare_data(data_dir: Path = DATA_DIR):
    raw_dir = download_movielens_1m()

    # Load ratings
    ratings = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    # Load movies
    movies = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    movies["year"] = movies["title"].apply(extract_year)

    # Load users
    users = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )
    users["gender_num"] = (users["gender"] == "M").astype(int)

    # Time-based split per user: last = test, second-last = val, rest = train
    ratings = ratings.sort_values(["user_id", "timestamp"], ascending=[True, True])
    # Distance from end: 0 = last, 1 = second last, >=2 = train
    ratings["rev_rank"] = ratings.groupby("user_id").cumcount(ascending=False)

    test = ratings[ratings["rev_rank"] == 0].drop(columns=["rev_rank"])
    val = ratings[ratings["rev_rank"] == 1].drop(columns=["rev_rank"])
    train = ratings[ratings["rev_rank"] >= 2].drop(columns=["rev_rank"])

    # Save to parquet
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(data_dir / "train.parquet", index=False)
    val.to_parquet(data_dir / "val.parquet", index=False)
    test.to_parquet(data_dir / "test.parquet", index=False)
    movies.to_parquet(data_dir / "movies.parquet", index=False)
    users.to_parquet(data_dir / "users.parquet", index=False)

    print("=== Phase 1: Data Preparation Complete ===")
    print(f"Users: {len(users):,}, Movies: {len(movies):,}")
    print(f"Train ratings: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test, movies, users

if __name__ == "__main__":
    prepare_data()
