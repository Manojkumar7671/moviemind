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
