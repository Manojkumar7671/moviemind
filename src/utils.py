import os
import re
from typing import List

# Ensure safe multi-threading across OSes without shell flags
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def clean_title(title: str) -> str:
    """Strip trailing release year in parentheses, e.g. 'Toy Story (1995)' -> 'Toy Story'."""
    return re.sub(r"\s*\(\d{4}\)$", "", str(title)).strip()

def parse_genres(genres_str: str) -> List[str]:
    """Parse pipe-separated genres string into a list of genre names."""
    if not genres_str:
        return []
    if isinstance(genres_str, (list, tuple)):
        return list(genres_str)
    return [g.strip() for g in str(genres_str).split("|") if g.strip()]
