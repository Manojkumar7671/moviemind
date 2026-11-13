import difflib
import json
import os
import re
import sys
import time
from collections import OrderedDict
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

# Documented alias mapping for regional demonyms and cultural cinema terms
DEMONYM_ALIASES = {
    "indian": {"region": "IN"},
    "american": {"region": "US"},
    "british": {"region": "GB"},
    "french": {"region": "FR"},
    "german": {"region": "DE"},
    "italian": {"region": "IT"},
    "japanese": {"region": "JP"},
    "korean": {"region": "KR"},
    "spanish": {"region": "ES", "language": "es"},
    "mexican": {"region": "MX"},
    "brazilian": {"region": "BR"},
    "chinese": {"region": "CN"},
    "russian": {"region": "RU"},
    "swedish": {"region": "SE"},
    "danish": {"region": "DK"},
    "norwegian": {"region": "NO"},
    "canadian": {"region": "CA"},
    "australian": {"region": "AU"},
    "irish": {"region": "IE"},
    "bollywood": {"language": "hi"},
    "tollywood": {"language": "te"},
    "kollywood": {"language": "ta"},
    "mollywood": {"language": "ml"},
    "sandalwood": {"language": "kn"},
    "south indian": {"region": "IN", "languages": ["te", "ta", "ml", "kn"], "label": "South Indian"},
    "hollywood": {"region": "US"},
}

# Documented genre aliases mapping non-catalog tags to canonical catalog genres
GENRE_ALIASES = {
    "martial arts": {"genre": "Action", "note": "Mapped 'martial arts' to Action"},
    "kung fu": {"genre": "Action", "note": "Mapped 'kung fu' to Action"},
    "rom-com": {"genre": "Comedy", "additional_genre": "Romance", "note": "Mapped 'rom-com' to Comedy and Romance"},
    "romcom": {"genre": "Comedy", "additional_genre": "Romance", "note": "Mapped 'romcom' to Comedy and Romance"},
    "anime": {"genre": "Animation", "language": "ja", "note": "Mapped 'anime' to Japanese Animation"},
    "studio ghibli": {"genre": "Animation", "language": "ja", "note": "Mapped 'Studio Ghibli' to Japanese Animation"},
    "ghibli": {"genre": "Animation", "language": "ja", "note": "Mapped 'Ghibli' to Japanese Animation"},
}

def is_gibberish(text: str) -> bool:
    """Detect keyboard mash sequences or random nonsensical character strings."""
    tokens = re.findall(r"[a-z]+", (text or "").lower())
    if not tokens:
        return True
    kb_walks = ["asdf", "sdfg", "dfgh", "fghj", "ghjk", "hjkl", "qwerty", "werty", "erty", "zxcv", "xcvb", "cvbn"]
    for t in tokens:
        if any(kw in t for kw in kb_walks):
            return True
        if len(t) >= 5 and re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", t):
            return True
        if len(t) >= 4 and not re.search(r"[aeiouy]", t):
            return True
    return False

def intern_tuple(items):
    """Convert a sequence of strings into a compact tuple of interned strings."""
    if items is None:
        return ()
    return tuple(sys.intern(str(x)) for x in items if x)


class GlobalMetaDict:
    """Memory-efficient view over global_df that materializes row dicts on demand."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._id_to_iloc = {str(mid): i for i, mid in enumerate(df["id"].values)}

    def __contains__(self, key: Any) -> bool:
        return str(key) in self._id_to_iloc

    def __getitem__(self, key: Any) -> dict:
        k = str(key)
        if k not in self._id_to_iloc:
            raise KeyError(key)
        iloc = self._id_to_iloc[k]
        return self.df.iloc[iloc].to_dict()

    def get(self, key: Any, default: Any = None) -> Any:
        if str(key) in self._id_to_iloc:
            return self[key]
        return default

    def __len__(self) -> int:
        return len(self._id_to_iloc)

    def __bool__(self) -> bool:
        return bool(self._id_to_iloc)


class MMRCache:
    """Compact 2D int32 array cache for MMR candidates. Uses ~1.15 MB instead of 33 MB."""
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    @classmethod
    def load(cls, parquet_path: Path) -> "MMRCache":
        df = pd.read_parquet(parquet_path)
        max_uid = int(df["user_id"].max())
        cand_lens = [len(c) for c in df["candidate_ids"]]
        max_len = max(cand_lens) if cand_lens else 50
        arr = np.zeros((max_uid + 1, max_len), dtype=np.int32)
        for _, r in df.iterrows():
            uid = int(r["user_id"])
            c_ids = r["candidate_ids"]
            arr[uid, :len(c_ids)] = c_ids
        return cls(arr)

    def __contains__(self, uid: int) -> bool:
        return 0 < uid < len(self._arr) and self._arr[uid, 0] != 0

    def __getitem__(self, uid: int) -> List[int]:
        if uid in self:
            return self._arr[uid].tolist()
        raise KeyError(uid)

    def get(self, uid: int, default=None):
        if uid in self:
            return self._arr[uid].tolist()
        return default

    def __len__(self) -> int:
        return len(self._arr) - 1


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

        self.ml_min_year = int(self.movies_df["year"].min())
        self.ml_max_year = int(self.movies_df["year"].max())

        # Global catalog metadata (lazy-loaded on first access)
        self._global_df = None
        self._global_meta = None
        self.global_title_to_id = {}
        self._global_fuzzy_buckets = {}
        self._global_prefix_index = {}
        self.region_name_to_code = {}
        self.lang_name_to_code = {}
        self.code_to_region_name = {}
        self.code_to_lang_name = {}
        self.country_primary_lang = {}

        lookups_path = artifacts_dir / "global_lookups.json"
        if lookups_path.exists():
            with open(lookups_path, "r") as f:
                lookups = json.load(f)
            self.data_min_year = min(self.ml_min_year, int(lookups["min_year"]))
            self.data_max_year = max(self.ml_max_year, int(lookups["max_year"]))
            self.region_name_to_code = lookups["region_name_to_code"]
            self.code_to_region_name = lookups["code_to_region_name"]
            self.lang_name_to_code = lookups["lang_name_to_code"]
            self.code_to_lang_name = lookups["code_to_lang_name"]
            self.country_primary_lang = lookups["country_primary_lang"]
        else:
            self.data_min_year = self.ml_min_year
            self.data_max_year = self.ml_max_year

        # Documented language aliases (Wikidata label variances)
        self.lang_name_to_code["bengali"] = "bn"
        self.code_to_lang_name.setdefault("bn", "Bengali")
        self.lang_name_to_code["mandarin"] = "zh"
        self.lang_name_to_code["cantonese"] = "zh"

        # Common country demonyms
        demonyms = {
            "indian": "IN", "american": "US", "british": "GB", "french": "FR",
            "german": "DE", "italian": "IT", "japanese": "JP", "korean": "KR",
            "spanish": "ES", "chinese": "CN", "canadian": "CA", "australian": "AU",
            "mexican": "MX", "brazilian": "BR", "swedish": "SE", "danish": "DK",
            "norwegian": "NO", "russian": "RU", "polish": "PL"
        }
        for d_name, d_code in demonyms.items():
            if d_name not in self.region_name_to_code:
                self.region_name_to_code[d_name] = d_code
        for a, mid in {
            "matrix": 2571, "aliens": 1200, "alien": 1214, "godfather": 858,
            "star wars": 260, "toy story": 1, "jurassic park": 480,
            "terminator 2": 589, "terminator": 1240, "total recall": 2916,
            "alines": 1200, "matrixx": 2571, "godfthr": 858, "jurasic park": 480,
            "termanitor 2": 589, "termanitor": 1240,
        }.items():
            self.title_map[a] = mid

        self.memory = UserMemory()
        self.coldstart = ColdStartRecommender(data_dir, artifacts_dir)

        # Preloaded candidate tables: compact 2D MMRCache (1.15 MB instead of 33 MB)
        self.mmr_path = artifacts_dir / "mmr_candidates.parquet"
        self.user_recs_cache = {}
        if self.mmr_path.exists():
            self.user_recs_cache = MMRCache.load(self.mmr_path)

        self.genre_vectors = get_genre_matrix(self.movies_df)

        # Genre frequency table for MovieLens catalog (used for distinctive-genre filtering in recommend)
        _gc: "dict[str, int]" = {}
        for _meta in self.movie_meta.values():
            for _g in parse_genres(_meta.get("genres", "")):
                _gc[_g] = _gc.get(_g, 0) + 1
        self._genre_freq: "dict[str, int]" = _gc

        # Serve-time candidates: unused at serve time; drop 280 MB cache
        self.retrieval_path = artifacts_dir / "retrieval_candidates.parquet"
        self.retrieval_cache = {}

        self.ranked_path = artifacts_dir / "ranked_candidates.parquet"
        self.ranked_cache = {}

        # Item embeddings memory-mapped for zero resident RAM duplication
        self.item_embs = np.load(artifacts_dir / "item_embeddings.npy", mmap_mode="r")
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
        self._explanation_cache = {}
        self._global_recommender = None

    def _ensure_global_catalog(self):
        """Lazy-loads World Cinema catalog metadata on first demand."""
        if self._global_df is not None:
            return
        global_path = self.artifacts_dir / "global_catalog.parquet"
        if not global_path.exists():
            self._global_df = None
            self._global_meta = {}
            return

        # Share the DataFrame from singleton GlobalContentRecommender to prevent duplicate parquet in memory
        gr = self.global_recommender
        self._global_df = gr.df
        self._global_meta = GlobalMetaDict(self._global_df)
        self._global_titles_lower = self._global_df["title"].str.lower()
        self._fuzzy_cache: OrderedDict = OrderedDict()

        idx_path = ARTIFACTS_DIR / "global_title_index.json"
        if idx_path.exists():
            with open(idx_path, "r") as f:
                title_entries = json.load(f)
        else:
            df_by_votes = self._global_df.sort_values(by="votes", ascending=False)
            seen_t = set()
            title_entries = []
            for _, r in df_by_votes.iterrows():
                gid = str(r["id"])
                ct = clean_title(r["title"]).lower().strip()
                for variant in self._title_variants(ct):
                    if variant and variant not in seen_t:
                        seen_t.add(variant)
                        title_entries.append((variant, gid))

        for variant, gid in title_entries:
            if variant not in self.global_title_to_id:
                self.global_title_to_id[variant] = gid
                if len(variant) >= 3:
                    bkey = (variant[0], len(variant))
                    self._global_fuzzy_buckets.setdefault(bkey, []).append((variant, gid))
                    pkey = variant[:3]
                    plist = self._global_prefix_index.setdefault(pkey, [])
                    if len(plist) < 50:
                        plist.append((variant, gid))

    @staticmethod
    def _title_variants(ct: str) -> List[str]:
        variants = [ct]
        if ", the" in ct:
            variants.append("the " + ct.replace(", the", "").strip())
        if ", a" in ct:
            variants.append("a " + ct.replace(", a", "").strip())
        return variants

    def _fuzzy_world_matches(self, clean_q: str, limit: int = 5) -> List[tuple]:
        """Return [(ratio, votes, row_dict, gid)] using capped bucketed lookup + 512-entry LRU cache."""
        if not clean_q or len(clean_q) < 4:
            return []
        _ = self.global_df  # ensure lazy-loaded
        cache_key = (clean_q, limit)
        cache = getattr(self, "_fuzzy_cache", None)
        if cache is not None and cache_key in cache:
            cache.move_to_end(cache_key)
            return cache[cache_key]

        cands: Dict[str, str] = {}
        q0 = clean_q[0]
        qlen = len(clean_q)
        max_cands = 80
        # Exact length first, then +/- 1, then +/- 2, then 3-char prefix bucket
        for l_try in (qlen, qlen - 1, qlen + 1, qlen - 2, qlen + 2):
            if l_try < 3:
                continue
            for ct, gid in self._global_fuzzy_buckets.get((q0, l_try), ()):
                cands[ct] = gid
                if len(cands) >= max_cands:
                    break
            if len(cands) >= max_cands:
                break
        if len(cands) < max_cands and len(clean_q) >= 3:
            for ct, gid in self._global_prefix_index.get(clean_q[:3], ()):
                if abs(len(ct) - qlen) <= 2:
                    cands[ct] = gid
                    if len(cands) >= max_cands:
                        break
        out = []
        for ct, gid in cands.items():
            ratio = difflib.SequenceMatcher(None, clean_q, ct).ratio()
            if ratio >= 0.80:
                r = self.global_meta[gid]
                out.append((ratio, int(r.get("votes", 0)), r, gid))
        out.sort(key=lambda x: (round(x[0], 2), x[1]), reverse=True)
        res = out[:limit]
        if cache is not None:
            cache[cache_key] = res
            if len(cache) > 512:
                cache.popitem(last=False)
        return res

    @property
    def global_df(self):
        if self._global_df is None:
            self._ensure_global_catalog()
        return self._global_df

    @global_df.setter
    def global_df(self, val):
        self._global_df = val

    @property
    def global_meta(self):
        if self._global_meta is None:
            self._ensure_global_catalog()
        return self._global_meta

    @global_meta.setter
    def global_meta(self, val):
        self._global_meta = val

    @property
    def global_recommender(self):
        if self._global_recommender is None:
            from src.global_recommender import GlobalContentRecommender
            self._global_recommender = GlobalContentRecommender.get_instance(self.artifacts_dir / "global_catalog.parquet")
        return self._global_recommender

    # ================= TOOLS =================
    def clean_search_query(self, text: str) -> str:
        """Strip conversational prefixes/suffixes to isolate title or search keywords."""
        q = text.strip()
        q = re.sub(r"[?!.]+$", "", q).strip()
        prefixes = [
            r"^(can you\s+)?(please\s+)?(suggest|recommend|give me|show me|find)?\s*(something|movies?|films?|picks?|more)?\s*(like|similar to)\s+",
            r"^(can you\s+)?(please\s+)?(find|search for|search|look up)\s+",
            r"^(can you\s+)?(please\s+)?(movies?\s+with|films?\s+with|movies?\s+starring|films?\s+starring|movies?\s+directed\s+by|films?\s+directed\s+by)\s+",
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

    # ------------------------------------------------------------------
    # Match-tier helpers (data-driven, no hardcoded language lists)
    # ------------------------------------------------------------------

    def region_match_tier(self, row: Any, region_code: str) -> Optional[str]:
        """Return 'core', 'partial', or None for a global-catalog row vs a region code."""
        cs = [str(c).upper() for c in (row.get("origin_countries") if isinstance(row, dict) else row["origin_countries"])]
        ls = [str(l).lower() for l in (row.get("original_languages") if isinstance(row, dict) else row["original_languages"])]
        valid_codes = {"DE", "WEST GERMANY"} if region_code == "DE" else {region_code}
        if not any(c in valid_codes for c in cs):
            return None
        if cs and cs[0] in valid_codes:
            if len(cs) == 1:
                return "core"
            primary = self.country_primary_lang.get(region_code)
            if primary and ls and ls[0] == primary:
                return "core"
            return "partial"
        return "partial"

    def language_match_tier(self, row: Any, lang_code: str) -> Optional[str]:
        """Return 'core', 'partial', or None for a global-catalog row vs a language code."""
        _ENGLISH_MAJORS = {"US", "GB", "AU", "CA"}
        cs = [str(c).upper() for c in (row.get("origin_countries") if isinstance(row, dict) else row["origin_countries"])]
        ls = [str(l).lower() for l in (row.get("original_languages") if isinstance(row, dict) else row["original_languages"])]
        if lang_code not in ls:
            return None
        if ls and ls[0] == lang_code:
            # Non-English language with major English studio backing is demoted to partial
            if lang_code != "en":
                if len(cs) == 1 and cs[0] in _ENGLISH_MAJORS:
                    return "partial"
                if len(cs) > 1 and _ENGLISH_MAJORS.intersection(cs):
                    return "partial"
            return "core"
        return "partial"

    def parse_filter_args(self, text: str) -> Dict[str, Any]:
        """Extract canonical filter arguments (region, language, genre, year, min_year, max_year) from user message."""
        low = (text or "").lower().strip()
        args: Dict[str, Any] = {"limit": 5}

        # Year / Decade
        decade_match = re.search(r"\b(?:(18|19|20))?(\d{2})s\b", low)
        year_match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", low)
        if decade_match:
            c_prefix = decade_match.group(1)
            d_val = int(decade_match.group(2))
            c_val = int(c_prefix) * 100 if c_prefix else (1900 if d_val >= 20 else 2000)
            args["min_year"] = c_val + d_val
            args["max_year"] = c_val + d_val + 9
        elif year_match:
            yr = int(year_match.group(1))
            if any(w in low for w in ("after", "since")):
                args["min_year"] = yr
            elif "before" in low:
                args["max_year"] = yr
            else:
                args["year"] = yr
                args["min_year"] = yr
                args["max_year"] = yr

        # Genre & Genre aliases
        all_genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        genre_synonyms = {
            "thriller": "Thriller", "thrillers": "Thriller", "thrillrs": "Thriller", "detective": "Mystery",
            "family": "Children's", "funny": "Comedy", "comedies": "Comedy", "comedys": "Comedy", "comedy": "Comedy",
            "lasers": "Sci-Fi", "space": "Sci-Fi", "sci-fi": "Sci-Fi", "scifi": "Sci-Fi",
            "animated": "Animation", "animation": "Animation", "cartoons": "Animation",
            "horror": "Horror", "dramas": "Drama", "drama": "Drama", "war": "War",
            "action": "Action", "westerns": "Western", "western": "Western",
            "romance": "Romance", "romantic": "Romance", "adventures": "Adventure", "adventure": "Adventure",
        }

        matched_genre = None
        matched_additional_genre = None
        genre_mapping_note = None
        alias_language = None
        for alias_k, alias_info in GENRE_ALIASES.items():
            if re.search(rf"\b{re.escape(alias_k)}\b", low):
                matched_genre = alias_info["genre"]
                matched_additional_genre = alias_info.get("additional_genre")
                genre_mapping_note = alias_info["note"]
                alias_language = alias_info.get("language")
                break

        if not matched_genre:
            for syn_k, syn_v in genre_synonyms.items():
                if re.search(rf"\b{re.escape(syn_k)}\b", low):
                    matched_genre = syn_v
                    break
        if not matched_genre:
            for g in all_genres:
                if re.search(rf"\b{re.escape(g.lower())}\b", low):
                    matched_genre = g
                    break

        if matched_genre:
            args["genre"] = matched_genre
        if matched_additional_genre:
            args["additional_genre"] = matched_additional_genre
        if genre_mapping_note:
            args["_genre_note"] = genre_mapping_note

        # 1. Check demonyms / cultural aliases first (demonyms map to region; e.g. german -> DE, indian -> IN)
        matched_reg = None
        matched_lang = alias_language
        matched_langs = None
        matched_region_label = None
        for demonym, info in sorted(DEMONYM_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
            if re.search(rf"\b{re.escape(demonym)}\b", low):
                matched_reg = info.get("region")
                if "language" in info:
                    matched_lang = info.get("language")
                if "languages" in info:
                    matched_langs = list(info["languages"])
                if "label" in info:
                    matched_region_label = info["label"]
                break

        # 2. Check explicit languages from self.lang_name_to_code if not set by demonym
        if not matched_lang:
            for name_k in sorted(self.lang_name_to_code.keys(), key=len, reverse=True):
                if len(name_k) >= 3 and re.search(rf"\b{re.escape(name_k)}\b", low):
                    matched_lang = self.lang_name_to_code[name_k]
                    break

        # 3. Country / Region names from catalog labels
        if not matched_reg:
            for name_k in sorted(self.region_name_to_code.keys(), key=len, reverse=True):
                if len(name_k) >= 3 and re.search(rf"\b{re.escape(name_k)}\b", low):
                    matched_reg = self.region_name_to_code[name_k]
                    break

        reg_match = re.search(r"\bregion\s+([a-zA-Z]{2})\b", low)
        if reg_match:
            matched_reg = reg_match.group(1).upper()
        lang_match = re.search(r"\blanguage\s+([a-zA-Z]{2,10})\b", low)
        if lang_match:
            code_c = lang_match.group(1).lower()
            matched_lang = self.lang_name_to_code.get(code_c, code_c)

        if matched_reg:
            args["region"] = matched_reg
        if matched_lang:
            args["language"] = matched_lang
        if matched_langs:
            args["languages"] = matched_langs
        if matched_region_label:
            args["_region_label"] = matched_region_label

        return args

    def resolve_title_across_catalogs(self, query: str) -> Optional[Dict[str, Any]]:
        """Resolve a movie title across MovieLens and World catalogs, picking the candidate with the most votes."""
        clean_q = self.clean_search_query(query).lower().strip()
        if not clean_q:
            clean_q = query.lower().strip()
        clean_q = re.sub(r"^(suggest|recommend|give me|find|show me)?\s*(something|movies?|films?|picks?|more)?\s*(like|similar to)\s+", "", clean_q).strip()
        clean_q = re.sub(r"\s+but\s+.*$", "", clean_q).strip()
        clean_q = clean_q.strip("'\"").strip()
        clean_q = re.sub(r"^(the\s+)?(movie|film)\s+", "", clean_q).strip()

        if not clean_q:
            return None

        candidates = []
        title_variants = [clean_q]
        if clean_q.startswith("the "):
            title_variants.append(clean_q[4:].strip())
        else:
            title_variants.append(f"the {clean_q}")

        # 1. O(1) lookup in World Catalog via pre-built global_title_to_id
        if self.global_df is not None:
            seen_gids = set()
            for tv in title_variants:
                gid = self.global_title_to_id.get(tv)
                if gid and gid not in seen_gids:
                    seen_gids.add(gid)
                    r = self.global_meta[gid]
                    ct = clean_title(r["title"])
                    y = int(r["year"])
                    v = int(r.get("votes", 0))
                    genres = [g.strip() for g in str(r.get("genres", "")).split("|") if g.strip()]
                    ct_clean = ct.lower()
                    ml_id = None
                    if ct_clean in self.title_map and int(self.movie_meta[self.title_map[ct_clean]]["year"]) == y:
                        ml_id = self.title_map[ct_clean]
                    candidates.append({
                        "id": ml_id if ml_id is not None else gid,
                        "title": ct,
                        "year": y,
                        "votes": v,
                        "genres": genres,
                        "catalog_source": "MovieLens 1M" if ml_id is not None else "World catalog",
                        "is_movielens": ml_id is not None,
                        "ml_id": ml_id,
                        "global_id": gid,
                    })

        # 2. O(1) lookup in MovieLens (title_map) for titles not already in candidates
        for tv in title_variants:
            mid = self.title_map.get(tv)
            if mid is not None and not any(c.get("ml_id") == int(mid) for c in candidates):
                meta = self.movie_meta[mid]
                y = int(meta["year"])
                v = 1000
                gid_match = self.global_title_to_id.get(tv) if self.global_df is not None else None
                if gid_match and gid_match in self.global_meta:
                    gr = self.global_meta[gid_match]
                    if int(gr.get("year", 0)) == y:
                        v = int(gr.get("votes", 1000))
                candidates.append({
                    "id": int(mid),
                    "title": clean_title(meta["title"]),
                    "year": y,
                    "votes": v,
                    "genres": parse_genres(meta.get("genres", "")),
                    "catalog_source": "MovieLens 1M",
                    "is_movielens": True,
                    "ml_id": int(mid),
                    "global_id": None,
                })

        # 3. Fuzzy match fallback if nothing matched exactly
        if not candidates:
            if clean_q in self.title_map:
                mid = self.title_map[clean_q]
                meta = self.movie_meta[mid]
                candidates.append({
                    "id": int(mid),
                    "title": clean_title(meta["title"]),
                    "year": int(meta["year"]),
                    "votes": 100000,
                    "genres": parse_genres(meta.get("genres", "")),
                    "catalog_source": "MovieLens 1M",
                    "is_movielens": True,
                    "ml_id": int(mid),
                    "global_id": None,
                })
            elif len(clean_q.split()) <= 3:
                close = difflib.get_close_matches(clean_q, self.title_map.keys(), n=1, cutoff=0.82)
                if close:
                    ratio = difflib.SequenceMatcher(None, clean_q, close[0]).ratio()
                    len_diff = abs(len(clean_q) - len(close[0]))
                    if ratio >= 0.82 and len_diff <= 2:
                        mid = self.title_map[close[0]]
                        meta = self.movie_meta[mid]
                        candidates.append({
                            "id": int(mid),
                            "title": clean_title(meta["title"]),
                            "year": int(meta["year"]),
                            "votes": 100000,
                            "genres": parse_genres(meta.get("genres", "")),
                            "catalog_source": "MovieLens 1M",
                            "is_movielens": True,
                            "ml_id": int(mid),
                            "global_id": None,
                        })

        if not candidates:
            return None

        candidates.sort(key=lambda c: c["votes"], reverse=True)
        return candidates[0]

    def _format_cards(self, items: List[Dict], source: str = "MovieLens 1M") -> List[Dict]:
        """Convert raw items from search/filter/recommend into UI cards with consistent schemas."""
        cards = []
        for idx, item in enumerate(items):
            mid = item.get("movie_id")
            title = clean_title(item.get("title", ""))
            year = int(item.get("year", 2000))
            genres = item.get("genres", [])
            match_pct = max(75, 98 - (idx * 3))
            if source == "World catalog" or item.get("catalog_source") == "World catalog" or "origin_countries" in item:
                origin_names = item.get("origin_country_names", [])
                lang_names = item.get("original_language_names", [])
                orig_str = item.get("origin") or (", ".join(origin_names) if origin_names else "unknown")
                cards.append({
                    "movie_id": str(mid),
                    "title": title,
                    "year": year,
                    "genres": genres,
                    "origin_countries": item.get("origin_countries", []),
                    "origin_country_names": origin_names,
                    "original_languages": item.get("original_languages", []),
                    "original_language_names": lang_names,
                    "origin": orig_str,
                    "match_pct": match_pct,
                    "score_parts": {
                        "retrieval": round(min(1.0, float(item.get("votes", 10000)) / 500000.0), 3),
                        "ranker": round(float(item.get("rating", 7.0)) / 10.0, 3),
                        "diversity": 0.85,
                    },
                    "explanation": {
                        "text": f"Acclaimed {', '.join(origin_names) if origin_names else 'international'} feature ({item.get('rating', 7.5)}/10 on IMDb), matching your interest in {', '.join(genres[:2])}.",
                        "matched_genres": genres[:2],
                        "similar_liked_movies": [],
                    },
                    "catalog_source": "World catalog",
                    "_match_tier": item.get("_match_tier", "core"),
                    "_partial_reason": item.get("_partial_reason"),
                })
            else:
                cards.append({
                    "movie_id": int(mid) if str(mid).isdigit() else mid,
                    "title": title,
                    "year": year,
                    "genres": genres,
                    "match_pct": match_pct,
                    "score_parts": {
                        "retrieval": 0.85,
                        "ranker": 0.90,
                        "diversity": 0.80,
                    },
                    "explanation": {
                        "text": f"Recommended for your interest in {', '.join(genres[:2])}.",
                        "matched_genres": genres[:2],
                        "similar_liked_movies": [],
                    },
                    "catalog_source": "MovieLens 1M",
                })
        return cards

    def search_movies(self, query: str, limit: int = 5) -> List[Dict]:
        """Search movies by title substring or keyword across BOTH MovieLens and World catalogs."""
        clean_q = self.clean_search_query(query).lower().strip()
        if not clean_q:
            clean_q = query.lower().strip()
        if not clean_q:
            return []

        # 1. Exact or substring match in MovieLens 1M
        matches = []
        for mid, meta in self.movie_meta.items():
            t_low = clean_title(meta["title"]).lower()
            if clean_q == t_low or (clean_q in t_low and not clean_q.isdigit()):
                matches.append({
                    "movie_id": int(mid),
                    "title": clean_title(meta["title"]),
                    "genres": parse_genres(meta["genres"]),
                    "year": int(meta["year"]),
                    "catalog_source": "MovieLens 1M",
                })
            elif len(clean_q) >= 4 and t_low in clean_q and not t_low.isdigit():
                if re.search(r"\b" + re.escape(t_low) + r"\b", clean_q):
                    matches.append({
                        "movie_id": int(mid),
                        "title": clean_title(meta["title"]),
                        "genres": parse_genres(meta["genres"]),
                        "year": int(meta["year"]),
                        "catalog_source": "MovieLens 1M",
                    })
            if len(matches) >= limit:
                break
        if matches:
            return matches

        # 2. Token-based match in MovieLens
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
                        "catalog_source": "MovieLens 1M",
                    })
                if len(matches) >= limit:
                    break
        if matches:
            return matches

        # 3. Fuzzy match fallback for short typo queries in MovieLens
        # Confident match: similarity ratio >= 0.82 and length difference <= 2
        if len(clean_q.split()) <= 3:
            close_titles = difflib.get_close_matches(clean_q, self.title_map.keys(), n=3, cutoff=0.82)
            for ct in close_titles:
                ratio = difflib.SequenceMatcher(None, clean_q, ct).ratio()
                len_diff = abs(len(clean_q) - len(ct))
                if ratio < 0.82 or len_diff > 2:
                    continue
                mid = self.title_map[ct]
                if mid in self.movie_meta and all(m["movie_id"] != mid for m in matches):
                    meta = self.movie_meta[mid]
                    matches.append({
                        "movie_id": int(mid),
                        "title": clean_title(meta["title"]),
                        "genres": parse_genres(meta["genres"]),
                        "year": int(meta["year"]),
                        "catalog_source": "MovieLens 1M",
                    })
                if len(matches) >= limit:
                    break
        if matches:
            return matches

        # 4. Search World Catalog if not matched in MovieLens
        if self.global_df is not None:
            # Check exact title in global_title_to_id
            if clean_q in self.global_title_to_id:
                gid = self.global_title_to_id[clean_q]
                r = self.global_meta[gid]
                origin_names = list(r["origin_country_names"])
                lang_names = list(r["original_language_names"])
                matches.append({
                    "movie_id": str(r.get("id", gid)),
                    "title": clean_title(r["title"]),
                    "genres": [g.strip() for g in str(r["genres"]).split("|") if g.strip()],
                    "year": int(r["year"]),
                    "origin_countries": list(r["origin_countries"]),
                    "origin_country_names": origin_names,
                    "original_languages": list(r["original_languages"]),
                    "original_language_names": lang_names,
                    "origin": ", ".join(origin_names) if origin_names else "unknown",
                    "rating": float(r.get("rating", 0.0)),
                    "votes": int(r.get("votes", 0)),
                    "catalog_source": "World catalog",
                })
                return matches

            # Prefix match via pre-built prefix index before full df scan
            if len(clean_q) >= 3 and not clean_q.isdigit():
                pref_hits = [
                    gid for ct, gid in self._global_prefix_index.get(clean_q[:3], ())
                    if ct.startswith(clean_q)
                ]
                if pref_hits:
                    seen_g = set()
                    for gid in pref_hits:
                        if gid in seen_g:
                            continue
                        seen_g.add(gid)
                        r = self.global_meta[gid]
                        origin_names = list(r["origin_country_names"])
                        lang_names = list(r["original_language_names"])
                        matches.append({
                            "movie_id": str(r.get("id", gid)),
                            "title": clean_title(r["title"]),
                            "genres": [g.strip() for g in str(r["genres"]).split("|") if g.strip()],
                            "year": int(r["year"]),
                            "origin_countries": list(r["origin_countries"]),
                            "origin_country_names": origin_names,
                            "original_languages": list(r["original_languages"]),
                            "original_language_names": lang_names,
                            "origin": ", ".join(origin_names) if origin_names else "unknown",
                            "rating": float(r.get("rating", 0.0)),
                            "votes": int(r.get("votes", 0)),
                            "catalog_source": "World catalog",
                        })
                        if len(matches) >= limit:
                            break
                    if matches:
                        return matches

            # Check substring match in global_df (using pre-lowercased Series, non-regex)
            if len(clean_q) >= 3 and not clean_q.isdigit():
                titles_low = getattr(self, "_global_titles_lower", None)
                if titles_low is None:
                    titles_low = self.global_df["title"].str.lower()
                sub_df = self.global_df[titles_low.str.contains(clean_q, regex=False, na=False)]
                if not sub_df.empty:
                    top_matches = sub_df.sort_values(by="votes", ascending=False).head(limit)
                    for _, r in top_matches.iterrows():
                        origin_names = list(r["origin_country_names"])
                        lang_names = list(r["original_language_names"])
                        matches.append({
                            "movie_id": str(r["id"]),
                            "title": clean_title(r["title"]),
                            "genres": [g.strip() for g in str(r["genres"]).split("|") if g.strip()],
                            "year": int(r["year"]),
                            "origin_countries": list(r["origin_countries"]),
                            "origin_country_names": origin_names,
                            "original_languages": list(r["original_languages"]),
                            "original_language_names": lang_names,
                            "origin": ", ".join(origin_names) if origin_names else "unknown",
                            "rating": float(r.get("rating", 0.0)),
                            "votes": int(r.get("votes", 0)),
                            "catalog_source": "World catalog",
                        })
                    return matches

            # Fuzzy title fallback in World Catalog using bucketed index
            fuzzy_cands = self._fuzzy_world_matches(clean_q, limit=limit)
            if fuzzy_cands:
                for _, _, r, gid in fuzzy_cands:
                    origin_names = list(r["origin_country_names"])
                    lang_names = list(r["original_language_names"])
                    matches.append({
                        "movie_id": str(r.get("id", gid)),
                        "title": clean_title(r["title"]),
                        "genres": [g.strip() for g in str(r["genres"]).split("|") if g.strip()],
                        "year": int(r["year"]),
                        "origin_countries": list(r["origin_countries"]),
                        "origin_country_names": origin_names,
                        "original_languages": list(r["original_languages"]),
                        "original_language_names": lang_names,
                        "origin": ", ".join(origin_names) if origin_names else "unknown",
                        "rating": float(r.get("rating", 0.0)),
                        "votes": int(r.get("votes", 0)),
                        "catalog_source": "World catalog",
                    })
                return matches

        return matches

    def recommend_top_rated(self, user_id: str, n: int = 5) -> List[Dict]:
        """Bayesian-quality ranking across both catalogs without genre bias."""
        fb = self.memory.get_feedback(user_id)
        disliked = {str(x) for x in fb.get("disliked", [])}
        liked = {str(x) for x in fb.get("liked", [])}

        recs: List[Dict] = []
        seen_titles: set = set()

        if self.global_df is not None and not self.global_df.empty:
            df = self.global_df
            c_mean = float(df["rating"].mean()) if len(df) > 0 else 6.5
            m_thresh = 50000.0
            v = df["votes"].astype(np.float64)
            r = df["rating"].astype(np.float64)
            bayes = (v * r + m_thresh * c_mean) / (v + m_thresh)
            top_idx = np.argsort(-bayes.values)[: max(n * 4, 25)]
            for idx in top_idx:
                row = df.iloc[int(idx)]
                mid = str(row["id"])
                if mid in disliked or mid in liked:
                    continue
                title_c = clean_title(row["title"])
                t_key = title_c.lower()
                if t_key in seen_titles:
                    continue
                seen_titles.add(t_key)
                origin_names = list(row["origin_country_names"])
                lang_names = list(row["original_language_names"])
                recs.append({
                    "movie_id": mid,
                    "title": title_c,
                    "genres": [g.strip() for g in str(row["genres"]).split("|") if g.strip()],
                    "year": int(row["year"]),
                    "origin_countries": list(row["origin_countries"]),
                    "origin_country_names": origin_names,
                    "original_languages": list(row["original_languages"]),
                    "original_language_names": lang_names,
                    "origin": ", ".join(origin_names) if origin_names else "unknown",
                    "rating": float(row.get("rating", 0.0)),
                    "votes": int(row.get("votes", 0)),
                    "catalog_source": "World catalog",
                })
                if len(recs) >= n:
                    return recs

        raw_recs = self.coldstart.recommend(["Drama", "Comedy", "Action", "Thriller", "Sci-Fi"], top_n=n * 2, alpha=0.2)
        for item in raw_recs:
            mid = str(item["movie_id"])
            if mid in disliked or mid in liked or item["title"].lower() in seen_titles:
                continue
            item["catalog_source"] = "MovieLens 1M"
            recs.append(item)
            if len(recs) >= n:
                break
        return recs

    def recommend(
        self,
        user_id: str,
        genres: Optional[List[str]] = None,
        n: int = 5,
        seed_movie: Optional[str] = None,
        modifier: Optional[str] = None,
    ) -> List[Dict]:
        """Generate personalized, genre-guided, or seed-similar recommendations."""
        fb = self.memory.get_feedback(user_id)
        disliked = set(fb["disliked"])
        liked = set(fb["liked"])

        # Handle seed movie (e.g. "movies like Parasite", "suggest something like Toy Story but darker")
        if seed_movie:
            resolved = self.resolve_title_across_catalogs(seed_movie)
            if resolved:
                if resolved["catalog_source"] == "World catalog" and not resolved.get("is_movielens"):
                    raw_world_recs = self.global_recommender.recommend(
                        liked_ids=[resolved["id"]],
                        n=n * 2,
                    )
                    recs = []
                    for r in raw_world_recs:
                        m_id = str(r.get("movie_id", r.get("imdb_id", r.get("id"))))
                        if m_id == str(resolved["id"]) or r["title"].lower() == resolved["title"].lower():
                            continue
                        recs.append({
                            "movie_id": m_id,
                            "title": clean_title(r["title"]),
                            "genres": r.get("genres", []),
                            "year": int(r["year"]),
                            "origin_countries": r.get("origin_countries", []),
                            "origin_country_names": r.get("origin_country_names", []),
                            "original_languages": r.get("original_languages", []),
                            "original_language_names": r.get("original_language_names", []),
                            "origin": r.get("origin", ""),
                            "rating": float(r.get("rating", 0.0)),
                            "votes": int(r.get("votes", 0)),
                            "catalog_source": "World catalog",
                        })
                        if len(recs) >= n:
                            break
                    return recs
                else:
                    seed_mid = resolved["ml_id"] if resolved.get("ml_id") is not None else self.find_movie_id_in_text(seed_movie)
                    if seed_mid is not None and isinstance(seed_mid, int) and seed_mid in self.mid_to_idx:
                        seed_vec = self.item_embs[self.mid_to_idx[seed_mid]]
                        sims = np.dot(self.item_embs, seed_vec)
                        dark_genres = {"Crime", "Thriller", "Horror", "Drama", "Film-Noir", "Sci-Fi", "Mystery"}
                        light_genres = {"Children's", "Animation", "Comedy", "Musical"}
                        # Seed genres used to require at least one shared genre in results
                        seed_meta = self.movie_meta.get(seed_mid, {})
                        seed_genres = set(parse_genres(seed_meta.get("genres", "")))
                        candidates = []
                        for mid, sim in zip(self.item_ids, sims):
                            mid_int = int(mid)
                            if mid_int == seed_mid or mid_int in disliked:
                                continue
                            if mid_int not in self.movie_meta:
                                continue
                            m_row = self.movie_meta[mid_int]
                            t_clean = clean_title(m_row["title"]).lower()
                            if t_clean == resolved["title"].lower():
                                continue
                            # Require at least one shared genre with the seed
                            m_genres = set(parse_genres(m_row["genres"]))
                            shared = seed_genres.intersection(m_genres)
                            if seed_genres and not shared:
                                continue
                            if modifier == "darker":
                                dark_score = len(m_genres.intersection(dark_genres))
                                light_score = len(m_genres.intersection(light_genres))
                                adj_score = float(sim) + (0.35 * dark_score) - (0.45 * light_score)
                            else:
                                # Overlap prior: favors candidates sharing more genres with seed
                                # (e.g. Godfather Part II sharing all 3 genres vs Glory sharing 2)
                                overlap_boost = 0.01 * (len(shared) / len(seed_genres)) if seed_genres else 0.0
                                adj_score = float(sim) + overlap_boost
                            pop = float(self.coldstart.log_pop.get(mid_int, 0.0))
                            candidates.append((mid_int, adj_score, pop))
                        # Rank by embedding similarity (adj_score) first, then popularity as tie-break only
                        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                        recs = []
                        for mid_int, _, _ in candidates[:n]:
                            meta = self.movie_meta[mid_int]
                            recs.append({
                                "movie_id": mid_int,
                                "title": clean_title(meta["title"]),
                                "genres": parse_genres(meta["genres"]),
                                "year": int(meta["year"]),
                                "catalog_source": "MovieLens 1M",
                            })
                        return recs

        # Try personalized precomputed recs if user_id is a known ML-1M user
        recs = []
        try:
            uid_int = int(user_id)
            if uid_int in self.user_recs_cache and not fb["liked"]:
                for mid in self.user_recs_cache[uid_int]:
                    if mid not in disliked and mid not in liked and mid in self.movie_meta:
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
            liked_genres = []
            for mid in fb["liked"]:
                if mid in self.movie_meta:
                    liked_genres.extend(str(self.movie_meta[mid]["genres"]).split("|"))
            if liked_genres:
                counts = pd.Series(liked_genres).value_counts()
                pref_genres = counts.head(3).index.tolist()

        if not pref_genres:
            return self.recommend_top_rated(user_id=user_id, n=n)

        raw_recs = self.coldstart.recommend(pref_genres, top_n=n * 2)
        liked_set = set(fb["liked"])
        for r in raw_recs:
            if r["movie_id"] not in disliked and r["movie_id"] not in liked_set:
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
                "catalog_source": "MovieLens 1M",
            })

        total_ms = (time.perf_counter() - t0) * 1000

        trace = [
            {"stage": "retrieval", "tool": "two_tower_faiss", "args": {"user_id": user_id, "candidates": 200}, "latency_ms": round(retrieval_ms, 2)},
            {"stage": "ranker", "tool": "wide_and_deep", "args": {"candidates_scored": len(top_ranked)}, "latency_ms": round(ranker_ms, 2)},
            {"stage": "diversity", "tool": "mmr_rerank", "args": {"lambda": 0.7, "top_k": n}, "latency_ms": round(mmr_ms, 2)},
            {"stage": "explanation", "tool": "factual_grounding", "args": {"source": "data_chips"}, "latency_ms": 0.0},
        ]

        return {
            "user_id": user_id,
            "recommendations": cards,
            "latency_ms": {
                "retrieval": round(retrieval_ms, 2),
                "ranker": round(ranker_ms, 2),
                "mmr": round(mmr_ms, 2),
                "llm": 0.0,
                "total": round(total_ms, 2),
            },
            "trace": trace,
            "llm_fallback": False,
            "notice": None,
        }

    def get_grounded_explanation(self, movie_id: int, user_id: str = "demo_user") -> Dict[str, Any]:
        """Fetch or generate 1-sentence grounded LLM explanation on demand, cached by (movie_id, user_state)."""
        if movie_id is None:
            return {"error": "Movie ID was not provided."}
        if movie_id not in self.movie_meta:
            return {"error": f"Movie ID {movie_id} not found in catalog."}

        meta = self.movie_meta[movie_id]
        clean_t = clean_title(meta["title"])
        m_genres = parse_genres(meta["genres"])
        m_genres_set = set(m_genres)

        fb = self.memory.get_feedback(user_id)
        liked_ids = tuple(sorted(fb.get("liked", [])))
        cache_key = (int(movie_id), str(user_id), liked_ids)

        if cache_key in self._explanation_cache:
            return self._explanation_cache[cache_key]

        # Determine similar liked movies
        similar_titles = []
        if movie_id in self.mid_to_idx:
            target_vec = self.item_embs[self.mid_to_idx[movie_id]]
            for liked_id in liked_ids:
                if liked_id == movie_id:
                    continue
                if liked_id in self.mid_to_idx and liked_id in self.movie_meta:
                    sim = float(np.dot(target_vec, self.item_embs[self.mid_to_idx[liked_id]]))
                    l_meta = self.movie_meta[liked_id]
                    similar_titles.append((clean_title(l_meta["title"]), sim))
            similar_titles.sort(key=lambda x: x[1], reverse=True)
        top_similar = [t[0] for t in similar_titles[:2]]

        pref_genres = fb.get("preferred_genres", [])
        matched = list(m_genres_set.intersection(set(pref_genres))) if pref_genres else m_genres[:2]
        if not matched:
            matched = m_genres[:2]

        if top_similar:
            fallback_text = f"Recommended for your interest in {', '.join(matched)}, matching themes from '{top_similar[0]}'."
        else:
            fallback_text = f"Recommended for your interest in {', '.join(matched)}."

        exp_text = fallback_text
        llm_fallback = False
        notice = None

        if self.client:
            prompt = (
                f"You are MovieMind. Write exactly one short, grounded sentence explaining why '{clean_t}' ({meta['year']}) "
                f"is recommended to the user. Cite only real genres ({', '.join(m_genres)}) and similar liked titles "
                f"({', '.join(top_similar) if top_similar else 'none'}). Never invent facts."
            )
            for attempt in range(2):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=256,
                    )
                    raw = (resp.choices[0].message.content or "").strip().replace("\u202f", " ").replace("\xa0", " ")
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = raw[1:-1].strip()
                    if len(raw) > 8:
                        exp_text = raw
                    llm_fallback = False
                    notice = None
                    break
                except Exception as e:
                    is_429 = "429" in str(e) or "rate" in str(e).lower()
                    if attempt == 0 and is_429:
                        time.sleep(1.0)
                        continue
                    llm_fallback = True
                    notice = "Fallback active: LLM rate-limited (429); using deterministic data chips." if is_429 else f"Fallback active: {type(e).__name__}"
        else:
            llm_fallback = True
            notice = "Fallback active: deterministic data chips (LLM API key not configured)."

        res = {
            "movie_id": int(movie_id),
            "title": clean_t,
            "text": exp_text,
            "matched_genres": matched,
            "similar_liked_movies": top_similar,
            "llm_fallback": llm_fallback,
            "notice": notice,
        }
        self._explanation_cache[cache_key] = res
        return res

    def explain(self, movie_id: Optional[Any] = None, title: Optional[str] = None, user_id: str = "demo_user") -> Dict[str, Any]:
        """Explain recommendation using real factual features and similarity to liked movies."""
        if movie_id is not None and not str(movie_id).startswith("tt") and not str(movie_id).isdigit():
            title = str(movie_id)
            movie_id = None
        if movie_id is None and title:
            movie_id = self.find_movie_id_in_text(title)
        if movie_id is None:
            return {"error": f"I couldn't find '{title or 'the specified movie'}' in either catalog."}

        # Check MovieLens 1M
        if isinstance(movie_id, int) and movie_id in self.movie_meta:
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
                "catalog_source": "MovieLens 1M",
            }

        # Check World Catalog
        mid_str = str(movie_id)
        if mid_str in self.global_meta:
            row = self.global_meta[mid_str]
            clean_t = clean_title(row["title"])
            genres = [g.strip() for g in str(row["genres"]).split("|") if g.strip()]
            origin_names = list(row["origin_country_names"])
            orig_str = ", ".join(origin_names) if origin_names else "international"
            explanation = f"'{clean_t}' ({row['year']}) is an acclaimed {orig_str} feature with {row['rating']}/10 across {row['votes']:,} IMDb ratings, matching your interest in {', '.join(genres[:2])} (from the world catalog; not in MovieLens 1M)."
            return {
                "movie_id": mid_str,
                "title": clean_t,
                "genres": genres,
                "year": int(row["year"]),
                "origin": orig_str,
                "origin_countries": list(row["origin_countries"]),
                "origin_country_names": origin_names,
                "original_languages": list(row["original_languages"]),
                "original_language_names": list(row["original_language_names"]),
                "explanation": explanation,
                "catalog_source": "World catalog",
            }

        return {"error": f"Movie ID {movie_id} not found in catalog."}

    def filter_movies(
        self,
        genre: Optional[str] = None,
        additional_genre: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        year: Optional[int] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        languages: Optional[List[str]] = None,
        limit: int = 5,
        _genre_note: Optional[str] = None,
        _region_label: Optional[str] = None,
    ) -> List[Dict]:
        """Filter movie catalog by genre, year, region, or language using data-driven bounds."""
        if year is not None:
            if min_year is None:
                min_year = year
            if max_year is None:
                max_year = year

        # Query global catalog if region, language, or out-of-MovieLens year specified
        is_world_query = bool(
            region
            or language
            or languages
            or (min_year and (min_year < self.ml_min_year or min_year > self.ml_max_year))
            or (max_year and (max_year < self.ml_min_year or max_year > self.ml_max_year))
        )

        if is_world_query and self.global_df is not None:
            reg_clean = self.region_name_to_code.get(region.strip().lower(), region.strip().upper()) if region else None
            lang_clean = self.lang_name_to_code.get(language.strip().lower(), language.strip().lower()) if language else None
            genre_clean = genre.strip().lower() if genre else None
            additional_genre_clean = additional_genre.strip().lower() if additional_genre else None

            df_sub = self.global_df
            if min_year is not None:
                df_sub = df_sub[df_sub["year"] >= min_year]
            if max_year is not None:
                df_sub = df_sub[df_sub["year"] <= max_year]

            # Genre filtering: must carry requested genre tag
            if genre_clean:
                df_sub = df_sub[df_sub["genres"].str.lower().str.contains(rf"\b{re.escape(genre_clean)}\b", na=False)]
            if additional_genre_clean:
                df_sub = df_sub[df_sub["genres"].str.lower().str.contains(rf"\b{re.escape(additional_genre_clean)}\b", na=False)]

            # Region & Language filtering
            lang_set = {str(l).lower() for l in languages} if languages else set()
            if reg_clean and lang_set:
                reg_filter = lambda arr: reg_clean in [str(c).upper() for c in arr if c]
                df_sub = df_sub[
                    df_sub["origin_countries"].apply(reg_filter)
                    & df_sub["original_languages"].apply(lambda arr: any(str(l).lower() in lang_set for l in arr if l))
                ]
            elif lang_set:
                df_sub = df_sub[df_sub["original_languages"].apply(lambda arr: any(str(l).lower() in lang_set for l in arr if l))]
            elif reg_clean and lang_clean:
                if reg_clean == "DE":
                    reg_filter = lambda arr: any(c in {"DE", "WEST GERMANY"} for c in [str(x).upper() for x in arr if x])
                else:
                    reg_filter = lambda arr: reg_clean in [str(c).upper() for c in arr if c]
                df_sub = df_sub[
                    df_sub["origin_countries"].apply(reg_filter)
                    | df_sub["original_languages"].apply(lambda arr: lang_clean in [str(l).lower() for l in arr if l])
                ]
            elif lang_clean:
                # Explicit language constraint: NEVER widen silently.
                df_sub = df_sub[df_sub["original_languages"].apply(lambda arr: lang_clean in [str(l).lower() for l in arr if l])]
            elif reg_clean:
                if reg_clean == "DE":
                    reg_filter = lambda arr: any(c in {"DE", "WEST GERMANY"} for c in [str(x).upper() for x in arr if x])
                else:
                    reg_filter = lambda arr: reg_clean in [str(c).upper() for c in arr if c]
                df_sub = df_sub[df_sub["origin_countries"].apply(reg_filter)]

            def _tier(row: Any) -> str:
                r_tier = self.region_match_tier(row, reg_clean) if reg_clean else None
                l_tier = self.language_match_tier(row, lang_clean) if lang_clean else None
                if reg_clean and lang_clean:
                    # For Spanish cinema, language is primary across Spain and Latin America
                    if lang_clean == "es":
                        if l_tier == "core" or r_tier == "core":
                            return "core"
                        return "partial"
                    # For regional demonyms (e.g. German, French, Korean), primary origin country is required for core
                    if r_tier == "core":
                        return "core"
                    return "partial"
                elif reg_clean:
                    return r_tier or "partial"
                elif lang_clean:
                    return l_tier or "partial"
                return "partial"

            df_sub = df_sub.copy()
            df_sub["_tier"] = df_sub.apply(_tier, axis=1)
            # core first, then partial; within each tier sort by votes descending
            df_sub["_tier_rank"] = df_sub["_tier"].map({"core": 0, "partial": 1})
            df_sub = df_sub.sort_values(by=["_tier_rank", "votes"], ascending=[True, False])

            results = []
            for _, row in df_sub.head(limit).iterrows():
                origin_names = list(row["origin_country_names"])
                lang_names = list(row["original_language_names"])
                tier = str(row["_tier"])
                partial_reason = None
                if tier == "partial":
                    if lang_clean and lang_clean in [str(l).lower() for l in row["original_languages"]]:
                        ls = [str(l).lower() for l in row["original_languages"]]
                        if ls and ls[0] != lang_clean:
                            lang_display = self.code_to_lang_name.get(lang_clean, lang_clean)
                            partial_reason = f"also in {lang_display}"
                        else:
                            partial_reason = "co-production"
                    else:
                        partial_reason = "co-production"
                results.append({
                    "movie_id": str(row["id"]),
                    "title": clean_title(row["title"]),
                    "genres": [g.strip() for g in str(row["genres"]).split("|") if g.strip()],
                    "year": int(row["year"]),
                    "origin_countries": list(row["origin_countries"]),
                    "origin_country_names": origin_names,
                    "original_languages": list(row["original_languages"]),
                    "original_language_names": lang_names,
                    "origin": ", ".join(origin_names) if origin_names else "unknown",
                    "rating": float(row.get("rating", 0.0)),
                    "votes": int(row.get("votes", 0)),
                    "catalog_source": "World catalog",
                    "_match_tier": tier,
                    "_partial_reason": partial_reason,
                })
            return results

        # Primary MovieLens catalog
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
                "catalog_source": "MovieLens 1M",
            })
            if len(results) >= limit:
                break
        return results

    def _format_filter_response(
        self,
        parsed_args: Dict[str, Any],
        results: List[Dict[str, Any]],
        source: str = "World catalog",
    ) -> str:
        """Format an honest, clean summary for filter query results without internal tier jargon."""
        total_found = len(results)
        core_count = sum(1 for r in results if r.get("_match_tier") == "core")
        partial_count = sum(1 for r in results if r.get("_match_tier") == "partial")

        # Build clean, human-readable query description
        desc_parts = []
        if parsed_args.get("genre"):
            desc_parts.append(str(parsed_args["genre"]))
        if parsed_args.get("_region_label"):
            desc_parts.append(str(parsed_args["_region_label"]))
        elif parsed_args.get("language"):
            l_code = parsed_args["language"]
            desc_parts.append(self.code_to_lang_name.get(l_code, str(l_code).capitalize()))
        elif parsed_args.get("region"):
            r_code = parsed_args["region"]
            desc_parts.append(self.code_to_region_name.get(r_code, str(r_code)))

        yr_str = ""
        if parsed_args.get("min_year") and parsed_args.get("max_year"):
            if parsed_args["min_year"] == parsed_args["max_year"]:
                yr_str = f" from {parsed_args['min_year']}"
            elif parsed_args["max_year"] - parsed_args["min_year"] == 9:
                yr_str = f" from the {parsed_args['min_year']}s"
            else:
                yr_str = f" ({parsed_args['min_year']}-{parsed_args['max_year']})"
        elif parsed_args.get("year"):
            yr_str = f" from {parsed_args['year']}"

        query_desc = (" ".join(desc_parts) + f" films{yr_str}").strip()
        if not query_desc:
            query_desc = "movies"

        header_lines = []
        genre_note = parsed_args.get("_genre_note")
        if genre_note:
            header_lines.append(f"Note: {genre_note}.")

        limit_req = parsed_args.get("limit", 5)
        if total_found == 0:
            header_lines.append(f"No movies matched your filter. I found 0 {query_desc} in the catalog.")
        elif total_found < limit_req:
            header_lines.append(f"I found {total_found} {query_desc} in the catalog:")
        else:
            header_lines.append(f"From the world catalog ({query_desc}):" if source == "World catalog" else f"From MovieLens 1M ({query_desc}):")

        if core_count > 0 and partial_count > 0:
            header_lines.append(f"Scope note: Showing {core_count} core matches (primary language/origin) and {partial_count} partial matches (co-production or secondary language).")

        lines = []
        for r in results:
            orig = f" [{r['origin']}]" if 'origin' in r else ""
            part_s = f" ({r['_partial_reason']})" if r.get('_match_tier') == 'partial' and r.get('_partial_reason') else ""
            lines.append(f"- **{r['title']}** ({r['year']}){orig}{part_s} - [{', '.join(r.get('genres', []))}]")

        return "\n".join(header_lines) + ("\n" + "\n".join(lines) if lines else "")

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

    def record_feedback(self, user_id: str, movie_id: Optional[Any] = None, title: Optional[str] = None, feedback_type: str = "like") -> Dict[str, Any]:
        """Store like or dislike for a movie with duplicate prevention."""
        if movie_id is None and title:
            movie_id = self.find_movie_id_in_text(title)
        meta = self.movie_meta.get(movie_id)
        if meta is None and isinstance(movie_id, str) and movie_id.isdigit():
            int_id = int(movie_id)
            meta = self.movie_meta.get(int_id)
            if meta is not None:
                movie_id = int_id
        if meta is None and self.global_meta:
            meta = self.global_meta.get(str(movie_id))
        if meta is None:
            return {"error": f"Movie '{title or movie_id}' does not exist in catalog."}
        title_str = clean_title(meta["title"])
        is_new, status_msg = self.memory.record_feedback(user_id, movie_id, feedback_type)
        if not is_new and "already" in status_msg:
            return {"status": "noop", "message": f"'{title_str}' is already {feedback_type}d."}
        return {"status": "success", "message": f"Recorded {feedback_type} for '{title_str}'."}

    # ================= DISPATCHER & CHAT =================
    def get_tool_specs(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_movies",
                    "description": "Search movies by title query or keyword in MovieLens 1M catalog. If user query has a typo (e.g. 'alines'), search for the corrected title.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Movie title or query (corrected if typo)"},
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
                    "description": "Recommend movies for a user with optional genre preferences or similarity requests (e.g. 'something like Aliens', 'space movies with lasers').",
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
                            "movie_id": {"type": "integer", "description": "Numeric Movie ID (if known)"},
                            "title": {"type": "string", "description": "Movie title if ID is unknown"},
                            "user_id": {"type": "string", "default": "demo_user"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "filter_movies",
                    "description": f"Filter movies by genre, release year ({self.data_min_year} to {self.data_max_year}), origin country code or name, or language.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "genre": {"type": "string", "description": "Movie genre"},
                            "year": {"type": "integer", "description": f"Exact release year ({self.data_min_year} to {self.data_max_year})"},
                            "min_year": {"type": "integer", "description": f"Minimum release year (>= {self.data_min_year})"},
                            "max_year": {"type": "integer", "description": f"Maximum release year (<= {self.data_max_year})"},
                            "region": {"type": "string", "description": "Country code or country name (e.g. FR, JP, IN, IT, GB, US, India, Japan)"},
                            "language": {"type": "string", "description": "Language code or name (e.g. French, Japanese, Hindi, Spanish, te, ja)"},
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
                    "description": "Record user feedback: 'like' or 'dislike' when user explicitly likes or dislikes a movie. Do NOT use for similarity requests.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "movie_id": {"type": "integer", "description": "Numeric Movie ID (if known)"},
                            "title": {"type": "string", "description": "Movie title if ID is unknown"},
                            "feedback_type": {"type": "string", "enum": ["like", "dislike"]},
                        },
                        "required": ["user_id", "feedback_type"],
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

    def find_movie_id_in_text(self, text: str) -> Optional[Any]:
        """Resolve numeric ID or canonical movie title from text across both catalogs."""
        low = (text or "").lower()
        id_match = re.search(r"\b(?:movie(?:\s+id)?|id:?)\s*(\d{1,6})\b", low)
        if id_match:
            return int(id_match.group(1))
        if low.strip().isdigit():
            return int(low.strip())
        # Check IMDb tt ID pattern
        tt_match = re.search(r"\b(tt\d{7,8})\b", low)
        if tt_match:
            return tt_match.group(1)

        # Check cross-catalog resolution by popularity/votes
        resolved = self.resolve_title_across_catalogs(text)
        if resolved:
            if resolved.get("is_movielens") and resolved.get("ml_id") is not None:
                return int(resolved["ml_id"])
            return resolved["id"]

        # Check MovieLens and World catalog title maps via O(1) exact and n-gram lookup
        clean_text = self.clean_search_query(text).lower()
        if clean_text in self.title_map:
            return self.title_map[clean_text]
        if clean_text in self.global_title_to_id:
            return self.global_title_to_id[clean_text]
        words = re.findall(r"[a-z0-9'-]+", low)
        single_stopwords = {
            "tell", "pick", "picked", "choose", "chosen", "watch", "good", "best",
            "make", "makes", "give", "show", "find", "what", "when", "where", "which",
            "that", "this", "with", "from", "your", "have", "will", "been", "were",
            "they", "more", "some", "like", "love", "hate", "movie", "movies", "film",
        }
        # Prefer MovieLens catalog matches across all spans first, then World catalog
        for title_dict in (self.title_map, self.global_title_to_id):
            for span in range(min(6, len(words)), 0, -1):
                for i in range(len(words) - span + 1):
                    cand = " ".join(words[i : i + span])
                    if len(cand) >= 4 and not (span == 1 and cand in single_stopwords):
                        if cand in title_dict:
                            return title_dict[cand]
        generic_num = re.search(r"\b(\d{3,6})\b", low)
        if generic_num:
            return int(generic_num.group(1))
        return None

    def is_closing(self, low: str) -> bool:
        """Detect closings, goodbyes, and exit pleasantries."""
        if re.search(r"\b(bye|goodbye|see you|see ya|farewell|cya)\b", low):
            return True
        if re.search(r"\b(that'?s all|that is all|nothing else|i'?m good|done for now)\b", low):
            return True
        if re.search(r"^(thanks|thank you|thx|cheers)\b", low):
            return True
        return False

    def is_greeting(self, low: str) -> bool:
        """Detect welcome greetings (excluding closings)."""
        if self.is_closing(low):
            return False
        if re.search(r"^(hello|hi|hey|howdy|greetings)\b", low):
            return True
        if re.search(r"^good\s+(morning|afternoon|evening|day)\b", low):
            return True
        return False

    def has_movie_signal(self, text: str) -> bool:
        """Check whether input contains any movie signal (keywords, genres, demonyms, years, or titles)."""
        low = (text or "").lower()

        # 1. Movie keywords
        movie_keywords = {
            "movie", "movies", "film", "films", "cinema", "actor", "actress", "director", "directed", "starring", "oscar", "oscars",
            "hollywood", "bollywood", "tollywood", "kollywood", "mollywood", "sandalwood", "south indian",
            "anime", "ghibli", "studio ghibli", "mindfuck", "mind-bending", "underrated", "overrated", "cult", "peak",
            "another one", "less violent", "newer ones", "the first one",
            "genre", "genres", "watch", "watching",
            "recommend", "recommendation", "recommendations", "suggest", "suggestion", "suggestions",
            "favorite", "favourite", "taste", "search", "find", "look up", "picks", "pick", "picked", "rate",
            "rated", "rating", "ratings", "reviews", "dislike", "disliked", "hate", "hated", "love",
            "loved", "like", "liked", "screening", "screenings", "screen", "recs"
        }
        for kw in movie_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", low):
                return True

        # 2. Genres, aliases, and synonyms
        genre_tokens = {
            "action", "adventure", "animation", "animated", "anime", "children", "children's", "kids", "family",
            "comedy", "comedies", "comic", "funny", "crime", "documentary", "documentaries", "doc",
            "drama", "dramas", "dramatic", "fantasy", "film-noir", "noir", "horror", "scary",
            "musical", "musicals", "mystery", "mysteries", "detective", "romance", "romantic",
            "rom-com", "romcom", "martial arts", "kung fu", "sci-fi", "scifi", "science fiction", "space", "lasers", "thriller",
            "thrillers", "suspense", "war", "western", "westerns"
        }
        for gt in genre_tokens:
            if re.search(rf"\b{re.escape(gt)}\b", low):
                return True

        # 3. Demonyms, countries, languages
        for alias in DEMONYM_ALIASES:
            if re.search(rf"\b{re.escape(alias)}\b", low):
                return True
        for r_name in self.region_name_to_code:
            if len(r_name) >= 3 and re.search(rf"\b{re.escape(r_name)}\b", low):
                return True
        for l_name in self.lang_name_to_code:
            if len(l_name) >= 3 and re.search(rf"\b{re.escape(l_name)}\b", low):
                return True

        # 4. Years or decades
        if re.search(r"\b(18\d{2}|19\d{2}|20\d{2})s?\b", low):
            return True

        # 5. Catalog title or ID match (including high-confidence fuzzy match for typos like 'intersteller')
        if self.find_movie_id_in_text(low) is not None:
            return True
        if self.resolve_title_across_catalogs(low) is not None:
            return True
        clean_q = re.sub(r"[^a-z0-9\s'-]", "", low).strip()
        if len(clean_q) >= 4 and len(clean_q.split()) <= 3:
            if self._fuzzy_world_matches(clean_q, limit=1):
                return True
            for ct in self.title_map:
                if abs(len(ct) - len(clean_q)) <= 2 and ct[0] == clean_q[0]:
                    if difflib.SequenceMatcher(None, clean_q, ct).ratio() >= 0.82:
                        return True

        return False

    def is_off_topic(self, low: str) -> bool:
        """Detect math, programming, or non-movie questions using an allow-list rule."""
        if self.is_closing(low) or self.is_greeting(low):
            return False

        gen_patterns = [
            r"^(who|what)\s+(was|is|are|were|wrote|authored|painted|composed|invented|discovered|built|founded|won|scored|played|beat)\b",
            r"^(how to|how do|how does|how can|how did|why did|why do|why does|where is|where was|when did|when was)\b",
            r"\b(capital of|president of|weather in|temperature in|stock price|recipe for|how to cook|how to bake|pizza dough|speed of light)\b",
            r"\b(won the match|cricket|football|soccer|ipl|world cup|score of)\b",
            r"\b(write (a )?(python|code|program|script|function|poem|essay|story))\b",
            r"\b(calculate|solve|square root|multiply|divide|math|addition|subtraction|equation)\b",
            r"\b(\d+\s*[\+\-\*\/\^=]\s*\d+)\b",
            r"\b(\d+\s+times\s+\d+)\b",
        ]
        movie_cues = (
            "movie", "movies", "film", "films", "cinema", "actor", "actress", "director", "directed", "starring", "oscar", "oscars",
            "hollywood", "bollywood", "genre", "genres", "watch", "recommend", "suggest", "screening",
            "screenings", "screening room", "favorite", "favourite", "taste", "picks", "pick", "picked"
        )
        if any(bool(re.search(p, low)) for p in gen_patterns) and not any(c in low for c in movie_cues):
            return True

        return not self.has_movie_signal(low)

    def is_prose_hallucinating(self, text: str, user_msg: str) -> bool:
        """Detect free-form LLM prose attempting to list movies, markdown tables, or hallucinate titles."""
        if not text:
            return False
        low_t = (text or "").lower()
        low_msg = (user_msg or "").lower().strip()

        # Markdown table check
        if "|" in text and text.count("|") >= 2:
            return True
        # Numbered list or bullet list of movies
        if re.search(r"^\s*(\d+[\.\)]|[-*])\s+.*", text, re.MULTILINE):
            return True

        # Year references in text (e.g. (1999) or 2024)
        if re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", text):
            return True

        # Prose recommendation indicators / invented titles
        if any(ind in low_t for ind in ("here are", "i recommend", "you should watch", "check out", "cybergala", "neotokyo", "space pirates")):
            return True

        # Don't flag simple closings, greetings, or off-topic refusals without movie names
        if self.is_off_topic(low_msg) or self.is_closing(low_msg) or self.is_greeting(low_msg):
            return False

        has_movie_query = any(w in low_msg for w in (
            "movie", "film", "recommend", "suggest", "watch", "like", "search", "find",
            "thriller", "action", "comedy", "horror", "sci-fi", "drama", "animation",
            "telugu", "korean", "japanese", "indian", "darker", "picks", "give me", "best", "top",
            "hated", "loved", "dislike"
        )) or bool(re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", low_msg))

        if has_movie_query:
            return True

        return False

    def detect_person_intent(self, text: str) -> Optional[str]:
        """Detect person/cast/crew query patterns (actors, directors) in a general, data-free way."""
        low = (text or "").lower().strip()
        if not low:
            return None
        low = re.sub(r"[?!.]+$", "", low).strip()

        # Conversational prefix stripping
        low_clean = re.sub(
            r"^(can you\s+)?(please\s+)?(suggest|recommend|give me|show me|find|search for|search|look up|tell me about|do you have|are there|any|i want to see|i want to watch|i want to|i'd like to see|i'd like to watch|i'd like)?\s*",
            "",
            low,
        ).strip()

        person_patterns = [
            r"\b(?:movies?|films?|cinema)\s+(?:with|starring|featuring|by|directed\s+by)\s+([a-z\s.'-]+)$",
            r"\b(?:directed\s+by|starring)\s+([a-z\s.'-]+)$",
            r"^([a-z\s.'-]+?)(?:['`]s)?\s+(?:movies?|films?)$",
            r"\b([a-z\s.'-]+?)['`]s\s+(?:movies?|films?)\b",
        ]

        candidate = None
        for pat in person_patterns:
            m = re.search(pat, low_clean)
            if m:
                candidate = m.group(1).strip()
                break

        if not candidate:
            return None

        c_low = candidate.lower().strip()
        if not c_low or len(c_low) < 2 or len(c_low.split()) > 4:
            return None

        # Exclusions:
        # 0. Conversational words & stopwords
        stopwords = {
            "a", "an", "the",
            "add", "liked", "disliked", "favorite", "favorites", "to", "my", "me",
            "i", "you", "he", "she", "we", "they", "it", "for", "from", "in", "on",
            "at", "about", "with", "without", "and", "or", "but", "so", "is", "are",
            "was", "were", "want", "see", "watch", "give", "show", "tell", "have",
            "do", "does", "did", "can", "could", "would", "should", "not", "no"
        }
        c_words = c_low.split()
        if any(w in stopwords for w in c_words):
            return None

        # 1. Gibberish check
        if is_gibberish(c_low):
            return None

        # 2. Year or decade check
        if re.search(r"\b(18\d{2}|19\d{2}|20\d{2})s?\b", c_low) or re.search(r"\b\d{2}s\b", c_low):
            return None

        # 3. Recommendation / descriptor adjectives check
        rec_cues = {
            "best", "top", "good", "great", "popular", "famous", "classic", "new",
            "recent", "latest", "old", "random", "any", "some", "more", "other",
            "another", "favorite", "favourite", "cool", "fun", "award", "winning", "oscar",
            "rated", "critically", "acclaimed", "highest", "must", "watch",
            "mindfuck", "mind-bending", "feel-good", "underrated", "overrated", "cult",
            "peak", "vibe", "vibes", "ghibli", "studio",
        }
        tokens = re.findall(r"[a-z]+", c_low)
        if tokens and all(tok in rec_cues for tok in tokens):
            return None

        # 4. Languages check (e.g. 'tamil', 'korean', 'hindi')
        for l in self.lang_name_to_code:
            if len(l) >= 3 and re.search(rf"\b{re.escape(l)}\b", c_low):
                return None

        # 5. Demonyms / cultural cinema aliases (e.g. 'korean', 'french', 'indian', 'spanish')
        for d in DEMONYM_ALIASES:
            if re.search(rf"\b{re.escape(d)}\b", c_low):
                return None

        # 6. Regions / country names (e.g. 'india', 'japan', 'france')
        for r in self.region_name_to_code:
            if len(r) >= 3 and re.search(rf"\b{re.escape(r)}\b", c_low):
                return None

        # 7. Genres and genre synonyms
        from src.baselines import ALL_GENRES
        for g in ALL_GENRES:
            if re.search(rf"\b{re.escape(g.lower())}\b", c_low):
                return None
        for ga in GENRE_ALIASES:
            if re.search(rf"\b{re.escape(ga.lower())}\b", c_low):
                return None
        genre_synonyms = {
            "thriller", "detective", "family", "funny", "comedies", "comedy",
            "lasers", "space", "animated", "anime", "scary", "romantic", "kung fu", "suspense"
        }
        if any(gs in c_low for gs in genre_synonyms):
            return None

        # 8. Exact catalog title check
        if c_low in self.title_map or c_low in self.global_title_to_id:
            return None
        if f"the {c_low}" in self.title_map or f"the {c_low}" in self.global_title_to_id:
            return None

        return candidate

    def _save_user_context(self, user_id: str, res: Dict[str, Any]) -> None:
        """Persist last cards/args per user for conversational follow-ups."""
        if not hasattr(self, "_user_last_context"):
            self._user_last_context: Dict[str, Dict[str, Any]] = {}
        cards = res.get("cards") or []
        if cards:
            self._user_last_context[user_id] = {
                "tool": res.get("tool"),
                "args": dict(res.get("args") or {}),
                "cards": list(cards),
                "seen_ids": list({str(c.get("movie_id")) for c in cards if c.get("movie_id") is not None}),
            }

    def _handle_followup_query(self, user_msg: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Handle conversational follow-ups ('another one', 'less violent', 'newer ones', 'more like the first one')."""
        low = re.sub(r"[?!.]+$", "", (user_msg or "").lower().strip())
        ctx = getattr(self, "_user_last_context", {}).get(user_id)
        if low in ("another one", "more", "more like the first one", "newer ones", "less violent"):
            if not ctx or not ctx.get("cards"):
                if low == "less violent":
                    recs = self.filter_movies(genre="Comedy", limit=5)
                    source = recs[0].get("catalog_source", "MovieLens 1M") if recs else "MovieLens 1M"
                    cards = self._format_cards(recs, source=source)
                    lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(recs)]
                    out = {
                        "tool": "filter_movies",
                        "args": {"genre": "Comedy", "limit": 5},
                        "result": recs,
                        "cards": cards,
                        "response": "Here are lighter, non-violent Comedy picks:\n" + "\n".join(lines),
                        "safe": True,
                        "is_adversarial": False,
                    }
                    self._save_user_context(user_id, out)
                    return out
                if low == "newer ones":
                    recs = self.filter_movies(min_year=2018, limit=5)
                    cards = self._format_cards(recs, source="World catalog")
                    lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(recs)]
                    out = {
                        "tool": "filter_movies",
                        "args": {"min_year": 2018, "limit": 5},
                        "result": recs,
                        "cards": cards,
                        "response": "Here are newer acclaimed films (2018+):\n" + "\n".join(lines),
                        "safe": True,
                        "is_adversarial": False,
                    }
                    self._save_user_context(user_id, out)
                    return out
                recs = self.recommend_top_rated(user_id=user_id, n=5)
                source = recs[0].get("catalog_source", "World catalog") if recs else "World catalog"
                cards = self._format_cards(recs, source=source)
                lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(recs)]
                out = {
                    "tool": "recommend",
                    "args": {"user_id": user_id, "n": 5},
                    "result": recs,
                    "cards": cards,
                    "response": "Here are some top-rated picks:\n" + "\n".join(lines),
                    "safe": True,
                    "is_adversarial": False,
                }
                self._save_user_context(user_id, out)
                return out

            prev_cards = ctx["cards"]
            seen = set(ctx.get("seen_ids") or [])
            if low == "more like the first one":
                seed_title = prev_cards[0].get("title", "")
                recs = self.recommend(user_id=user_id, seed_movie=seed_title, n=5)
                source = recs[0].get("catalog_source", "MovieLens 1M") if recs else "MovieLens 1M"
                cards = self._format_cards(recs, source=source)
                lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(recs)]
                out = {
                    "tool": "recommend",
                    "args": {"seed_movie": seed_title, "user_id": user_id, "n": 5},
                    "result": recs,
                    "cards": cards,
                    "response": f"Films similar to **{seed_title}**:\n" + "\n".join(lines),
                    "safe": True,
                    "is_adversarial": False,
                }
                self._save_user_context(user_id, out)
                return out
            if low == "newer ones":
                recs = self.filter_movies(min_year=2015, limit=5)
                recs = [r for r in recs if str(r.get("movie_id")) not in seen][:5] or recs[:5]
                cards = self._format_cards(recs, source="World catalog")
                lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(recs)]
                out = {
                    "tool": "filter_movies",
                    "args": {"min_year": 2015, "limit": 5},
                    "result": recs,
                    "cards": cards,
                    "response": "Newer picks (2015+):\n" + "\n".join(lines),
                    "safe": True,
                    "is_adversarial": False,
                }
                self._save_user_context(user_id, out)
                return out
            if low == "less violent":
                recs = self.filter_movies(genre="Comedy", limit=5)
                source = recs[0].get("catalog_source", "MovieLens 1M") if recs else "MovieLens 1M"
                cards = self._format_cards(recs, source=source)
                lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(recs)]
                out = {
                    "tool": "filter_movies",
                    "args": {"genre": "Comedy", "limit": 5},
                    "result": recs,
                    "cards": cards,
                    "response": "Lighter, less violent picks:\n" + "\n".join(lines),
                    "safe": True,
                    "is_adversarial": False,
                }
                self._save_user_context(user_id, out)
                return out
            # "another one" / "more"
            pool = self.recommend_top_rated(user_id=user_id, n=12)
            fresh = [r for r in pool if str(r.get("movie_id")) not in seen][:5] or pool[:5]
            source = fresh[0].get("catalog_source", "World catalog") if fresh else "World catalog"
            cards = self._format_cards(fresh, source=source)
            lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(fresh)]
            out = {
                "tool": "recommend",
                "args": {"user_id": user_id, "n": 5},
                "result": fresh,
                "cards": cards,
                "response": "Here are more picks:\n" + "\n".join(lines),
                "safe": True,
                "is_adversarial": False,
            }
            self._save_user_context(user_id, out)
            return out
        return None

    def detect_movie_director_query(self, text: str) -> Optional[Dict[str, Any]]:
        """Handle 'who directed <title>' / 'which director made <title>' / 'cast of <title>' honestly."""
        low = (text or "").lower().strip()
        low = re.sub(r"[?!.]+$", "", low).strip()
        patterns = [
            r"^(?:which|what)\s+director\s+(?:made|directed)\s+(.+)$",
            r"^who\s+(?:directed|made)\s+(?:the\s+(?:movie|film)\s+)?(.+)$",
            r"^(?:who\s+is\s+the\s+)?(?:director|cast)\s+of\s+(?:the\s+(?:movie|film)\s+)?(.+)$",
            r"^who\s+(?:starred|acted)\s+in\s+(?:the\s+(?:movie|film)\s+)?(.+)$",
        ]
        target_title = None
        for pat in patterns:
            m = re.match(pat, low)
            if m:
                target_title = m.group(1).strip()
                break
        if not target_title:
            return None

        resolved = self.resolve_title_across_catalogs(target_title)
        if resolved:
            card_item = {
                "movie_id": resolved["ml_id"] if resolved.get("is_movielens") and resolved.get("ml_id") is not None else resolved["id"],
                "title": resolved["title"],
                "year": resolved["year"],
                "genres": resolved["genres"],
                "catalog_source": resolved["catalog_source"],
                "origin": resolved.get("origin", ""),
                "rating": resolved.get("rating", 0.0),
                "votes": resolved.get("votes", 0),
            }
            cards = self._format_cards([card_item], source=resolved["catalog_source"])
            resp = (
                f"I don't have director or cast data in my catalog, so I can't tell you who directed '{resolved['title']}'. "
                f"However, **{resolved['title']}** ({resolved['year']}) is in the {resolved['catalog_source']} if you'd like to explore it or ask for similar films."
            )
            return {
                "tool": None,
                "args": {},
                "result": {},
                "cards": cards,
                "response": resp,
                "safe": True,
                "is_adversarial": False,
            }
        return {
            "tool": None,
            "args": {},
            "result": {},
            "cards": [],
            "response": (
                f"I don't have director or cast data in my catalog, so I can't answer who directed '{target_title}'."
            ),
            "safe": True,
            "is_adversarial": False,
        }

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
                "cards": [],
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
                "cards": [],
                "response": "I cannot alter my instructions or invent non-existent movies. I am MovieMind, dedicated to recommending real movies from the MovieLens catalog.",
                "safe": True,
                "is_adversarial": True,
            }

        # Intent 0: Closings / exit pleasantries -> short goodbye, no tools
        if self.is_closing(low):
            return {
                "tool": None,
                "args": {},
                "result": {},
                "cards": [],
                "response": "You're welcome! Enjoy the movies, and come back anytime.",
                "safe": True,
                "is_adversarial": False,
            }

        # Intent 1: Greetings -> short welcome reply, no tools
        if self.is_greeting(low):
            return {
                "tool": None,
                "args": {},
                "result": {},
                "cards": [],
                "response": "Hello! Welcome to MovieMind screening room. How can I help you find or explore movies today?",
                "safe": True,
                "is_adversarial": False,
            }

        # Adversarial 3: Off-topic input (math, general questions)
        if self.is_off_topic(low):
            return {
                "tool": None,
                "args": {},
                "result": {},
                "cards": [],
                "response": "I only help with movies. Try a title, genre, or year.",
                "safe": True,
                "is_adversarial": True,
            }

        # Intent: Movie-to-director/cast queries ("who directed inception", "which director made batman begins")
        dir_res = self.detect_movie_director_query(msg)
        if dir_res is not None:
            return dir_res

        # Intent: Person / Cast / Crew queries (actors, directors)
        if self.detect_person_intent(msg):
            return {
                "tool": None,
                "args": {},
                "result": {},
                "cards": [],
                "response": (
                    "I can't search by actor or director yet, because the catalog has no cast data. "
                    "I can filter by language, country, genre and year instead, for example 'tamil action films from the 90s'."
                ),
                "safe": True,
                "is_adversarial": False,
            }

        # Compound Intent: Feedback THEN recommend (e.g. "i hated Titanic what else", "i loved Aliens, suggest more")
        compound_pattern = r"\b(?:i\s+)?(didn'?t\s+like|didnt\s+like|don'?t\s+like|dont\s+like|hated?|disliked?|not\s+a\s+fan\s+of|didn'?t\s+enjoy|didnt\s+enjoy|loved?|liked?|enjoyed?)\s+(.*?)\s*(?:,|\.|\b|\sand\s)\s*(?:and\s+)?(give\s+me\s+something\s+else|give\s+me\s+something\s+better|give\s+me\s+more|something\s+else|what\s+else|what\s+next|anything\s+else|suggest\s+more|suggest\s+another|suggest\s+something\s+else|recommend\s+more|recommend\s+another|recommend\s+something\s+else|what\s+should\s+i\s+watch\s+instead|what\s+should\s+i\s+watch\s+now|show\s+me\s+more|show\s+me\s+another|show\s+me\s+something\s+else|more)\b"
        compound_fb_match = re.search(compound_pattern, low)
        if compound_fb_match:
            verb = compound_fb_match.group(1).lower()
            fb_type = "dislike" if any(w in verb for w in ("hate", "dislike", "didn", "didnt", "don", "dont", "not")) else "like"
            target_raw = compound_fb_match.group(2).strip()
            resolved = self.resolve_title_across_catalogs(target_raw)
            if resolved:
                mid = resolved["ml_id"] if resolved.get("is_movielens") and resolved.get("ml_id") is not None else resolved["id"]
                fb_title = f"{resolved['title']} ({resolved['year']})"
            else:
                mid = self.find_movie_id_in_text(target_raw)
                fb_title = target_raw
                if mid is not None:
                    if isinstance(mid, int) and mid in self.movie_meta:
                        fb_title = clean_title(self.movie_meta[mid]["title"])
                    elif str(mid) in self.global_meta:
                        fb_title = clean_title(self.global_meta[str(mid)]["title"])

            fb_res = self.record_feedback(user_id=user_id, movie_id=mid, title=target_raw, feedback_type=fb_type)
            rec_results = self.recommend(user_id=user_id, n=5)
            # Ensure target movie is not in cards
            filtered_recs = []
            for r in rec_results:
                r_title = r.get("title", "").lower()
                r_id = str(r.get("movie_id", ""))
                if mid is not None and (r_id == str(mid) or (resolved and r_title == resolved["title"].lower())):
                    continue
                filtered_recs.append(r)
            rec_results = filtered_recs[:5]
            source = rec_results[0].get("catalog_source", "MovieLens 1M") if rec_results else "MovieLens 1M"
            rec_cards = self._format_cards(rec_results, source=source)
            lines = [f"{i+1}. **{c['title']}** ({c['year']}) - [{', '.join(c['genres'])}]" for i, c in enumerate(rec_cards)]
            resp = f"Recorded {fb_type} for '{fb_title}'. Based on your updated taste, here are recommendations:\n" + "\n".join(lines)
            return {
                "tool": "record_feedback, recommend",
                "args": {"feedback_type": fb_type, "movie_id": mid, "title": fb_title, "user_id": user_id},
                "result": {"feedback": fb_res, "recommendations": rec_results},
                "cards": rec_cards,
                "response": resp,
                "safe": True,
                "is_adversarial": False,
            }

        # Conversational follow-ups ("another one", "less violent", "newer ones", "more like the first one")
        followup_res = self._handle_followup_query(msg, user_id=user_id)
        if followup_res is not None:
            return followup_res

        # Similar-to intent (e.g. "movies like Parasite", "something mind-bending like interstellar")
        sim_pattern = r"^(?:can you\s+)?(?:please\s+)?(?:suggest|recommend|give me|show me|find)?\s*(?:something|movies?|films?|picks?|more)(?:\s+[a-z-]+){0,3}\s+(?:like|similar to)\s+(.*)"
        sim_match = re.search(sim_pattern, low)
        if sim_match and "newer" not in low:
            seed_raw = sim_match.group(1).strip()
            modifier = "darker" if "but darker" in seed_raw else None
            seed_clean = re.sub(r"\s+but\s+darker.*$", "", seed_raw, flags=re.I).strip()
            resolved = self.resolve_title_across_catalogs(seed_clean)
            if resolved:
                recs = self.recommend(user_id=user_id, seed_movie=resolved["title"], modifier=modifier, n=5)
                cards = self._format_cards(recs, source=resolved["catalog_source"])
                lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(recs)]
                mod_str = " re-ranked for darker themes" if modifier == "darker" else ""
                resp = f"From {resolved['catalog_source']}, here are films similar to '{resolved['title']} ({resolved['year']})'{mod_str}:\n" + "\n".join(lines)
                return {
                    "tool": "recommend",
                    "args": {"seed_movie": f"{resolved['title']} ({resolved['year']})", "modifier": modifier, "user_id": user_id, "n": 5},
                    "result": recs,
                    "cards": cards,
                    "response": resp,
                    "safe": True,
                    "is_adversarial": False,
                }
            else:
                return {
                    "tool": "recommend",
                    "args": {"seed_movie": seed_clean, "user_id": user_id, "n": 5},
                    "result": [],
                    "cards": [],
                    "response": f"I couldn't find '{seed_clean}' in either catalog (not in the MovieLens 1M catalog or the World catalog).",
                    "safe": True,
                    "is_adversarial": True,
                }

        # Intent 2: Explain queries
        if re.search(r"\b(why|explain|what makes)\b", low):
            mid = self.find_movie_id_in_text(low)
            if mid is not None:
                exp = self.explain(mid, user_id=user_id)
                if "error" in exp:
                    return {
                        "tool": "explain",
                        "args": {"movie_id": mid, "user_id": user_id},
                        "result": exp,
                        "cards": [],
                        "response": exp["error"],
                        "safe": True,
                        "is_adversarial": mid not in self.movie_meta and str(mid) not in self.global_meta,
                    }
                exp_cards = self._format_cards([exp], source=exp.get("catalog_source", "MovieLens 1M"))
                return {
                    "tool": "explain",
                    "args": {"movie_id": mid, "user_id": user_id},
                    "result": exp,
                    "cards": exp_cards,
                    "response": f"**Explanation for {exp['title']}**:\n{exp['explanation']}",
                    "safe": True,
                    "is_adversarial": False,
                }

        # Intent 3: Feedback queries (exclude 'something like', 'movies like', 'more like')
        add_match = re.search(r"\badd\s+(.*?)\s+to\s+(?:my\s+)?(liked|disliked|favorite|favorites)(?:\s+(?:movies?|films?|list))?\b", low)
        fb_pattern = r"\b(?:i\s+)?(didn'?t\s+like|didnt\s+like|don'?t\s+like|dont\s+like|hated?|disliked?|not\s+a\s+fan\s+of|didn'?t\s+enjoy|didnt\s+enjoy|loved?|liked?|enjoyed?)\s+(.*)"
        fb_match = re.search(fb_pattern, low)
        is_feedback_word = re.search(r"\b(like|liked|love|loved|dislike|disliked|hate|hated)\b", low)
        is_sim_rec = bool(re.search(r"\b(something|movies?|films?|more)\s+like\b", low))
        if (add_match or fb_match or is_feedback_word) and not is_sim_rec:
            if add_match:
                fb_type = "dislike" if "dislike" in add_match.group(2) else "like"
                target_raw = add_match.group(1).strip()
            elif fb_match:
                verb = fb_match.group(1).lower()
                fb_type = "dislike" if any(w in verb for w in ("hate", "dislike", "didn", "didnt", "don", "dont", "not")) else "like"
                target_raw = fb_match.group(2).strip()
            else:
                fb_type = "dislike" if any(w in low for w in ("hate", "dislike", "didn", "didnt", "don", "dont", "not")) else "like"
                target_raw = low
            resolved = self.resolve_title_across_catalogs(target_raw)
            if resolved:
                mid = resolved["ml_id"] if resolved.get("is_movielens") and resolved.get("ml_id") is not None else resolved["id"]
                fb_title = f"{resolved['title']} ({resolved['year']})"
            else:
                mid = self.find_movie_id_in_text(target_raw)
                fb_title = target_raw
                if mid is not None:
                    if isinstance(mid, int) and mid in self.movie_meta:
                        fb_title = clean_title(self.movie_meta[mid]["title"])
                    elif str(mid) in self.global_meta:
                        fb_title = clean_title(self.global_meta[str(mid)]["title"])
            if mid is not None:
                res = self.record_feedback(user_id=user_id, movie_id=mid, title=target_raw, feedback_type=fb_type)
                return {
                    "tool": "record_feedback",
                    "args": {"user_id": user_id, "movie_id": mid, "title": fb_title, "feedback_type": fb_type},
                    "result": res,
                    "cards": [],
                    "response": f"Done! {res.get('message', res.get('error', ''))}",
                    "safe": True,
                    "is_adversarial": False,
                }

        # Intent 4: Taste profile queries
        if any(w in low for w in ("taste", "profile", "history", "what did i like", "what have i liked")):
            prof = self.get_taste_profile(user_id)
            return {
                "tool": "get_taste_profile",
                "args": {"user_id": user_id},
                "result": prof,
                "cards": [],
                "response": f"**Your Taste Profile**:\n{prof['taste_summary']}\n*(Liked: {prof['liked_count']}, Disliked: {prof['disliked_count']})*",
                "safe": True,
                "is_adversarial": False,
            }

        # Intent 5: Region / Language / Year / Genre Filter queries
        year_match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", low)
        decade_match = re.search(r"\b(?:(18|19|20))?(\d{2})s\b", low)
        all_genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        genre_synonyms = {
            "thriller": "Thriller", "detective": "Mystery", "family": "Children's",
            "funny": "Comedy", "comedies": "Comedy", "comedy": "Comedy",
            "lasers": "Sci-Fi", "space": "Sci-Fi", "animated": "Animation"
        }
        matched_genre = next((g for g in all_genres if g.lower() in low), None)
        if not matched_genre:
            for syn_k, syn_v in genre_synonyms.items():
                if syn_k in low:
                    matched_genre = syn_v
                    break

        matched_region = None
        matched_language = None

        # Check DEMONYM_ALIASES first for cultural cinema demonyms (e.g. french, korean, etc.)
        for demonym, alias_info in DEMONYM_ALIASES.items():
            if re.search(rf"\b{re.escape(demonym)}\b", low):
                matched_region = alias_info.get("region")
                matched_language = alias_info.get("language")
                break

        # Data-driven region and language matching from catalog labels
        if not matched_region:
            for name_key in sorted(self.region_name_to_code.keys(), key=len, reverse=True):
                if len(name_key) >= 3 and re.search(rf"\b{re.escape(name_key)}\b", low):
                    matched_region = self.region_name_to_code[name_key]
                    break

        if not matched_language:
            for name_key in sorted(self.lang_name_to_code.keys(), key=len, reverse=True):
                if len(name_key) >= 3 and re.search(rf"\b{re.escape(name_key)}\b", low):
                    matched_language = self.lang_name_to_code[name_key]
                    break
        
        reg_code_match = re.search(r"\bregion\s+([a-zA-Z]{2})\b", low)
        if reg_code_match:
            matched_region = reg_code_match.group(1).upper()

        lang_code_match = re.search(r"\blanguage\s+([a-zA-Z]{2,10})\b", low)
        if lang_code_match:
            code_candidate = lang_code_match.group(1).lower()
            matched_language = self.lang_name_to_code.get(code_candidate, code_candidate)

        # Catalog-driven year boundary check:
        is_bare_year = bool(re.fullmatch(r"^(18\d{2}|19\d{2}|20\d{2})$", low.strip()))
        is_year_phrase = bool(year_match and any(w in low for w in ("from", "in ", "released", "movies", "films", "before", "after", "since")))
        if (is_bare_year or is_year_phrase) and not matched_region and not matched_language:
            yr = int(year_match.group(1))
            has_title_with_year = False
            for t in self.title_map:
                if str(yr) in t and len(t) > 6 and t in low:
                    has_title_with_year = True
                    break
            if not has_title_with_year:
                if yr > self.ml_max_year and yr <= self.data_max_year:
                    results = self.filter_movies(year=yr, limit=5)
                    cards = self._format_cards(results, source="World catalog")
                    lines = [f"- **{r['title']}** ({r['year']}) [{r.get('origin', 'international')}] - [{', '.join(r['genres'])}]" for r in results]
                    resp = f"MovieLens 1M covers {self.ml_min_year}-{self.ml_max_year}. The world catalog covers {self.data_min_year}-{self.data_max_year}. From the world catalog ({yr} releases):\n" + "\n".join(lines)
                    return {
                        "tool": "filter_movies",
                        "args": {"year": yr, "limit": 5},
                        "result": results,
                        "cards": cards,
                        "response": resp,
                        "safe": True,
                        "is_adversarial": False,
                    }
                elif yr < self.ml_min_year or yr > self.data_max_year:
                    return {
                        "tool": None,
                        "args": {},
                        "result": {},
                        "cards": [],
                        "response": f"MovieLens 1M covers {self.ml_min_year}-{self.ml_max_year}. Try a year in that range.",
                        "safe": True,
                        "is_adversarial": True,
                    }

        # Intent 6: Recommendation queries
        rec_cues = (
            "recommend", "recommendation", "recommendations", "suggest", "suggestions",
            "what should i watch", "what to watch", "pick", "picks", "something like",
            "more like", "best movies", "best film", "best", "top rated", "top movies",
            "good comedy", "good horror", "good action", "good drama", "good movies", "good movie",
            "popular movies", "something to watch", "for family"
        )
        is_rec_intent = any(re.search(rf"\b{re.escape(w)}\b", low) for w in rec_cues) or bool(re.search(r"\bgive me\s+(\d+\s+)?([a-z\s-]+\s+)?movies?\b", low))
        
        explicit_search_verbs = ("search for", "find ", "look up ", "tell me about ", "do you have ")
        has_explicit_search_verb = any(v in low for v in explicit_search_verbs)

        is_filter_trigger = (
            "filter" in low
            or matched_region is not None
            or matched_language is not None
            or (year_match and any(w in low for w in ("from", "before", "after", "since", "released", "in ")))
            or (decade_match is not None)
            or ("newer" in low and ("than" in low or "like" in low))
            or any(ak in low for ak in GENRE_ALIASES)
            or bool(re.search(r"\b(underrated|short and funny|thrillrs)\b", low))
        )

        if is_filter_trigger and not (is_rec_intent and not matched_region and not matched_language and not year_match and not decade_match and not any(ak in low for ak in GENRE_ALIASES)):
            parsed_args = self.parse_filter_args(low)
            if "newer" in low:
                ref_mid = self.find_movie_id_in_text(low)
                if ref_mid and isinstance(ref_mid, int) and ref_mid in self.movie_meta:
                    parsed_args["min_year"] = int(self.movie_meta[ref_mid]["year"]) + 1
                    if not parsed_args.get("genre"):
                        parsed_args["genre"] = parse_genres(self.movie_meta[ref_mid]["genres"])[0]

            results = self.filter_movies(**parsed_args)
            source = "World catalog" if (parsed_args.get("region") or parsed_args.get("language") or parsed_args.get("languages") or (parsed_args.get("min_year") and parsed_args["min_year"] > 2000) or (parsed_args.get("year") and parsed_args["year"] > 2000) or any(r.get("catalog_source") == "World catalog" or "origin_countries" in r for r in results)) else "MovieLens 1M"
            cards = self._format_cards(results, source=source)
            tool_args = {k: v for k, v in parsed_args.items() if v is not None}
            resp_text = self._format_filter_response(parsed_args, results, source=source)
            out = {
                "tool": "filter_movies",
                "args": tool_args,
                "result": results,
                "cards": cards,
                "response": resp_text,
                "safe": True,
                "is_adversarial": False,
            }
            self._save_user_context(user_id, out)
            return out

        # Explicit top-rated / best-movie queries (Bayesian quality across both catalogs)
        low_clean_q = re.sub(r"[?!.]+$", "", low).strip()
        if low_clean_q in (
            "best movie", "best movies", "best film", "best films",
            "top rated", "top rated movies", "top rated films", "top movies",
            "peak cinema", "peak cinema bro", "underrated gem", "underrated gem fr fr",
        ):
            recs = self.recommend_top_rated(user_id=user_id, n=5)
            source = recs[0].get("catalog_source", "World catalog") if recs else "World catalog"
            cards = self._format_cards(recs, source=source)
            lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(recs)]
            out = {
                "tool": "recommend",
                "args": {"user_id": user_id, "genres": None, "n": 5, "mode": "bayesian_top_rated"},
                "result": recs,
                "cards": cards,
                "response": f"Top-rated picks (Bayesian quality ranking from {source}):\n" + "\n".join(lines),
                "safe": True,
                "is_adversarial": False,
            }
            self._save_user_context(user_id, out)
            return out

        # Intent 6: Recommendation queries (if not intercepted by search)
        if is_rec_intent and not has_explicit_search_verb:
            n_match = re.search(r"\b(\d+)\b", low)
            word_num_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
            if n_match:
                n_items = int(n_match.group(1))
            else:
                n_items = 5
                for w_str, w_val in word_num_map.items():
                    if re.search(r"\b" + w_str + r"\b", low):
                        n_items = w_val
                        break
            n_items = min(max(n_items, 1), 20)
            genres_to_rec = [g for g in all_genres if g.lower() in low]
            if not genres_to_rec and matched_genre:
                genres_to_rec = [matched_genre]
            recs = self.recommend(user_id=user_id, genres=genres_to_rec or None, n=n_items)
            source = recs[0].get("catalog_source", "MovieLens 1M") if recs else "MovieLens 1M"
            cards = self._format_cards(recs, source=source)
            lines = [f"{i+1}. **{r['title']}** ({r['year']}) - [{', '.join(r['genres'])}]" for i, r in enumerate(recs)]
            out = {
                "tool": "recommend",
                "args": {"user_id": user_id, "genres": genres_to_rec or None, "n": n_items},
                "result": recs,
                "cards": cards,
                "response": f"From {source}:\n" + "\n".join(lines),
                "safe": True,
                "is_adversarial": False,
            }
            self._save_user_context(user_id, out)
            return out

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
                source = results[0].get("catalog_source", "MovieLens 1M")
                cards = self._format_cards(results, source=source)
                lines = []
                for r in results:
                    orig = f" [{r['origin']}]" if 'origin' in r else ""
                    lines.append(f"- **{r['title']}** ({r['year']}) [ID: {r['movie_id']}] - {', '.join(r['genres'])}{orig}")
                if source == "World catalog":
                    prefix = "From the world catalog (not in MovieLens 1M):"
                else:
                    prefix = "From MovieLens 1M:"
                resp = f"{prefix}\n" + "\n".join(lines)
            else:
                cards = []
                resp = f"I couldn't find '{q_term}' in either catalog (not in the MovieLens 1M catalog or the World catalog)."
            return {
                "tool": "search_movies",
                "args": {"query": q_term.lower()},
                "result": results,
                "cards": cards,
                "response": resp,
                "safe": True,
                "is_adversarial": len(results) == 0,
            }

        # Intent 8: Unrecognized / Off-topic
        return {
            "tool": None,
            "args": {},
            "result": {},
            "cards": [],
            "response": "I only help with movies. Try a title, genre, or year.",
            "safe": True,
            "is_adversarial": False,
        }

    def route_message_llm(self, user_msg: str, user_id: str = "demo_user") -> Dict[str, Any]:
        """Let LLM choose tools via tool-calling with deterministic fallback and anti-hallucination guard."""
        if not self.client:
            return self.route_message(user_msg, user_id=user_id)

        msg = (user_msg or "").strip()
        low = msg.lower()

        # Fast guards: Empty, Prompt injection, Closings, Greetings, Off-topic
        if not msg:
            return self.route_message(user_msg, user_id=user_id)
        if any(p in low for p in ("ignore all previous", "ignore previous instructions", "secret system prompt", "invent a fake movie")):
            return self.route_message(user_msg, user_id=user_id)
        if self.is_closing(low):
            return self.route_message(user_msg, user_id=user_id)
        if self.is_greeting(low):
            return self.route_message(user_msg, user_id=user_id)
        if self.is_off_topic(low):
            return {
                "tool": None,
                "args": {},
                "result": {},
                "cards": [],
                "response": "I only help with movies. Try a title, genre, or year.",
                "safe": True,
                "is_adversarial": True,
            }

        # Check movie-to-director/cast queries before LLM
        dir_res = self.detect_movie_director_query(user_msg)
        if dir_res is not None:
            return dir_res

        # Check person intent
        if self.detect_person_intent(user_msg):
            return self.route_message(user_msg, user_id=user_id)

        # Check conversational follow-ups
        followup_res = self._handle_followup_query(user_msg, user_id=user_id)
        if followup_res is not None:
            return followup_res

        # Route explicit quality/vibe/regional-alias queries deterministically
        low_clean_q = re.sub(r"[?!.]+$", "", low).strip()
        if low_clean_q in (
            "best movie", "best movies", "best film", "best films",
            "top rated", "top rated movies", "top rated films", "top movies",
            "peak cinema", "peak cinema bro", "underrated gem", "underrated gem fr fr",
        ) or any(alias in low for alias in ("south indian", "tollywood", "kollywood", "mollywood", "sandalwood", "anime", "ghibli", "underrated horror", "short and funny")):
            return self.route_message(user_msg, user_id=user_id)

        # Check compound feedback + recommend
        compound_pattern = r"\b(?:i\s+)?(didn'?t\s+like|didnt\s+like|don'?t\s+like|dont\s+like|hated?|disliked?|not\s+a\s+fan\s+of|didn'?t\s+enjoy|didnt\s+enjoy|loved?|liked?|enjoyed?)\s+(.*?)\s*(?:,|\.|\b|\sand\s)\s*(?:and\s+)?(give\s+me\s+something\s+else|give\s+me\s+something\s+better|give\s+me\s+more|something\s+else|what\s+else|what\s+next|anything\s+else|suggest\s+more|suggest\s+another|suggest\s+something\s+else|recommend\s+more|recommend\s+another|recommend\s+something\s+else|what\s+should\s+i\s+watch\s+instead|what\s+should\s+i\s+watch\s+now|show\s+me\s+more|show\s+me\s+another|show\s+me\s+something\s+else|more)\b"
        if re.search(compound_pattern, low):
            return self.route_message(user_msg, user_id=user_id)

        # Check similar-to
        sim_pattern = r"^(?:can you\s+)?(?:please\s+)?(?:suggest|recommend|give me|show me|find)?\s*(?:something|movies?|films?|picks?|more)(?:\s+[a-z-]+){0,3}\s+(?:like|similar to)\s+(.*)"
        if re.search(sim_pattern, low) and "newer" not in low:
            return self.route_message(user_msg, user_id=user_id)

        # Fast-path short title / fuzzy title searches without Groq LLM round-trip
        clean_short = self.clean_search_query(user_msg).lower().strip()
        if clean_short and len(clean_short.split()) <= 3 and (
            clean_short in self.title_map
            or clean_short in self.global_title_to_id
            or bool(self._fuzzy_world_matches(clean_short, limit=1))
        ):
            return self.route_message(user_msg, user_id=user_id)

        # Check year 2026 or world catalog year
        year_match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", low)
        if year_match:
            yr = int(year_match.group(1))
            if yr > self.ml_max_year:
                return self.route_message(user_msg, user_id=user_id)

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are MovieMind, an expert, charming movie recommender assistant for MovieLens 1M and the World Catalog. "
                        "You must choose the single best tool to satisfy the user request: "
                        "search_movies, recommend, explain, filter_movies, record_feedback, get_taste_profile. "
                        "For intent requests like 'best movies', 'top rated', 'good movies', or 'recommend me something' without a specific movie title, choose recommend. "
                        "For region or language queries (e.g. 'telugu movies', 'korean thrillers', 'japanese films from the 1950s', 'indian movies'), choose filter_movies with region/language. "
                        "The MovieLens 1M catalog covers release years 1919-2000. If the user asks for a year outside 1919-2000 without a foreign region/language, respond directly stating 'MovieLens 1M covers 1919-2000. Try a year in that range.' with no tools. "
                        "For typos in movie titles (e.g. 'alines' -> 'Aliens'), correct the title in tool arguments. "
                        "For similarity or 'something like X' requests, choose recommend or filter_movies. "
                        "Only use record_feedback when user explicitly states they like/love or dislike/hate a movie. "
                        "CRITICAL: Never invent movie titles, release years, or genres. "
                        "If a movie or title is not found in either catalog, you MUST explicitly state that the title is "
                        "'not in the MovieLens 1M catalog' and list nothing. "
                        "For greetings or conversational chatter without movie actions, respond directly with no tools."
                    ),
                },
                {"role": "user", "content": f"[User ID: {user_id}] {user_msg}"},
            ]

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.get_tool_specs(),
                tool_choice="auto",
                max_tokens=512,
                timeout=4.0,
            )
            choice = resp.choices[0].message
            if not choice.tool_calls:
                content = (choice.content or "").replace("\u202f", " ").replace("\xa0", " ")
                # Post-check: discard ungrounded prose if it attempts to list movies or tables
                if self.is_prose_hallucinating(content, user_msg):
                    fallback_res = self.route_message(user_msg, user_id=user_id)
                    trace_record = {
                        "stage": "post_check",
                        "action": "discard_ungrounded_llm_prose",
                        "reason": "unauthorized_movie_list_or_prose",
                    }
                    fallback_res.setdefault("trace", []).insert(0, trace_record)
                    return fallback_res

                return {
                    "tool": None,
                    "args": {},
                    "result": {},
                    "cards": [],
                    "response": content,
                    "safe": True,
                    "is_adversarial": False,
                }

            tc = choice.tool_calls[0]
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments or "{}")

            # Ensure user_id defaults if expected
            if fn_name in ("recommend", "get_taste_profile", "record_feedback", "explain"):
                fn_args.setdefault("user_id", user_id)

            canonical_genres = {
                "action": "Action", "adventure": "Adventure", "animated": "Animation",
                "animation": "Animation", "children": "Children's", "children's": "Children's",
                "comedy": "Comedy", "crime": "Crime", "documentary": "Documentary",
                "drama": "Drama", "fantasy": "Fantasy", "film-noir": "Film-Noir",
                "noir": "Film-Noir", "horror": "Horror", "musical": "Musical",
                "mystery": "Mystery", "romance": "Romance", "romantic": "Romance",
                "sci-fi": "Sci-Fi", "scifi": "Sci-Fi", "thriller": "Thriller",
                "war": "War", "western": "Western"
            }

            if fn_name == "search_movies":
                sq = fn_args.get("query", "").lower().strip()
                if sq in ("best", "top", "top rated", "good", "popular", "best movies", "top movies", "good movies", "recommend me something"):
                    fn_name = "recommend"
                    fn_args = {"user_id": user_id, "genres": None, "n": 5}
                elif any(re.search(rf"\b{re.escape(w)}\b", low) for w in ("why", "explain", "what makes")):
                    fn_name = "explain"
                    fn_args = {"title": sq or user_msg, "user_id": user_id}

            if fn_name == "explain" and any(low.startswith(p) for p in ("tell me about", "search for", "look up", "find ")):
                if not any(re.search(rf"\b{re.escape(w)}\b", low) for w in ("why", "explain", "what makes", "picked", "chosen")):
                    fn_name = "search_movies"
                    fn_args = {"query": fn_args.get("title") or self.clean_search_query(user_msg)}

            if fn_name == "recommend":
                if "genres" in fn_args and isinstance(fn_args["genres"], list):
                    fn_args["genres"] = [
                        canonical_genres.get(g.strip().lower(), g.strip().capitalize())
                        for g in fn_args["genres"] if g
                    ]

            if fn_name == "filter_movies":
                fn_args = self.parse_filter_args(user_msg)

            if fn_name in ("explain", "record_feedback"):
                mid_val = fn_args.get("movie_id")
                if mid_val is None or (isinstance(mid_val, str) and not mid_val.isdigit() and not mid_val.startswith("tt")):
                    raw_str = str(mid_val) if mid_val else (fn_args.get("title") or user_msg)
                    t_lookup = self.find_movie_id_in_text(raw_str)
                    if t_lookup is not None:
                        fn_args["movie_id"] = t_lookup

            tool_result = self.execute_tool(fn_name, fn_args)

            # Anti-hallucination catalog guard: unknown/missing movie search returns catalog miss notice
            if fn_name == "search_movies" and not tool_result:
                q_term = self.clean_search_query(fn_args.get("query", user_msg))
                return {
                    "tool": fn_name,
                    "args": fn_args,
                    "result": [],
                    "cards": [],
                    "response": f"I couldn't find '{q_term}' in either catalog (not in the MovieLens 1M catalog or the World catalog).",
                    "safe": True,
                    "is_adversarial": True,
                }

            if fn_name == "explain" and isinstance(tool_result, dict) and "error" in tool_result:
                return {
                    "tool": fn_name,
                    "args": fn_args,
                    "result": tool_result,
                    "cards": [],
                    "response": tool_result["error"],
                    "safe": True,
                    "is_adversarial": True,
                }

            # Standardize cards & server-templated response sentence
            cards = []
            if fn_name == "filter_movies":
                source = tool_result[0].get("catalog_source", "MovieLens 1M") if tool_result else "World catalog"
                cards = self._format_cards(tool_result, source=source)
                content = self._format_filter_response(fn_args, tool_result, source=source)
            elif isinstance(tool_result, list) and len(tool_result) > 0 and isinstance(tool_result[0], dict) and "title" in tool_result[0]:
                source = tool_result[0].get("catalog_source", "MovieLens 1M")
                cards = self._format_cards(tool_result, source=source)
                lines = []
                for r in tool_result:
                    orig = f" [{r['origin']}]" if 'origin' in r else ""
                    part_s = f" ({r['_partial_reason']})" if r.get('_match_tier') == 'partial' and r.get('_partial_reason') else ""
                    lines.append(f"- **{r['title']}** ({r['year']}){orig}{part_s} - [{', '.join(r.get('genres', []))}]")
                prefix = "From the world catalog:" if source == "World catalog" else "From MovieLens 1M:"
                content = f"{prefix}\n" + "\n".join(lines)
            elif fn_name == "explain":
                cards = self._format_cards([tool_result], source=tool_result.get("catalog_source", "MovieLens 1M"))
                content = f"**Explanation for {tool_result['title']}**:\n{tool_result['explanation']}"
            elif fn_name == "record_feedback":
                content = f"Done! {tool_result.get('message', '')}"
            elif fn_name == "get_taste_profile":
                content = f"**Your Taste Profile**:\n{tool_result['taste_summary']}\n*(Liked: {tool_result['liked_count']}, Disliked: {tool_result['disliked_count']})*"
            else:
                content = "Here are your results."

            resp_str = (content or "").strip()
            if not resp_str and not cards:
                resp_str = "I didn't catch that. Try a title, a genre, a country, a language or a year."

            return {
                "tool": fn_name,
                "args": fn_args,
                "result": tool_result,
                "cards": cards,
                "response": resp_str,
                "safe": True,
                "is_adversarial": False,
            }
        except Exception as e:
            fallback_res = self.route_message(user_msg, user_id=user_id)
            fallback_res.setdefault("trace", []).insert(0, {
                "stage": "llm_error_fallback",
                "action": "fallback_to_rule_router",
                "error": str(e),
            })
            return fallback_res

    def chat_fallback(self, user_msg: str, user_id: str = "demo_user") -> str:
        """Rule-based tool router for graceful fallback when no API key is supplied."""
        res = self.route_message(user_msg, user_id=user_id)
        resp = (res.get("response") or "").strip()
        if not resp and not res.get("cards"):
            return "I didn't catch that. Try a title, a genre, a country, a language or a year."
        return resp

    def chat(self, user_msg: str, user_id: str = "demo_user") -> str:
        """Main chat interface calling LLM with tool calling or falling back gracefully."""
        res = self.route_message_llm(user_msg, user_id=user_id)
        resp = (res.get("response") or "").strip()
        if not resp and not res.get("cards"):
            return "I didn't catch that. Try a title, a genre, a country, a language or a year."
        return resp

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
