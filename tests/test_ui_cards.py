"""UI and Card Header DOM regression tests.

Verifies:
1. Discover (MovieLens) cards do NOT render "Origin: ..." or language badges.
2. World cinema cards DO render "Origin: ..." and language badges when present.
3. Card header keeps year and match badge on one line without wrapping,
   with the shortened "98% match" badge and "Relative to this list" tooltip.
"""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from app.main import app
from scripts.check_card_dom import run_dom_check

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_card_dom_rendering_and_header_constraints():
    """Verify via DOM simulation that origin labels NEVER appear on Discover cards."""
    assert run_dom_check() is True, "Card DOM check failed"


def test_discover_api_payload_has_no_origin():
    """Verify that MovieLens recommendations API payload does not contain origin fields."""
    client = TestClient(app)
    res = client.post("/api/recommend", json={"user_id": "demo_user", "genres": ["Action"], "n": 6})
    assert res.status_code == 200
    data = res.json()
    recs = data.get("recommendations", [])
    assert len(recs) > 0

    for r in recs:
        # Discover cards must not include origin fields
        assert "origin" not in r or r["origin"] is None
        assert "origin_countries" not in r
        assert "origin_country_names" not in r
        assert "original_languages" not in r
        assert "original_language_names" not in r


@pytest.mark.needs_catalog
@pytest.mark.skipif(
    not (ROOT_DIR / "artifacts" / "global_catalog.parquet").exists(),
    reason="artifacts/global_catalog.parquet missing (run scripts/build_catalog.py)",
)
def test_world_api_payload_has_origin():
    """Verify that World Cinema recommendations API payload includes origin fields."""
    client = TestClient(app)
    res = client.get("/api/global/recommend?n=5&region=FR")
    assert res.status_code == 200
    data = res.json()
    recs = data.get("recommendations", [])
    assert len(recs) > 0

    for r in recs:
        assert "origin_countries" in r
        assert "origin_country_names" in r
        assert "FR" in r["origin_countries"]
        assert len(r["origin"]) > 0
