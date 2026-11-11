"""tests/test_parity.py

Asserts 100% identical top-10 recommendations for 20 fixed users and
100% identical card IDs for 10 chat messages against the saved baseline.
"""

import json
from pathlib import Path
import pytest

from src.agent import MovieMindAgent

TESTS_DIR = Path(__file__).resolve().parent
BASELINE_FILE = TESTS_DIR / "baseline_parity.json"


@pytest.fixture(scope="module")
def agent():
    return MovieMindAgent()


@pytest.fixture(scope="module")
def baseline_data():
    assert BASELINE_FILE.exists(), f"Baseline parity file missing: {BASELINE_FILE}"
    with open(BASELINE_FILE, "r") as f:
        return json.load(f)


def test_fixed_users_recommendation_parity(agent, baseline_data):
    """Verify that top-10 recommendations for 20 users match baseline exactly."""
    users_baseline = baseline_data["users"]
    assert len(users_baseline) == 20, f"Expected 20 baseline users, got {len(users_baseline)}"

    for uid_str, expected_ids in users_baseline.items():
        recs = agent.recommend(int(uid_str), n=10)
        actual_ids = [r["movie_id"] for r in recs]
        assert actual_ids == expected_ids, (
            f"User {uid_str} recommendation mismatch!\n"
            f"Expected: {expected_ids}\n"
            f"Actual:   {actual_ids}"
        )


def test_chat_queries_card_parity(agent, baseline_data):
    """Verify that returned card IDs for 10 chat queries match baseline exactly."""
    chat_baseline = baseline_data["chat"]
    assert len(chat_baseline) == 10, f"Expected 10 chat baseline queries, got {len(chat_baseline)}"

    for msg, expected_card_ids in chat_baseline.items():
        res = agent.route_message(msg)
        actual_cards = res.get("cards", [])
        actual_card_ids = [c["movie_id"] for c in actual_cards]
        assert actual_card_ids == expected_card_ids, (
            f"Chat query '{msg}' card mismatch!\n"
            f"Expected: {expected_card_ids}\n"
            f"Actual:   {actual_card_ids}"
        )
