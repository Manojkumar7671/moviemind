import sys
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import MovieMindAgent

class MockChoice:
    def __init__(self, content: str):
        self.message = MagicMock()
        self.message.content = content
        self.message.tool_calls = None

class MockCompletionResponse:
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]

def test_chat_grounding_property():
    agent = MovieMindAgent()
    # Ensure client is present so route_message_llm enters LLM path
    if not agent.client:
        agent.client = MagicMock()

    messages = [
        # 14 PART 2 messages
        "telugu movies",
        "korean thrillers",
        "japanese films from the 1950s",
        "indian movies",
        "suggest something like Toy Story but darker",
        "i hated Titanic what else",
        "i loved Aliens, suggest more",
        "tell me about Inception",
        "tell me about alines",
        "tell me about Zzxqv movie",
        "movies in 2026",
        "hello",
        "what is 2+2",
        "best movies",
        # 16 varied phrasings (typos, vague asks, mixed intents)
        "tell me about alian",
        "find something like the matrix but more philosophical",
        "i really disliked Jurassic Park, give me something better",
        "good sci-fi comedies from the 1980s",
        "french crime dramas from 1995",
        "can you recommend some exciting action movies",
        "german thrillers",
        "i loved The Godfather what else should i watch",
        "tell me about Terminator 2",
        "what should i watch tonight",
        "spanish horror films",
        "films released in 1994",
        "suggest movies similar to Toy Story",
        "top animated movies",
        "i hate Star Wars what else",
        "movies from 2025",
    ]

    assert len(messages) == 30, f"Expected 30 messages, got {len(messages)}"

    fake_prose = (
        "Here are some great movies you should watch:\n"
        "1. **CyberGala 2099** (2099) - Sci-Fi thriller\n"
        "2. **NeoTokyo Drift** (2045) - Action racing\n"
        "3. **Space Pirates of Jupiter** (2030) - Adventure"
    )

    total_messages = len(messages)
    discarded_llm_prose_count = 0
    violations = 0
    invented_titles = ["CyberGala 2099", "NeoTokyo Drift", "Space Pirates of Jupiter"]

    for msg in messages:
        # Mock client to return free-form prose with invented titles
        agent.client.chat.completions.create = MagicMock(return_value=MockCompletionResponse(fake_prose))

        res = agent.route_message_llm(msg)
        resp = res.get("response", "")
        cards = res.get("cards", [])
        trace = res.get("trace", [])

        # Check for discarded prose in trace
        is_discarded = any(
            t.get("stage") == "post_check" and t.get("action") == "discard_ungrounded_llm_prose"
            for t in trace
        )
        if is_discarded:
            discarded_llm_prose_count += 1

        # Violation check 1: None of the invented titles appear in final response or cards
        for it in invented_titles:
            if it.lower() in resp.lower():
                violations += 1
            if any(it.lower() in c.get("title", "").lower() for c in cards):
                violations += 1

        # Violation check 2: Forbidden string "Movie ID None"
        if "Movie ID None" in resp:
            violations += 1

        # Violation check 3: For movie queries that return cards, verify all cards come from real tools
        for c in cards:
            assert c.get("catalog_source") in ("MovieLens 1M", "World catalog")

    print(f"\ntotal messages: {total_messages}, discarded LLM prose count: {discarded_llm_prose_count}, violations: {violations}")

    assert violations == 0
    assert discarded_llm_prose_count > 0

if __name__ == "__main__":
    test_chat_grounding_property()
