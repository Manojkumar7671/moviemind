import sys
import os
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import MovieMindAgent

def evaluate_message(msg: str, res: dict, router_name: str) -> tuple[str, bool]:
    """Check result against ground-truth expectations for PART 2 and new intent cases."""
    tool = str(res.get("tool"))
    cards = res.get("cards", [])
    first_3 = [c.get("title", "") for c in cards[:3]]
    src = cards[0].get("catalog_source") if cards else "None"
    resp = res.get("response", "")

    passed = True

    if msg == "movies like Parasite":
        if "recommend" not in tool or len(cards) < 5:
            passed = False
        elif src != "World catalog":
            passed = False
        elif "Parasite" in [c.get("title") for c in cards]:
            passed = False
        elif "Parasite (2019)" not in resp:
            passed = False

    elif msg == "show me french comedies":
        if "filter_movies" not in tool or len(cards) < 5:
            passed = False
        elif src != "World catalog":
            passed = False
        elif not any("FR" in c.get("origin_countries", []) or "fr" in c.get("original_languages", []) for c in cards):
            passed = False

    elif msg == "i didnt like The Matrix, give me something else":
        if "record_feedback" not in tool or "recommend" not in tool or len(cards) < 5:
            passed = False
        elif any("The Matrix" in c.get("title", "") or "Matrix" == c.get("title", "") for c in cards):
            passed = False

    elif msg in ("thanks bye", "thank you goodbye", "that's all for today bye"):
        if tool != "None" or len(cards) > 0:
            passed = False

    elif msg == "telugu movies":
        if "filter_movies" not in tool or len(cards) == 0:
            passed = False
        elif src != "World catalog":
            passed = False
        elif "Mouna Ragam" in [c.get("title") for c in cards]:
            passed = False
        elif not all("te" in c.get("original_languages", []) for c in cards):
            passed = False

    elif msg == "korean thrillers":
        if "filter_movies" not in tool or len(cards) == 0:
            passed = False
        elif src != "World catalog":
            passed = False
        elif not all("ko" in c.get("original_languages", []) for c in cards):
            passed = False
        elif not all("Thriller" in c.get("genres", []) for c in cards):
            passed = False

    elif msg == "japanese films from the 1950s":
        if "filter_movies" not in tool or len(cards) == 0:
            passed = False
        elif src != "World catalog":
            passed = False
        elif not all("JP" in c.get("origin_countries", []) for c in cards):
            passed = False
        elif not all(1950 <= int(c.get("year", 0)) <= 1959 for c in cards):
            passed = False

    elif msg == "indian movies":
        if "filter_movies" not in tool or len(cards) == 0:
            passed = False
        elif src != "World catalog":
            passed = False
        elif not all("IN" in c.get("origin_countries", []) for c in cards):
            passed = False

    elif msg in ("suggest something like Toy Story but darker", "movies like Aliens", "films similar to Pulp Fiction"):
        if "recommend" not in tool or len(cards) == 0:
            passed = False
        elif msg == "films similar to Pulp Fiction" and any("Pulp Fiction" == c.get("title") for c in cards):
            passed = False
        elif msg == "movies like Aliens" and any("Aliens" == c.get("title") for c in cards):
            passed = False

    elif msg in ("i hated Titanic what else", "i loved Aliens, suggest more", "i didnt like Jurassic Park, what else", "loved Fargo, suggest more"):
        if "record_feedback" not in tool or "recommend" not in tool or len(cards) == 0:
            passed = False
        elif "Titanic" in msg and any("Titanic" in c.get("title", "") for c in cards):
            passed = False
        elif "Jurassic Park" in msg and any("Jurassic Park" in c.get("title", "") for c in cards):
            passed = False
        elif "Fargo" in msg and any("Fargo" in c.get("title", "") for c in cards):
            passed = False

    elif msg == "tell me about Inception":
        if "search_movies" not in tool or len(cards) == 0:
            passed = False
        elif cards[0].get("title") != "Inception" or src != "World catalog":
            passed = False

    elif msg in ("tell me about alines", "tell me about jurasic park"):
        if "search_movies" not in tool or len(cards) == 0:
            passed = False
        elif msg == "tell me about alines" and (cards[0].get("title") != "Aliens" or src != "MovieLens 1M"):
            passed = False
        elif msg == "tell me about jurasic park" and (cards[0].get("title") != "Jurassic Park" or src != "MovieLens 1M"):
            passed = False

    elif msg == "tell me about Zzxqv movie":
        if len(cards) > 0:
            passed = False
        elif "couldn't find" not in resp.lower() and "not in" not in resp.lower():
            passed = False
        elif "Movie ID None" in resp:
            passed = False

    elif msg == "movies in 2026":
        if "filter_movies" not in tool or len(cards) == 0:
            passed = False
        elif not all(c.get("year") == 2026 for c in cards):
            passed = False

    elif msg == "movies from 1999":
        if "filter_movies" not in tool or len(cards) == 0:
            passed = False
        elif not all(c.get("year") == 1999 for c in cards):
            passed = False

    elif msg in ("spanish horror", "german sci-fi", "italian dramas"):
        if "filter_movies" not in tool or len(cards) == 0:
            passed = False
        elif src != "World catalog":
            passed = False

    elif msg in ("hello", "thanks", "good evening"):
        if tool != "None" or len(cards) > 0:
            passed = False

    elif msg in ("what is 2+2", "what is the speed of light"):
        if tool != "None" or len(cards) > 0:
            passed = False
        elif "I only help with movies" not in resp:
            passed = False

    elif msg in ("best movies", "top rated"):
        if "recommend" not in tool or len(cards) == 0:
            passed = False

    status_str = "PASS" if passed else "FAIL"
    return status_str

def main():
    agent = MovieMindAgent()
    messages = [
        # 4 Core Intent Fixes
        "movies like Parasite",
        "show me french comedies",
        "i didnt like The Matrix, give me something else",
        "thanks bye",
        # Original PART 2 Grounding Messages
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
        # 12 New Phrasings across Intent Families
        "films similar to Pulp Fiction",
        "movies like Aliens",
        "spanish horror",
        "german sci-fi",
        "italian dramas",
        "i didnt like Jurassic Park, what else",
        "loved Fargo, suggest more",
        "thank you goodbye",
        "that's all for today bye",
        "good evening",
        "what is the speed of light",
        "movies from 1999",
        "tell me about jurasic park",
    ]

    for router_type in ("fallback", "llm"):
        print(f"\n### {router_type.upper()} Router")
        print("| message | tool(s) | cards | first 3 titles with year | catalog source | PASS/FAIL |")
        print("|---|---|---|---|---|---|")
        for msg in messages:
            if router_type == "fallback":
                res = agent.route_message(msg)
            else:
                res = agent.route_message_llm(msg)

            tool = str(res.get("tool"))
            cards = res.get("cards", [])
            card_count = len(cards)
            first_3 = [f"{c.get('title', '')} ({c.get('year', '')})" for c in cards[:3]]
            first_3_str = ", ".join(first_3) if first_3 else "[]"
            src = cards[0].get("catalog_source") if cards else "None"

            status = evaluate_message(msg, res, router_type)
            print(f"| {msg} | {tool} | {card_count} | {first_3_str} | {src} | {status} |")

if __name__ == "__main__":
    main()
