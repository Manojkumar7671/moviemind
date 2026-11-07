import sys
import os
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import MovieMindAgent

def evaluate_message(msg: str, res: dict, router_name: str) -> tuple[str, bool]:
    """Check result against ground-truth expectations for PART 2 messages."""
    tool = str(res.get("tool"))
    cards = res.get("cards", [])
    first_3 = [c.get("title", "") for c in cards[:3]]
    src = cards[0].get("catalog_source") if cards else "None"
    resp = res.get("response", "")

    passed = True
    reason = "OK"

    if msg == "telugu movies":
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

    elif msg == "suggest something like Toy Story but darker":
        if "recommend" not in tool or len(cards) == 0:
            passed = False

    elif msg == "i hated Titanic what else":
        if "record_feedback" not in tool or "recommend" not in tool or len(cards) == 0:
            passed = False
        elif any("Titanic" in c.get("title", "") for c in cards):
            passed = False

    elif msg == "i loved Aliens, suggest more":
        if "record_feedback" not in tool or "recommend" not in tool or len(cards) == 0:
            passed = False

    elif msg == "tell me about Inception":
        if "search_movies" not in tool or len(cards) == 0:
            passed = False
        elif cards[0].get("title") != "Inception" or src != "World catalog":
            passed = False

    elif msg == "tell me about alines":
        if "search_movies" not in tool or len(cards) == 0:
            passed = False
        elif cards[0].get("title") != "Aliens" or src != "MovieLens 1M":
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

    elif msg in ("hello", "thanks"):
        if tool != "None" or len(cards) > 0:
            passed = False

    elif msg == "what is 2+2":
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
    ]

    print("| message | router | tool(s) called | number of cards | first 3 titles | catalog source | PASS/FAIL |")
    print("|---|---|---|---|---|---|---|")

    for router_type in ("fallback", "llm"):
        for msg in messages:
            if router_type == "fallback":
                res = agent.route_message(msg)
            else:
                res = agent.route_message_llm(msg)
            
            tool = str(res.get("tool"))
            cards = res.get("cards", [])
            card_count = len(cards)
            first_3 = [c.get("title", "") for c in cards[:3]]
            first_3_str = ", ".join(first_3) if first_3 else "[]"
            src = cards[0].get("catalog_source") if cards else "None"
            
            status = evaluate_message(msg, res, router_type)
            print(f"| {msg} | {router_type} | {tool} | {card_count} | {first_3_str} | {src} | {status} |")

if __name__ == "__main__":
    main()
