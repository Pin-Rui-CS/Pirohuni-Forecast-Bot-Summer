from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


DEFAULT_MARKET_URL = "https://manifold.markets/aidan0626/how-many-perfect-scores-will-there"
API_BASE = "https://api.manifold.markets/v0"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch one Manifold market through the public API and print useful fields.",
    )
    parser.add_argument("url", nargs="?", default=DEFAULT_MARKET_URL)
    parser.add_argument("--raw", action="store_true", help="Print the full JSON payload.")
    args = parser.parse_args()

    market = fetch_market_from_url(args.url)
    print_market_summary(market)
    if args.raw:
        print("\n=== Raw JSON ===")
        print(json.dumps(market, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def fetch_market_from_url(url: str) -> dict[str, Any]:
    path_parts = [part for part in urlparse(url).path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError(f"Could not parse Manifold user/slug from URL: {url}")

    username, slug = path_parts[0], path_parts[1]
    errors: list[str] = []

    for endpoint in [
        f"{API_BASE}/slug/{quote(slug)}",
        f"{API_BASE}/slug/{quote(username)}/{quote(slug)}",
    ]:
        try:
            data = get_json(endpoint)
            if isinstance(data, dict) and data.get("id"):
                return data
        except Exception as exc:
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")

    search_results = get_json(
        f"{API_BASE}/search-markets?term={quote(slug.replace('-', ' '))}&limit=20&filter=all"
    )
    if not isinstance(search_results, list):
        raise ValueError(f"Unexpected search response: {type(search_results).__name__}")

    for market in search_results:
        if not isinstance(market, dict):
            continue
        if _market_matches_url(market, username, slug):
            market_id = market.get("id")
            if market_id:
                return get_json(f"{API_BASE}/market/{quote(str(market_id))}")
            return market

    searched = "\n".join(
        f"- {m.get('question')} | slug={m.get('slug')} | url={m.get('url')}"
        for m in search_results[:10]
        if isinstance(m, dict)
    )
    raise RuntimeError(
        "Could not fetch exact market by slug or search.\n"
        f"Slug endpoint errors:\n{chr(10).join(errors)}\n"
        f"Search results considered:\n{searched}"
    )


def get_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "Pirohuni-Forecast-Bot test script"})
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc)) from exc
    return json.loads(body)


def print_market_summary(market: dict[str, Any]) -> None:
    print("=== Manifold API Market Summary ===")
    for key in [
        "id",
        "question",
        "url",
        "slug",
        "outcomeType",
        "isResolved",
        "probability",
        "volume",
        "totalLiquidity",
        "closeTime",
        "createdTime",
    ]:
        print(f"{key}: {market.get(key)!r}")

    answers = market.get("answers")
    if isinstance(answers, list):
        print("\n=== Answers ===")
        for answer in answers:
            if not isinstance(answer, dict):
                continue
            print(
                "- {text}: probability={probability!r}, number={number!r}, id={id!r}".format(
                    text=answer.get("text"),
                    probability=answer.get("probability"),
                    number=answer.get("number"),
                    id=answer.get("id"),
                )
            )

    print("\n=== Long Text Fields ===")
    for key in sorted(market):
        value = market.get(key)
        if isinstance(value, str) and len(value.strip()) >= 80:
            print(f"\n--- {key} ({len(value)} chars) ---")
            print(value)

    print("\n=== Description/Criteria-Like Keys Present ===")
    for key in sorted(market):
        lowered = key.lower()
        if any(marker in lowered for marker in ("description", "criteria", "resolution", "text")):
            print(f"{key}: {type(market.get(key)).__name__}")


def _market_matches_url(market: dict[str, Any], username: str, slug: str) -> bool:
    market_url = str(market.get("url") or "")
    market_slug = str(market.get("slug") or "")
    return (
        market_url.rstrip("/").endswith(f"/{username}/{slug}")
        or market_url.rstrip("/").endswith(f"/{slug}")
        or market_slug == slug
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
