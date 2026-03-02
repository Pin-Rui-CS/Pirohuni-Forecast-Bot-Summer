from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import time
from urllib.parse import urlparse

import httpx
from forecasting_bot import (
    API_BASE_URL,
    AUTH_HEADERS,
    TOURNAMENT_MAPPING,
    CURRENT_METACULUS_CUP_ID,
    METACULUS_REQUEST_INTERVAL,
)
from resolution_scraper.extraction import classify_url, extract_resolution_urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List URLs found in Metaculus Cup questions and classify common URL types."
        )
    )
    parser.add_argument(
        "--tournament",
        type=str,
        default="metaculus-cup",
        help=(
            "Tournament name/id. Defaults to 'metaculus-cup'. "
            f"Known names: {', '.join(TOURNAMENT_MAPPING.keys())}"
        ),
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Number of posts to fetch per API page.",
    )
    return parser.parse_args()


def resolve_tournament_id(tournament_arg: str) -> int | str:
    key = tournament_arg.lower().strip()
    if key in TOURNAMENT_MAPPING:
        return TOURNAMENT_MAPPING[key]
    try:
        return int(tournament_arg)
    except ValueError:
        return tournament_arg if tournament_arg else CURRENT_METACULUS_CUP_ID


def list_posts_from_tournament_all_accessible(
    tournament_id: int | str, offset: int = 0, count: int = 50
) -> list[dict]:
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
                "discrete",
            ]
        ),
        "tournaments": [tournament_id],
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    time.sleep(METACULUS_REQUEST_INTERVAL)
    response = httpx.get(url, headers=AUTH_HEADERS, params=url_qparams, timeout=30.0)
    if response.status_code >= 400:
        raise Exception(response.text)
    return response.json()


def get_all_accessible_posts_from_tournament(
    tournament_id: int | str, page_size: int
) -> list[dict]:
    all_posts: list[dict] = []
    offset = 0
    seen_post_ids: set[int] = set()

    while True:
        page = list_posts_from_tournament_all_accessible(
            tournament_id=tournament_id,
            offset=offset,
            count=page_size,
        )
        results = page.get("results", [])
        if not results:
            break

        for post in results:
            post_id = int(post.get("id", -1))
            if post_id in seen_post_ids:
                continue
            seen_post_ids.add(post_id)
            all_posts.append(post)

        if len(results) < page_size:
            break
        offset += page_size

    return all_posts


def get_question_payloads(posts: list[dict]) -> list[tuple[int, dict]]:
    payloads: list[tuple[int, dict]] = []
    for post in posts:
        post_id = int(post["id"])
        question = post.get("question") or {}
        if not question:
            continue
        payloads.append((post_id, question))

    return payloads


def normalize_domain(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or "unknown"


def print_rows_table(rows: list[tuple[int, int, str, str, str]]) -> None:
    headers = ["post_id", "question_id", "url_type", "domain", "url"]
    table_rows = [
        [str(post_id), str(question_id), url_type, domain, url]
        for post_id, question_id, url_type, domain, url in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        if table_rows
        else len(headers[i])
        for i in range(len(headers))
    ]

    def render(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    divider = "-+-".join("-" * width for width in widths)
    print(render(headers))
    print(divider)
    for row in table_rows:
        print(render(row))


def print_grouped_urls(rows: list[tuple[int, int, str, str, str]]) -> None:
    urls_by_type: dict[str, set[str]] = {}
    for _post_id, _question_id, url_type, _domain, url in rows:
        if url_type not in urls_by_type:
            urls_by_type[url_type] = set()
        urls_by_type[url_type].add(url)

    print("\nURLs grouped by type:")
    for url_type in sorted(urls_by_type.keys()):
        urls = sorted(urls_by_type[url_type])
        print(f"\n[{url_type}] ({len(urls)} unique URLs)")
        for url in urls:
            print(f"- {url}")


async def main(args: argparse.Namespace) -> None:
    tournament_id = resolve_tournament_id(args.tournament)

    print(f"Fetching posts from tournament: {tournament_id}")
    posts = get_all_accessible_posts_from_tournament(tournament_id, args.page_size)
    print(f"Found {len(posts)} accessible posts.")

    question_payloads = get_question_payloads(posts)
    print(f"Prepared {len(question_payloads)} question payload(s).")

    url_type_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    rows: list[tuple[int, int, str, str, str]] = []

    for post_id, question in question_payloads:
        question_id = int(question.get("id", -1))
        urls = extract_resolution_urls(
            resolution_criteria=str(question.get("resolution_criteria", "")),
            fine_print=str(question.get("fine_print", "")),
            description=str(question.get("description", "")),
        )
        for url in urls:
            url_type = classify_url(url)
            domain = normalize_domain(url)
            rows.append((post_id, question_id, url_type, domain, url))
            url_type_counts[url_type] += 1
            domain_counts[domain] += 1

    print()
    if rows:
        print_rows_table(rows)
    else:
        print("No URLs found in resolution criteria, fine print, or description.")

    print("\nURL type counts:")
    if url_type_counts:
        for url_type, count in url_type_counts.most_common():
            print(f"- {url_type}: {count}")
    else:
        print("- none")

    print("\nTop domains:")
    if domain_counts:
        for domain, count in domain_counts.most_common(10):
            print(f"- {domain}: {count}")
    else:
        print("- none")

    if rows:
        print_grouped_urls(rows)


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))
