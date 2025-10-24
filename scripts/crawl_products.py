#!/usr/bin/env python3
"""Crawl Amazon product pages to enrich metadata for All Beauty items.

The crawler fetches the public product detail page for each ASIN using a
conservative request cadence.  It extracts brand, price and category
information from embedded JSON-LD blocks as well as selected DOM elements.

The module exposes a ``crawl_products`` function that returns a dictionary keyed
by ASIN.  Each entry includes the scraped attributes along with metadata about
how they were obtained.  The script can also be executed directly by passing a
list of ASINs via the command line.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

DEFAULT_RATE_LIMIT = 2.0  # seconds between requests
PRODUCT_URL_TEMPLATE = "https://www.amazon.com/dp/{asin}"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36"
)


def fetch_html(url: str, *, timeout: float = 10.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="ignore")


def parse_structured_data(html: str) -> Dict[str, Optional[str]]:
    results: Dict[str, Optional[str]] = {}

    script_pattern = re.compile(
        r"<script[^>]+type=\"application/ld\+json\"[^>]*>(.*?)</script>",
        re.IGNORECASE | re.DOTALL,
    )

    for match in script_pattern.finditer(html):
        block = match.group(1).strip()
        if not block:
            continue
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            # Some pages contain multiple JSON objects separated by newlines.
            fragments = [frag for frag in re.split(r"\n+(?=\{)", block) if frag.strip()]
            for fragment in fragments:
                try:
                    payload = json.loads(fragment)
                except json.JSONDecodeError:
                    continue
                _extract_from_payload(payload, results)
            continue
        _extract_from_payload(payload, results)
    return results


def _extract_from_payload(payload, results: Dict[str, Optional[str]]) -> None:
    if isinstance(payload, list):
        for item in payload:
            _extract_from_payload(item, results)
        return

    if not isinstance(payload, dict):
        return

    payload_type = payload.get("@type")
    if payload_type == "Product":
        brand = payload.get("brand")
        if isinstance(brand, dict):
            results.setdefault("brand", brand.get("name") or brand.get("@id"))
        elif isinstance(brand, str):
            results.setdefault("brand", brand)

        offers = payload.get("offers")
        if isinstance(offers, list):
            offers = offers[0] if offers else None
        if isinstance(offers, dict):
            price = offers.get("price")
            if price:
                if isinstance(price, str):
                    price_clean = price.replace("$", "").replace(",", "").strip()
                else:
                    price_clean = price
                try:
                    results.setdefault("price", float(price_clean))
                except (TypeError, ValueError):
                    pass
            currency = offers.get("priceCurrency")
            if currency:
                results.setdefault("currency", currency)
        category = payload.get("category")
        if isinstance(category, list):
            results.setdefault("category", [str(item) for item in category if item])
        elif isinstance(category, str):
            results.setdefault("category", category)

    if payload_type == "BreadcrumbList":
        items = payload.get("itemListElement") or []
        categories: List[str] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            item = entry.get("item")
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str):
                    categories.append(name.strip())
        if categories:
            results.setdefault("category", categories)


def parse_dom_fallback(html: str, results: Dict[str, Optional[str]]) -> None:
    if "brand" not in results:
        brand_match = re.search(r"id=\"bylineInfo\"[^>]*>(.*?)</", html, re.IGNORECASE | re.DOTALL)
        if brand_match:
            brand_text = re.sub(r"<.*?>", "", brand_match.group(1))
            results["brand"] = brand_text.strip()

    if "price" not in results:
        price_match = re.search(
            r"id=\"priceblock_[^\"]+\"[^>]*>\s*\$?([0-9,.]+)",
            html,
            re.IGNORECASE,
        )
        if price_match:
            try:
                results["price"] = float(price_match.group(1).replace(",", ""))
            except ValueError:
                pass

    if "category" not in results:
        breadcrumb_match = re.search(
            r"id=\"wayfinding-breadcrumbs_feature_div\".*?</ul>",
            html,
            re.IGNORECASE | re.DOTALL,
        )
        if breadcrumb_match:
            breadcrumb_html = breadcrumb_match.group(0)
            categories = [
                re.sub(r"<.*?>", "", part).strip()
                for part in re.findall(r"<a[^>]*>(.*?)</a>", breadcrumb_html, re.DOTALL)
            ]
            categories = [cat for cat in categories if cat]
            if categories:
                results["category"] = categories


def crawl_products(asins: Iterable[str], rate_limit: float = DEFAULT_RATE_LIMIT) -> Dict[str, Dict[str, Optional[str]]]:
    asins_list = [asin.strip() for asin in asins if asin and asin.strip()]
    enriched: Dict[str, Dict[str, Optional[str]]] = {}

    for index, asin in enumerate(asins_list):
        url = PRODUCT_URL_TEMPLATE.format(asin=urllib.parse.quote(asin))
        payload: Dict[str, Optional[str]] = {
            "url": url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "brand_source": "",
            "price_source": "",
            "category_source": "",
            "rate_limit_seconds": rate_limit,
        }
        try:
            html = fetch_html(url)
        except urllib.error.URLError as exc:
            payload["error"] = str(exc)
            enriched[asin] = payload
            continue

        results = parse_structured_data(html)
        structured_brand = "brand" in results
        structured_price = "price" in results
        structured_category = "category" in results
        parse_dom_fallback(html, results)

        if "brand" in results and results["brand"]:
            payload["brand"] = results["brand"].strip() if isinstance(results["brand"], str) else results["brand"]
            payload["brand_source"] = "structured_data" if structured_brand else "dom"
        if "price" in results and isinstance(results["price"], (int, float)):
            payload["price"] = float(results["price"])
            payload["price_source"] = "structured_data" if structured_price else "dom"
        if "currency" in results:
            payload["currency"] = results["currency"]
        if "category" in results and results["category"]:
            payload["category"] = results["category"]
            payload["category_source"] = "structured_data" if structured_category else "dom"

        enriched[asin] = payload

        if rate_limit and index < len(asins_list) - 1:
            time.sleep(rate_limit)

    return enriched


def cli(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "asins",
        nargs="+",
        help="One or more ASINs to crawl.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help="Seconds to wait between successive requests.",
    )
    args = parser.parse_args(argv)

    results = crawl_products(args.asins, rate_limit=args.rate_limit)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
