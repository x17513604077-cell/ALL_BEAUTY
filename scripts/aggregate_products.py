#!/usr/bin/env python3
"""Aggregate product-level metadata from sampled All Beauty reviews.

The script consumes the sampled JSONL file produced by ``sample_reviews.py``
and computes product-level aggregates keyed by ``asin``.  When key attributes
such as ``brand``, ``price`` or ``category`` are missing from the reviews, the
script attempts to enrich the information by crawling the corresponding product
page using :mod:`scripts.crawl_products`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

try:  # pragma: no cover - runtime import safety for script execution
    from . import crawl_products
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from scripts import crawl_products  # type: ignore

@dataclass
class ProductStats:
    asin: str
    review_titles: Counter = field(default_factory=Counter)
    review_brands: Counter = field(default_factory=Counter)
    review_categories: Counter = field(default_factory=Counter)
    prices: List[float] = field(default_factory=list)
    rating_sum: float = 0.0
    rating_count: int = 0

    def add_review(self, review: Mapping) -> None:
        title = (review.get("title") or "").strip()
        if title:
            self.review_titles[title] += 1

        brand = review.get("brand") or review.get("Brand")
        if isinstance(brand, str) and brand.strip():
            self.review_brands[brand.strip()] += 1

        category = review.get("category") or review.get("categories")
        if isinstance(category, list):
            for entry in category:
                if isinstance(entry, str) and entry.strip():
                    self.review_categories[entry.strip()] += 1
        elif isinstance(category, str) and category.strip():
            self.review_categories[category.strip()] += 1

        price = review.get("price")
        if isinstance(price, str):
            price = _parse_price(price)
        elif isinstance(price, (int, float)):
            price = float(price)
        if isinstance(price, float) and math.isfinite(price):
            self.prices.append(price)

        rating = review.get("overall")
        if isinstance(rating, (int, float)) and math.isfinite(rating):
            self.rating_sum += float(rating)
            self.rating_count += 1

    @property
    def average_rating(self) -> Optional[float]:
        if not self.rating_count:
            return None
        return round(self.rating_sum / self.rating_count, 3)

    @property
    def median_price(self) -> Optional[float]:
        if not self.prices:
            return None
        return round(statistics.median(self.prices), 2)

    @property
    def most_common_title(self) -> Optional[str]:
        return _most_common(self.review_titles)

    @property
    def most_common_brand(self) -> Optional[str]:
        return _most_common(self.review_brands)

    @property
    def most_common_category(self) -> Optional[str]:
        return _most_common(self.review_categories)


@dataclass
class AggregatedProduct:
    asin: str
    title: Optional[str]
    brand: Optional[str]
    category: Optional[str]
    price: Optional[float]
    average_rating: Optional[float]
    review_count: int
    brand_source: str
    price_source: str
    category_source: str

    def to_csv_row(self) -> Dict[str, Optional[str]]:
        return {
            "asin": self.asin,
            "title": self.title or "",
            "brand": self.brand or "",
            "category": self.category or "",
            "price": f"{self.price:.2f}" if isinstance(self.price, float) else "",
            "average_rating": f"{self.average_rating:.3f}" if isinstance(self.average_rating, float) else "",
            "review_count": str(self.review_count),
            "brand_source": self.brand_source,
            "price_source": self.price_source,
            "category_source": self.category_source,
        }


def _parse_price(value: str) -> Optional[float]:
    try:
        clean = (
            value.replace("$", "")
            .replace(",", "")
            .replace("USD", "")
            .strip()
        )
        return float(clean)
    except (ValueError, AttributeError):
        return None


def _most_common(counter: Counter) -> Optional[str]:
    if not counter:
        return None
    value, _ = counter.most_common(1)[0]
    return value


def load_reviews(path: Path, limit: Optional[int] = None) -> Iterable[MutableMapping]:
    if not path.exists():
        raise FileNotFoundError(f"Review sample not found: {path}")
    with path.open("rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def aggregate_reviews(reviews: Iterable[Mapping]) -> Dict[str, ProductStats]:
    stats: Dict[str, ProductStats] = {}
    for review in reviews:
        asin = review.get("asin")
        if not asin:
            continue
        bucket = stats.setdefault(asin, ProductStats(asin=asin))
        bucket.add_review(review)
    return stats


def enrich_with_crawler(
    stats: Dict[str, ProductStats],
    *,
    enable_crawl: bool = True,
    rate_limit: float = crawl_products.DEFAULT_RATE_LIMIT,
) -> Dict[str, AggregatedProduct]:
    aggregated: Dict[str, AggregatedProduct] = {}

    missing_asins = []
    for asin, bucket in stats.items():
        aggregated[asin] = AggregatedProduct(
            asin=asin,
            title=bucket.most_common_title,
            brand=bucket.most_common_brand,
            category=bucket.most_common_category,
            price=bucket.median_price,
            average_rating=bucket.average_rating,
            review_count=bucket.rating_count,
            brand_source="reviews" if bucket.most_common_brand else "",
            price_source="reviews" if bucket.median_price is not None else "",
            category_source="reviews" if bucket.most_common_category else "",
        )
        if not bucket.most_common_brand or bucket.median_price is None or not bucket.most_common_category:
            missing_asins.append(asin)

    if not enable_crawl or not missing_asins:
        return aggregated

    crawled = crawl_products.crawl_products(missing_asins, rate_limit=rate_limit)

    for asin, data in crawled.items():
        if asin not in aggregated:
            continue
        record = aggregated[asin]
        if not record.brand and data.get("brand"):
            record.brand = data["brand"]
            record.brand_source = data.get("brand_source", "crawler")
        if record.price is None and isinstance(data.get("price"), (int, float)):
            record.price = float(data["price"])
            record.price_source = data.get("price_source", "crawler")
        if not record.category and data.get("category"):
            category = data["category"]
            if isinstance(category, list):
                record.category = " > ".join(category)
            else:
                record.category = str(category)
            record.category_source = data.get("category_source", "crawler")

    for record in aggregated.values():
        if not record.brand_source:
            record.brand_source = "crawler" if record.brand else "unknown"
        if not record.price_source:
            record.price_source = "crawler" if record.price is not None else "unknown"
        if not record.category_source:
            record.category_source = "crawler" if record.category else "unknown"

    return aggregated


def write_csv(records: Iterable[AggregatedProduct], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wt", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "asin",
            "title",
            "brand",
            "category",
            "price",
            "average_rating",
            "review_count",
            "brand_source",
            "price_source",
            "category_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for product in records:
            writer.writerow(product.to_csv_row())


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/raw/all_beauty_reviews_10k.jsonl"),
        help="Path to the sampled review JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/all_beauty_products_10k.csv"),
        help="Destination CSV for the aggregated product data.",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Optional limit on the number of reviews to process (useful for testing).",
    )
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Skip crawling product pages even when metadata is missing.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=crawl_products.DEFAULT_RATE_LIMIT,
        help="Delay in seconds between crawler requests when enrichment is enabled.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    reviews = list(load_reviews(args.reviews, limit=args.max_reviews))
    stats = aggregate_reviews(reviews)
    aggregated = enrich_with_crawler(
        stats,
        enable_crawl=not args.skip_crawl,
        rate_limit=args.rate_limit,
    )
    write_csv(aggregated.values(), args.output)
    print(
        f"Wrote {len(aggregated)} aggregated products to {args.output}."
    )


if __name__ == "__main__":
    main()
