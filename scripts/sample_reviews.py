#!/usr/bin/env python3
"""Sample Amazon All Beauty reviews into a smaller JSONL subset.

This script reads the compressed :code:`All_Beauty.jsonl.gz` dataset and
performs reservoir sampling to produce a reproducible subset.  The output is
stored as newline-delimited JSON records that mirror the structure of the
original dataset.

Example
-------
    python scripts/sample_reviews.py \
        --source data/raw/All_Beauty.jsonl.gz \
        --output data/raw/all_beauty_reviews_10k.jsonl \
        --sample-size 10000

The script requires only the Python standard library and can therefore be used
in restricted environments.  If the source file contains a single JSON array
(rather than newline-delimited JSON), it is automatically flattened so that the
sampling logic still works.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
from pathlib import Path
from typing import Iterable, Iterator, List, MutableMapping


def _open_text(path: Path) -> Iterator[str]:
    """Yield lines from ``path`` transparently handling gzip compression."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                yield line
    else:
        with path.open("rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                yield line


def _iter_records(path: Path) -> Iterator[MutableMapping]:
    """Iterate over JSON records in ``path``.

    The helper automatically supports JSONL and JSON-array formats.
    """

    buffer: List[str] = []
    for raw_line in _open_text(path):
        line = raw_line.strip()
        if not line:
            continue
        buffer.append(raw_line)
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        else:
            yield record

    # If no valid records have been yielded yet, the file might be a JSON array.
    if not buffer:
        return

    if buffer and buffer[0].lstrip().startswith("["):
        text = "".join(buffer)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(
                "Unable to parse the JSON dataset. Ensure it is valid JSONL or a JSON array."
            ) from exc
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, MutableMapping):
                    yield item


def _reservoir_sample(records: Iterable[MutableMapping], sample_size: int, *, seed: int | None) -> List[MutableMapping]:
    """Return ``sample_size`` items sampled uniformly using reservoir sampling."""

    rng = random.Random(seed)
    sample: List[MutableMapping] = []

    for index, record in enumerate(records):
        if index < sample_size:
            sample.append(record)
        else:
            position = rng.randint(0, index)
            if position < sample_size:
                sample[position] = record

    if len(sample) < sample_size:
        raise ValueError(
            f"Requested sample of {sample_size} items, but dataset only contained {len(sample)} valid records."
        )

    return sample


def _write_jsonl(records: Iterable[MutableMapping], destination: Path) -> None:
    """Persist records to ``destination`` as newline-delimited JSON."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wt", encoding="utf-8") as handle:
        for record in records:
            json_record = json.dumps(record, ensure_ascii=False)
            handle.write(json_record + "\n")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/All_Beauty.jsonl.gz"),
        help="Path to the All_Beauty dataset (JSONL or JSON array, optionally gzipped).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/all_beauty_reviews_10k.jsonl"),
        help="Output path for the sampled reviews in JSONL format.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Number of reviews to sample. Defaults to 10,000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    records = _iter_records(args.source)
    sample = _reservoir_sample(records, args.sample_size, seed=args.seed)
    _write_jsonl(sample, args.output)

    print(
        f"Wrote {len(sample)} sampled reviews to {args.output} (seed={args.seed})."
    )


if __name__ == "__main__":
    main()
