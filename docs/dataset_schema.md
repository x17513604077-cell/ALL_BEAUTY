# ALL_BEAUTY Processed Dataset Schema

This document describes the schema of the final datasets produced by the processing
pipeline (`make data`). Both files live under `data/processed/` and are written in
UTF-8 encoding.

## `products_clean.csv`

| Field | Type | Description |
| --- | --- | --- |
| `asin` | string | Unique Amazon Standard Identification Number for the product. |
| `title` | string | Cleaned product title with extraneous whitespace removed. |
| `brand` | string | Normalised brand name; defaults to `"Unknown"` when not supplied. |
| `category` | string | Primary product category label. |
| `price` | number | Imputed unit price in USD with non-positive values set to the median of observed prices. |
| `average_rating` | number | Average star rating reported in the source catalogue (1–5). |
| `review_count` | integer | Count of user reviews reported in the source catalogue. |
| `brand_source` | string | Provenance of the `brand` value (e.g. `reviews`, `catalogue`, `unknown`). |
| `price_source` | string | Provenance of the `price` value used after imputation. |
| `category_source` | string | Provenance of the `category` value. |

## `reviews_clean.jsonl`

Each line is a JSON object with the following fields.

| Field | Type | Description |
| --- | --- | --- |
| `asin` | string | Product identifier matching `products_clean.csv`. |
| `overall` | number | Star rating provided in the review, clipped to the 1–5 range. |
| `title` | string | Review headline with surrounding whitespace removed. |
| `reviewText` | string | Review body text, normalised for spacing and empty strings dropped. |
| `price` | number or null | Transaction price in USD; backfilled from the catalogue when missing or invalid. |
| `category` | array[string] | Ordered list of product categories derived from review metadata or the catalogue. |
| `brand` | string | Brand associated with the review after enrichment from the product catalogue. |

## `reviews_aug.jsonl`

The augmentation file under `data/augmented/` contains the same columns as
`reviews_clean.jsonl` plus an `augmentation_type` string describing the technique
used (`synonym_replacement`, `positive_emphasis`, or `negative_emphasis`).
