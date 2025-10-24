from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
AUGMENTED_DIR = BASE_DIR / "data" / "augmented"
REPORTS_DIR = BASE_DIR / "reports"

REVIEWS_RAW_PATH = RAW_DATA_DIR / "all_beauty_reviews_10k.jsonl"
PRODUCTS_RAW_PATH = RAW_DATA_DIR / "all_beauty_products_10k.csv"
REVIEWS_CLEAN_PATH = PROCESSED_DIR / "reviews_clean.jsonl"
PRODUCTS_CLEAN_PATH = PROCESSED_DIR / "products_clean.csv"
REVIEWS_AUG_PATH = AUGMENTED_DIR / "reviews_aug.jsonl"
DATA_QUALITY_PATH = REPORTS_DIR / "data_quality_metrics.csv"
AUG_EVAL_PATH = REPORTS_DIR / "augmentation_eval.csv"


def ensure_directories() -> None:
    for directory in (PROCESSED_DIR, AUGMENTED_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews = pd.read_json(REVIEWS_RAW_PATH, lines=True)
    products = pd.read_csv(PRODUCTS_RAW_PATH)
    return reviews, products


def enrich_reviews_with_products(
    reviews: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    product_subset = products[
        ["asin", "brand", "category", "price", "average_rating"]
    ].rename(
        columns={
            "brand": "brand_product",
            "category": "category_product",
            "price": "price_product",
            "average_rating": "average_rating_product",
        }
    )
    enriched = reviews.merge(product_subset, on="asin", how="left")
    return enriched


def to_category_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        # split on common separators
        if ";" in value:
            parts = value.split(";")
        elif "," in value and not value.strip().startswith("["):
            parts = value.split(",")
        else:
            parts = [value]
        return [part.strip() for part in parts if part.strip()]
    return []


def clean_reviews(reviews: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_reviews_with_products(reviews, products)

    cleaned = enriched.copy()
    cleaned["reviewText"] = (
        cleaned["reviewText"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    cleaned.loc[cleaned["reviewText"] == "nan", "reviewText"] = np.nan
    cleaned["title"] = cleaned["title"].astype(str).str.strip()

    cleaned["overall"] = pd.to_numeric(cleaned["overall"], errors="coerce")
    cleaned["overall"] = cleaned["overall"].clip(lower=1.0, upper=5.0)

    cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce")
    cleaned["price_product"] = pd.to_numeric(cleaned["price_product"], errors="coerce")
    cleaned["price"] = cleaned["price"].fillna(cleaned["price_product"])
    cleaned.loc[cleaned["price"] <= 0, "price"] = np.nan

    cleaned["brand"] = cleaned["brand"].fillna(cleaned["brand_product"])
    cleaned["brand"] = cleaned["brand"].fillna("Unknown")
    cleaned["brand"] = cleaned["brand"].astype(str).str.strip()

    cleaned["category"] = cleaned["category"].apply(to_category_list)
    cleaned["category_product"] = cleaned["category_product"].apply(to_category_list)
    cleaned["category"] = cleaned.apply(
        lambda row: row["category"]
        if row["category"]
        else row["category_product"]
        if isinstance(row["category_product"], list)
        else [],
        axis=1,
    )

    cleaned = cleaned.drop(columns=[
        "brand_product",
        "category_product",
        "price_product",
        "average_rating_product",
    ])

    cleaned = cleaned.dropna(subset=["asin", "reviewText", "overall"])
    cleaned = cleaned[cleaned["reviewText"].str.len() > 0]
    cleaned = cleaned.drop_duplicates(subset=["asin", "reviewText", "overall"])

    return cleaned.reset_index(drop=True)


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    cleaned = products.copy()

    cleaned["asin"] = cleaned["asin"].astype(str).str.strip()
    cleaned = cleaned.dropna(subset=["asin"])
    cleaned = cleaned.drop_duplicates(subset=["asin"])

    for col in ["price", "average_rating"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned.loc[cleaned["price"] <= 0, "price"] = np.nan
    cleaned["price"] = cleaned["price"].fillna(cleaned["price"].median())

    cleaned["brand"] = cleaned["brand"].fillna("Unknown").astype(str).str.strip()
    cleaned["category"] = cleaned["category"].fillna("Uncategorized")
    cleaned["category"] = cleaned["category"].astype(str).str.strip()

    cleaned["review_count"] = pd.to_numeric(cleaned["review_count"], errors="coerce").fillna(0)
    cleaned.loc[cleaned["review_count"] < 0, "review_count"] = 0

    cleaned["brand_source"] = cleaned["brand_source"].fillna("unknown")
    cleaned["price_source"] = cleaned["price_source"].fillna("unknown")
    cleaned["category_source"] = cleaned["category_source"].fillna("unknown")

    return cleaned.reset_index(drop=True)


def augment_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    augmented_rows: List[dict] = []

    for _, row in reviews.iterrows():
        text = row["reviewText"]
        if not isinstance(text, str) or not text.strip():
            continue

        base = row.to_dict()

        if "Synthetic" in text:
            synonym_row = base.copy()
            synonym_row["reviewText"] = text.replace("Synthetic", "Artificial")
            synonym_row["augmentation_type"] = "synonym_replacement"
            augmented_rows.append(synonym_row)

        emphasis_row = base.copy()
        if float(row["overall"]) >= 4.0:
            emphasis_row["reviewText"] = f"{text} I would happily purchase it again."
            emphasis_row["augmentation_type"] = "positive_emphasis"
        else:
            emphasis_row["reviewText"] = f"{text} Unfortunately, it did not meet my expectations."
            emphasis_row["augmentation_type"] = "negative_emphasis"
        augmented_rows.append(emphasis_row)

    augmented_df = pd.DataFrame(augmented_rows)
    if augmented_df.empty:
        augmented_df = pd.DataFrame(columns=list(reviews.columns) + ["augmentation_type"])
    else:
        augmented_df["category"] = augmented_df["category"].apply(to_category_list)

    return augmented_df.reset_index(drop=True)


def compute_quality_metrics(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: List[dict] = []

    for name, df in datasets.items():
        for column in df.columns:
            series = df[column]
            missing_rate = float(series.isna().mean())
            anomaly_count = 0

            if pd.api.types.is_numeric_dtype(series):
                anomaly_count += int((series < 0).sum())
            elif series.dtype == object:
                anomaly_count += int(
                    series.apply(
                        lambda x: isinstance(x, str) and not x.strip()
                    ).sum()
                )

            records.append(
                {
                    "dataset": name,
                    "column": column,
                    "missing_rate": round(missing_rate, 4),
                    "anomaly_count": anomaly_count,
                }
            )

    return pd.DataFrame(records)


def _prepare_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    training_df = df[["reviewText", "overall"]].dropna()
    training_df = training_df[training_df["reviewText"].str.strip().astype(bool)]
    training_df = training_df.copy()
    training_df["label"] = (training_df["overall"] >= 4.0).astype(int)
    return training_df


def train_and_evaluate(df: pd.DataFrame) -> dict:
    training_df = _prepare_training_dataframe(df)

    if training_df["label"].nunique() < 2:
        return {"accuracy": np.nan, "f1": np.nan, "note": "insufficient_class_variation"}

    X_train, X_test, y_train, y_test = train_test_split(
        training_df["reviewText"],
        training_df["label"],
        test_size=0.2,
        random_state=42,
        stratify=training_df["label"],
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            (
                "logreg",
                LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions))

    return {"accuracy": round(accuracy, 4), "f1": round(f1, 4)}


def evaluate_augmentation(
    clean_reviews_df: pd.DataFrame, augmented_reviews_df: pd.DataFrame
) -> pd.DataFrame:
    results: List[dict] = []

    clean_metrics = train_and_evaluate(clean_reviews_df)
    clean_metrics["dataset"] = "clean"
    results.append(clean_metrics)

    augmented_combined = clean_reviews_df.copy()
    if not augmented_reviews_df.empty:
        augmented_subset = augmented_reviews_df[clean_reviews_df.columns]
        augmented_combined = pd.concat(
            [clean_reviews_df, augmented_subset], ignore_index=True
        )

    augmented_metrics = train_and_evaluate(augmented_combined)
    augmented_metrics["dataset"] = "clean+augmented"
    results.append(augmented_metrics)

    return pd.DataFrame(results)


def _normalise_for_json(value):
    if isinstance(value, (set, tuple)):
        return [_normalise_for_json(v) for v in value]
    if isinstance(value, list):
        return [_normalise_for_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        if pd.isna(value):
            return None
        return value.item()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    records = []
    for record in df.to_dict(orient="records"):
        normalised = {key: _normalise_for_json(value) for key, value in record.items()}
        records.append(normalised)

    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    ensure_directories()
    reviews_raw, products_raw = load_raw_data()

    reviews_clean = clean_reviews(reviews_raw, products_raw)
    products_clean = clean_products(products_raw)
    reviews_augmented = augment_reviews(reviews_clean)

    quality_metrics = compute_quality_metrics(
        {
            "reviews_clean": reviews_clean,
            "products_clean": products_clean,
            "reviews_augmented": reviews_augmented,
        }
    )
    augmentation_eval = evaluate_augmentation(reviews_clean, reviews_augmented)

    save_jsonl(reviews_clean, REVIEWS_CLEAN_PATH)
    products_clean.to_csv(PRODUCTS_CLEAN_PATH, index=False)
    save_jsonl(reviews_augmented, REVIEWS_AUG_PATH)

    quality_metrics.to_csv(DATA_QUALITY_PATH, index=False)
    augmentation_eval.to_csv(AUG_EVAL_PATH, index=False)

    print(f"Saved cleaned reviews to {REVIEWS_CLEAN_PATH}")
    print(f"Saved cleaned products to {PRODUCTS_CLEAN_PATH}")
    print(f"Saved augmented reviews to {REVIEWS_AUG_PATH}")
    print(f"Saved quality metrics to {DATA_QUALITY_PATH}")
    print(f"Saved augmentation evaluation to {AUG_EVAL_PATH}")


if __name__ == "__main__":
    main()
