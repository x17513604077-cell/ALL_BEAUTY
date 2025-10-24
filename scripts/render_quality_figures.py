"""Generate visual summaries for experiment one reports.

This utility reads the tabular metrics generated during the cleaning
pipeline and renders lightweight vector charts (SVG by default).  SVG is
text-based so it can be tracked in git without triggering the binary
file restriction that previously blocked the pull request.

When PNG assets are required locally, pass ``--format png`` and optionally
``--export-base64`` to emit ``.b64`` text files that can be downloaded or
copied safely before decoding back into images.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

QUALITY_METRICS_PATH = REPORTS_DIR / "data_quality_metrics.csv"
AUGMENTATION_EVAL_PATH = REPORTS_DIR / "augmentation_eval.csv"


@dataclass
class FigureArtifact:
    """Metadata about a generated figure."""

    path: Path
    description: str


def _ensure_directories() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_quality_metrics() -> pd.DataFrame:
    if not QUALITY_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Quality metrics file not found. Run scripts/clean_data.py first."
        )
    return pd.read_csv(QUALITY_METRICS_PATH)


def _load_augmentation_eval() -> pd.DataFrame:
    if not AUGMENTATION_EVAL_PATH.exists():
        raise FileNotFoundError(
            "Augmentation evaluation file not found. Run scripts/clean_data.py first."
        )
    return pd.read_csv(AUGMENTATION_EVAL_PATH)


def _save_base64(figure_path: Path) -> Path:
    encoded_dir = FIGURES_DIR / "base64"
    encoded_dir.mkdir(exist_ok=True)

    data = figure_path.read_bytes()
    encoded_path = encoded_dir / f"{figure_path.stem}.b64"
    encoded_path.write_text(base64.b64encode(data).decode("utf-8"))
    return encoded_path


def _render_missing_heatmap(df: pd.DataFrame, file_format: str) -> FigureArtifact:
    pivot = df.pivot_table(
        index="column", columns="dataset", values="missing_rate", fill_value=0.0
    )
    pivot = pivot.sort_index()

    plt.figure(figsize=(8, max(4, len(pivot) * 0.3)))
    sns.heatmap(
        pivot.mul(100),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Missing rate (%)"},
    )
    plt.title("Missing rate by dataset & column")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"missing_rate_heatmap.{file_format}"
    plt.savefig(output_path, format=file_format)
    plt.close()
    return FigureArtifact(path=output_path, description="Missing rate heatmap")


def _render_anomaly_barplot(df: pd.DataFrame, file_format: str) -> FigureArtifact:
    summary = (
        df.groupby(["dataset", "column"], as_index=False)["anomaly_count"].sum()
        .sort_values(["dataset", "column"])
    )
    plt.figure(figsize=(10, 4 + summary["column"].nunique() * 0.2))
    sns.barplot(
        data=summary,
        x="anomaly_count",
        y="column",
        hue="dataset",
        palette="Set2",
    )
    plt.xlabel("Anomaly count")
    plt.ylabel("Column")
    plt.title("Anomaly counts per column")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"anomaly_counts.{file_format}"
    plt.savefig(output_path, format=file_format)
    plt.close()
    return FigureArtifact(path=output_path, description="Anomaly count comparison")


def _render_augmentation_scores(df: pd.DataFrame, file_format: str) -> FigureArtifact:
    melted = df.melt(id_vars=["dataset"], var_name="metric", value_name="score")
    plt.figure(figsize=(6, 4))
    sns.barplot(data=melted, x="dataset", y="score", hue="metric", palette="Dark2")
    plt.ylim(0, 1)
    plt.title("Augmentation impact on model performance")
    plt.ylabel("Score")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"augmentation_scores.{file_format}"
    plt.savefig(output_path, format=file_format)
    plt.close()
    return FigureArtifact(path=output_path, description="Augmentation evaluation")


def _export_base64_if_requested(
    figures: Iterable[FigureArtifact], export_base64: bool
) -> List[Path]:
    if not export_base64:
        return []
    generated: List[Path] = []
    for artifact in figures:
        generated.append(_save_base64(artifact.path))
    return generated


def render_figures(file_format: str, export_base64: bool) -> List[FigureArtifact]:
    _ensure_directories()
    quality = _load_quality_metrics()
    augmentation = _load_augmentation_eval()

    figures = [
        _render_missing_heatmap(quality, file_format),
        _render_anomaly_barplot(quality, file_format),
        _render_augmentation_scores(augmentation, file_format),
    ]

    _export_base64_if_requested(figures, export_base64)
    return figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=["svg", "png"],
        default="svg",
        help="Image format for exported figures (default: svg).",
    )
    parser.add_argument(
        "--export-base64",
        action="store_true",
        help=(
            "Generate companion .b64 files with base64-encoded image data. "
            "Useful when sharing through systems that disallow binary uploads."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures = render_figures(args.format, args.export_base64)

    for artifact in figures:
        print(f"Created {artifact.description}: {artifact.path.relative_to(BASE_DIR)}")
        if args.export_base64:
            b64_path = FIGURES_DIR / "base64" / f"{artifact.path.stem}.b64"
            print(
                "  Base64 copy:"
                f" {b64_path.relative_to(BASE_DIR)} (decode with `base64 -d`)."
            )


if __name__ == "__main__":
    main()
