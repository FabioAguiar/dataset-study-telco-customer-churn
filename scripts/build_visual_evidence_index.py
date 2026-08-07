"""Deterministic, non-generating index of docs/images/*.png visual evidence.

This module never renders or regenerates a chart. It only enumerates
already-existing PNG assets, classifies each one by filename pattern into a
logical role and chart kind, hashes its bytes, and writes a machine-readable
index. ``scripts/export_figures.py`` is an unrelated, empty (0-byte) file and
is never invoked by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping


VISUAL_EVIDENCE_INDEX_CONTRACT_VERSION: Final[str] = "visual-evidence-index.v1"
DATASET_SLUG: Final[str] = "telco-customer-churn"
DEFAULT_IMAGES_DIR: Final[str] = "docs/images"
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

_NOTEBOOK: Final[str] = "notebooks/01_data_understanding_and_exploration.ipynb"


class VisualEvidenceIndexError(ValueError):
    """Raised when a discovered asset cannot be classified or referenced."""


@dataclass(frozen=True, slots=True)
class VisualClassification:
    """Deterministic filename-pattern classification for one visual asset."""

    logical_role: str
    chart_kind: str
    source_artifact_or_data_reference: str
    method: str


def _notebook_ref(section: str) -> str:
    return f"{_NOTEBOOK}#{section}"


# Ordered from most specific to least specific. The first matching pattern
# wins, so multi-field prefixes (``feature_relationships_``,
# ``key_exploratory_``, and similar section-summary families) are checked
# before the generic per-feature suffix rules.
_RULES: Final[tuple[tuple[re.Pattern[str], VisualClassification], ...]] = (
    (
        re.compile(r"^churn_target_class_distribution$"),
        VisualClassification(
            "target_distribution",
            "bar_chart",
            _notebook_ref("Section-10-Target-Distribution"),
            "matplotlib bar chart of the target class counts, generated "
            "directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r"^feature_relationships_.*_heatmap$"),
        VisualClassification(
            "feature_relationship_heatmap",
            "heatmap",
            _notebook_ref("Section-13-Feature-Relationships"),
            "matplotlib heatmap of a pairwise association matrix (Cramer's "
            "V, eta-squared, or Pearson correlation), generated directly in "
            "the notebook cell.",
        ),
    ),
    (
        re.compile(r"^feature_to_target_.*_ranking$"),
        VisualClassification(
            "feature_to_target_ranking",
            "bar_chart",
            _notebook_ref("Section-14-Feature-to-Target-Relationships"),
            "matplotlib ranked bar chart of feature-to-target association "
            "or effect-size metrics, generated directly in the notebook "
            "cell.",
        ),
    ),
    (
        re.compile(r"^key_exploratory_.*"),
        VisualClassification(
            "exploratory_insight_summary",
            "summary_matrix",
            _notebook_ref("Section-17-Key-Exploratory-Insights"),
            "matplotlib summary chart consolidating "
            "scripts/consolidate_exploratory_insights.py insight records, "
            "generated directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r"^initial_data_quality_.*"),
        VisualClassification(
            "data_quality_summary",
            "summary_chart",
            _notebook_ref("Section-16-Initial-Data-Quality-Findings"),
            "matplotlib summary chart consolidating "
            "scripts/consolidate_quality_findings.py findings, generated "
            "directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r"^potential_data_leakage_.*"),
        VisualClassification(
            "leakage_summary",
            "governance_matrix",
            _notebook_ref("Section-15-Potential-Data-Leakage"),
            "matplotlib governance matrix or summary chart consolidating "
            "scripts/analyze_leakage.py audit output, generated directly in "
            "the notebook cell.",
        ),
    ),
    (
        re.compile(r"^preparation_decisions_.*"),
        VisualClassification(
            "preparation_decision_summary",
            "summary_chart",
            _notebook_ref("Section-18-Preparation-Decisions"),
            "matplotlib summary chart consolidating "
            "scripts/record_preparation_decisions.py decisions, generated "
            "directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r".*_vs_.*_contingency_heatmap$"),
        VisualClassification(
            "categorical_pairwise_contingency_heatmap",
            "heatmap",
            _notebook_ref("Section-13-Feature-Relationships"),
            "matplotlib heatmap of a pairwise categorical contingency "
            "table, generated directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r".*_churn_rate_by_quantile$"),
        VisualClassification(
            "numerical_feature_churn_rate_by_quantile",
            "bar_chart",
            _notebook_ref("Section-14-Feature-to-Target-Relationships"),
            "matplotlib bar chart of churn rate by numerical-feature "
            "quantile bin, generated directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r".*_distribution_histogram$"),
        VisualClassification(
            "numerical_feature_distribution",
            "histogram",
            _notebook_ref("Section-11-Numerical-Feature-Exploration"),
            "matplotlib histogram of a numerical feature's distribution, "
            "generated directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r".*_iqr_boxplot$"),
        VisualClassification(
            "numerical_feature_outlier_boxplot",
            "boxplot",
            _notebook_ref("Section-11-Numerical-Feature-Exploration"),
            "matplotlib boxplot highlighting IQR-based outlier candidates "
            "for a numerical feature, generated directly in the notebook "
            "cell.",
        ),
    ),
    (
        re.compile(r".*_by_churn_boxplot$"),
        VisualClassification(
            "numerical_feature_vs_target_boxplot",
            "boxplot",
            _notebook_ref("Section-14-Feature-to-Target-Relationships"),
            "matplotlib boxplot of a numerical feature split by target "
            "class, generated directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r".*_vs_.*_boxplot$"),
        VisualClassification(
            "categorical_vs_numerical_boxplot",
            "boxplot",
            _notebook_ref("Section-13-Feature-Relationships"),
            "matplotlib boxplot of a numerical feature grouped by a "
            "categorical feature, generated directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r".*_vs_.*_scatter$"),
        VisualClassification(
            "numerical_relationship_scatter",
            "scatter_plot",
            _notebook_ref("Section-13-Feature-Relationships"),
            "matplotlib scatter plot of a numerical-feature relationship, "
            "generated directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r".*_churn_rate_by_category$"),
        VisualClassification(
            "categorical_feature_churn_rate",
            "bar_chart",
            _notebook_ref("Section-14-Feature-to-Target-Relationships"),
            "matplotlib bar chart of churn rate by categorical-feature "
            "value, generated directly in the notebook cell.",
        ),
    ),
    (
        re.compile(r".*_category_distribution$"),
        VisualClassification(
            "categorical_feature_distribution",
            "bar_chart",
            _notebook_ref("Section-12-Categorical-Feature-Exploration"),
            "matplotlib bar chart of a categorical feature's value "
            "distribution, generated directly in the notebook cell.",
        ),
    ),
)


def classify_visual(stem: str) -> VisualClassification:
    """Return the deterministic classification for one asset filename stem."""
    for pattern, classification in _RULES:
        if pattern.match(stem):
            return classification
    raise VisualEvidenceIndexError(
        f"No visual-evidence classification rule matched: {stem!r}"
    )


def _relative_posix(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    if relative == Path(".") or ".." in relative.parts:
        raise VisualEvidenceIndexError(
            f"Invalid project-relative asset path: {relative.as_posix()}"
        )
    return relative.as_posix()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _title_from_stem(stem: str) -> str:
    return stem.replace("_", " ").title()


def build_visual_evidence_index(
    *,
    project_root: str | Path,
    images_dir: str | Path = DEFAULT_IMAGES_DIR,
    dataset_slug: str = DATASET_SLUG,
    contract_version: str = VISUAL_EVIDENCE_INDEX_CONTRACT_VERSION,
) -> dict[str, Any]:
    """Build a deterministic visual-evidence index over existing PNG assets.

    No image is opened, rendered, or regenerated. Every asset already
    present under ``images_dir`` is hashed and classified by filename
    pattern; any filename that matches no known pattern is reported under
    ``excluded_assets`` with an explicit reason rather than silently
    dropped or fabricated.
    """
    root = Path(project_root).expanduser().resolve()
    directory = (root / Path(images_dir)).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise VisualEvidenceIndexError(
            "images_dir must remain inside the project root."
        ) from exc
    if not directory.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    png_paths = sorted(directory.glob("*.png"), key=lambda path: path.name)

    visuals: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for path in png_paths:
        stem = path.stem
        try:
            classification = classify_visual(stem)
        except VisualEvidenceIndexError as exc:
            excluded.append({"asset": path.name, "reason": str(exc)})
            continue

        title = _title_from_stem(stem)
        visuals.append(
            {
                "visual_id": stem,
                "logical_role": classification.logical_role,
                "chart_kind": classification.chart_kind,
                "title": title,
                "caption": (
                    f"{title} "
                    f"({classification.logical_role.replace('_', ' ')})."
                ),
                "source_artifact_or_data_reference": (
                    classification.source_artifact_or_data_reference
                ),
                "method": classification.method,
                "relative_asset_path": _relative_posix(root, path),
                "sha256": _sha256_file(path),
                "public_suitability": True,
                "public_suitability_reason": (
                    "Aggregate statistical or distributional chart with no "
                    "per-record identifiers; no customerID axis or "
                    "row-level label is plotted."
                ),
                "provenance": {
                    "producer_notebook": _NOTEBOOK,
                    "generation_mechanism": (
                        "matplotlib figure saved directly by a notebook "
                        "cell; scripts/export_figures.py is an empty, "
                        "unused 0-byte file and is not the producer."
                    ),
                },
            }
        )

    return {
        "schema_version": VISUAL_EVIDENCE_INDEX_CONTRACT_VERSION,
        "artifact_type": "visual_evidence_index",
        "contract_version": contract_version,
        "dataset_slug": dataset_slug,
        "source_directory": _relative_posix(root, directory),
        "total_assets_scanned": len(png_paths),
        "total_assets_indexed": len(visuals),
        "excluded_assets": excluded,
        "visuals": visuals,
    }


def write_visual_evidence_index(
    index: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> Path:
    """Write the index as pretty-printed, deterministic JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic visual-evidence index over "
            "docs/images/*.png without generating any new chart."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root directory (default: repository root).",
    )
    parser.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help="Project-relative images directory (default: docs/images).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(DEFAULT_IMAGES_DIR, "visual-evidence-index.json"),
        help=(
            "Project-relative output path "
            "(default: docs/images/visual-evidence-index.json)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()

    index = build_visual_evidence_index(
        project_root=root,
        images_dir=args.images_dir,
    )
    destination = write_visual_evidence_index(index, output_path=root / args.output)

    print(f"Indexed {index['total_assets_indexed']} of {index['total_assets_scanned']} assets.")
    print(f"Wrote: {destination.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
