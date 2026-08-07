"""Tests for the deterministic docs/images visual-evidence indexer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_visual_evidence_index import (
    VisualEvidenceIndexError,
    build_visual_evidence_index,
    classify_visual,
    write_visual_evidence_index,
)


def _make_png(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.mark.parametrize(
    ("stem", "expected_role"),
    [
        ("churn_target_class_distribution", "target_distribution"),
        (
            "feature_relationships_numerical_correlation_heatmap",
            "feature_relationship_heatmap",
        ),
        (
            "feature_to_target_numerical_effect_size_ranking",
            "feature_to_target_ranking",
        ),
        ("key_exploratory_insights_theme_summary", "exploratory_insight_summary"),
        ("initial_data_quality_findings_priority_matrix", "data_quality_summary"),
        ("potential_data_leakage_disposition_summary", "leakage_summary"),
        ("preparation_decisions_execution_plan", "preparation_decision_summary"),
        (
            "internet_service_vs_device_protection_contingency_heatmap",
            "categorical_pairwise_contingency_heatmap",
        ),
        ("tenure_churn_rate_by_quantile", "numerical_feature_churn_rate_by_quantile"),
        ("tenure_distribution_histogram", "numerical_feature_distribution"),
        ("tenure_iqr_boxplot", "numerical_feature_outlier_boxplot"),
        ("tenure_by_churn_boxplot", "numerical_feature_vs_target_boxplot"),
        ("contract_vs_tenure_boxplot", "categorical_vs_numerical_boxplot"),
        ("tenure_vs_monthly_charges_scatter", "numerical_relationship_scatter"),
        ("contract_churn_rate_by_category", "categorical_feature_churn_rate"),
        ("contract_category_distribution", "categorical_feature_distribution"),
    ],
)
def test_classify_visual_matches_expected_role(stem: str, expected_role: str) -> None:
    assert classify_visual(stem).logical_role == expected_role


def test_classify_visual_rejects_unknown_pattern() -> None:
    with pytest.raises(VisualEvidenceIndexError, match="No visual-evidence"):
        classify_visual("totally_unrecognized_chart_name")


def test_build_visual_evidence_index_covers_every_asset(tmp_path: Path) -> None:
    project = tmp_path / "study"
    images_dir = project / "docs" / "images"
    _make_png(images_dir / "churn_target_class_distribution.png", b"a")
    _make_png(images_dir / "tenure_category_distribution.png", b"b")
    _make_png(images_dir / "tenure_churn_rate_by_category.png", b"c")

    index = build_visual_evidence_index(project_root=project)

    assert index["total_assets_scanned"] == 3
    assert index["total_assets_indexed"] == 3
    assert index["excluded_assets"] == []
    assert {visual["visual_id"] for visual in index["visuals"]} == {
        "churn_target_class_distribution",
        "tenure_category_distribution",
        "tenure_churn_rate_by_category",
    }


def test_build_visual_evidence_index_reports_unmatched_assets_explicitly(
    tmp_path: Path,
) -> None:
    project = tmp_path / "study"
    images_dir = project / "docs" / "images"
    _make_png(images_dir / "churn_target_class_distribution.png", b"a")
    _make_png(images_dir / "mystery_chart.png", b"b")

    index = build_visual_evidence_index(project_root=project)

    assert index["total_assets_scanned"] == 2
    assert index["total_assets_indexed"] == 1
    assert index["excluded_assets"] == [
        {
            "asset": "mystery_chart.png",
            "reason": (
                "No visual-evidence classification rule matched: "
                "'mystery_chart'"
            ),
        }
    ]


def test_build_visual_evidence_index_computes_real_sha256(tmp_path: Path) -> None:
    project = tmp_path / "study"
    images_dir = project / "docs" / "images"
    content = b"deterministic-bytes"
    _make_png(images_dir / "churn_target_class_distribution.png", content)

    index = build_visual_evidence_index(project_root=project)

    expected = hashlib.sha256(content).hexdigest()
    assert index["visuals"][0]["sha256"] == expected


def test_build_visual_evidence_index_uses_relative_paths_only(tmp_path: Path) -> None:
    project = tmp_path / "study"
    images_dir = project / "docs" / "images"
    _make_png(images_dir / "churn_target_class_distribution.png", b"a")

    index = build_visual_evidence_index(project_root=project)

    relative_path = index["visuals"][0]["relative_asset_path"]
    assert not Path(relative_path).is_absolute()
    assert str(project) not in relative_path
    assert relative_path == "docs/images/churn_target_class_distribution.png"


def test_build_visual_evidence_index_marks_aggregate_charts_public_suitable(
    tmp_path: Path,
) -> None:
    project = tmp_path / "study"
    images_dir = project / "docs" / "images"
    _make_png(images_dir / "churn_target_class_distribution.png", b"a")

    index = build_visual_evidence_index(project_root=project)

    assert index["visuals"][0]["public_suitability"] is True
    assert index["visuals"][0]["public_suitability_reason"]


def test_build_visual_evidence_index_is_deterministic(tmp_path: Path) -> None:
    project = tmp_path / "study"
    images_dir = project / "docs" / "images"
    _make_png(images_dir / "churn_target_class_distribution.png", b"a")
    _make_png(images_dir / "tenure_category_distribution.png", b"b")

    first = build_visual_evidence_index(project_root=project)
    second = build_visual_evidence_index(project_root=project)

    assert first == second


def test_build_visual_evidence_index_missing_directory_raises(tmp_path: Path) -> None:
    project = tmp_path / "study"
    project.mkdir()

    with pytest.raises(FileNotFoundError):
        build_visual_evidence_index(project_root=project)


def test_write_visual_evidence_index_produces_valid_json(tmp_path: Path) -> None:
    project = tmp_path / "study"
    images_dir = project / "docs" / "images"
    _make_png(images_dir / "churn_target_class_distribution.png", b"a")
    index = build_visual_evidence_index(project_root=project)

    output_path = project / "docs" / "images" / "visual-evidence-index.json"
    write_visual_evidence_index(index, output_path=output_path)

    reloaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert reloaded == index


def test_all_current_repository_images_are_indexed_without_exclusion() -> None:
    project_root = Path(__file__).resolve().parents[1]
    images_dir = project_root / "docs" / "images"
    if not images_dir.is_dir():
        pytest.skip("docs/images is not present in this checkout.")

    on_disk_count = len(list(images_dir.glob("*.png")))
    if on_disk_count == 0:
        pytest.skip("No PNG assets are present in docs/images.")

    index = build_visual_evidence_index(project_root=project_root)

    assert index["total_assets_scanned"] == on_disk_count
    assert index["total_assets_indexed"] == on_disk_count
    assert index["excluded_assets"] == []
