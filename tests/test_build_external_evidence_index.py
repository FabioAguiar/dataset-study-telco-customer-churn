"""Tests for the top-level external-evidence discovery index builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_external_evidence_index import (
    ExternalEvidenceIndexError,
    build_external_evidence_index,
    write_external_evidence_index,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_requires_revision_or_reason(tmp_path: Path) -> None:
    with pytest.raises(ExternalEvidenceIndexError, match="unavailable_reason"):
        build_external_evidence_index(project_root=tmp_path)


def test_missing_artifacts_are_reported_not_fabricated(tmp_path: Path) -> None:
    index = build_external_evidence_index(
        project_root=tmp_path,
        repository_revision_unavailable_reason="No git repository in this fixture.",
    )

    assert len(index["missing_artifacts"]) == len(index["evidence_inventory"])
    for entry in index["evidence_inventory"]:
        assert entry["status"] == "missing"
        assert entry["sha256"] is None
        assert entry["artifact_type"] is None


def test_present_artifact_is_hashed_and_type_extracted(tmp_path: Path) -> None:
    _write_json(
        tmp_path
        / "artifacts"
        / "preparation"
        / "telco-customer-churn"
        / "preparation-manifest.json",
        {"artifact_type": "preparation_manifest", "schema_version": "tabular-preparation.v1"},
    )

    index = build_external_evidence_index(
        project_root=tmp_path,
        repository_revision_unavailable_reason="No git repository in this fixture.",
    )

    entry = next(
        item
        for item in index["evidence_inventory"]
        if item["logical_role"] == "source_identity_and_preparation"
    )
    assert entry["status"] == "present"
    assert entry["artifact_type"] == "preparation_manifest"
    assert entry["artifact_version"] == "tabular-preparation.v1"
    expected_hash = hashlib.sha256(
        (
            tmp_path
            / "artifacts"
            / "preparation"
            / "telco-customer-churn"
            / "preparation-manifest.json"
        ).read_bytes()
    ).hexdigest()
    assert entry["sha256"] == expected_hash


def test_index_declares_it_is_not_an_atlas_manifest(tmp_path: Path) -> None:
    index = build_external_evidence_index(
        project_root=tmp_path,
        repository_revision_unavailable_reason="No git repository in this fixture.",
    )

    assert index["not_an_atlas_runtime_manifest"] is True
    assert index["not_an_atlas_release_manifest"] is True
    assert index["not_a_dataset_integration_authoring_manifest"] is True
    assert index["not_an_external_handoff_runtime_dependency"] is True


def test_index_never_contains_absolute_paths(tmp_path: Path) -> None:
    index = build_external_evidence_index(
        project_root=tmp_path,
        repository_revision_unavailable_reason="No git repository in this fixture.",
    )

    for entry in index["evidence_inventory"]:
        assert not Path(entry["relative_path"]).is_absolute()
        assert str(tmp_path) not in entry["relative_path"]


def test_index_does_not_copy_artifact_payloads(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "docs" / "images" / "visual-evidence-index.json",
        {
            "artifact_type": "visual_evidence_index",
            "schema_version": "visual-evidence-index.v1",
            "visuals": [{"visual_id": "x"} for _ in range(50)],
        },
    )

    index = build_external_evidence_index(
        project_root=tmp_path,
        repository_revision_unavailable_reason="No git repository in this fixture.",
    )

    entry = next(
        item
        for item in index["evidence_inventory"]
        if item["logical_role"] == "visual_evidence"
    )
    assert "visuals" not in entry
    assert set(entry.keys()) == {
        "logical_role",
        "relative_path",
        "artifact_type",
        "artifact_version",
        "status",
        "sha256",
        "input_references",
    }


def test_provenance_carries_repository_revision_when_supplied(tmp_path: Path) -> None:
    index = build_external_evidence_index(
        project_root=tmp_path,
        repository_revision="a" * 40,
        generation_timestamp="2026-08-07T00:00:00+00:00",
    )

    assert index["provenance"]["repository_revision"] == "a" * 40
    assert index["provenance"]["repository_revision_unavailable_reason"] is None
    assert index["provenance"]["generation_timestamp"] == "2026-08-07T00:00:00+00:00"
    assert index["provenance"]["logical_producer_project_id"] == (
        "dataset-study-telco-customer-churn"
    )


def test_input_references_reflect_declared_dependencies(tmp_path: Path) -> None:
    index = build_external_evidence_index(
        project_root=tmp_path,
        repository_revision_unavailable_reason="No git repository in this fixture.",
    )

    leakage_entry = next(
        item for item in index["evidence_inventory"] if item["logical_role"] == "leakage"
    )
    assert "feature_semantics" in leakage_entry["input_references"]
    assert "split_identity" in leakage_entry["input_references"]


def test_write_external_evidence_index_produces_valid_json(tmp_path: Path) -> None:
    index = build_external_evidence_index(
        project_root=tmp_path,
        repository_revision_unavailable_reason="No git repository in this fixture.",
    )
    output_path = tmp_path / "artifacts" / "telco-customer-churn" / "external-evidence-index.json"

    write_external_evidence_index(index, output_path=output_path)

    reloaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert reloaded == index


def test_malformed_json_artifact_is_reported_present_with_null_type(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "artifacts"
        / "preparation"
        / "telco-customer-churn"
        / "preparation-manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    index = build_external_evidence_index(
        project_root=tmp_path,
        repository_revision_unavailable_reason="No git repository in this fixture.",
    )

    entry = next(
        item
        for item in index["evidence_inventory"]
        if item["logical_role"] == "source_identity_and_preparation"
    )
    assert entry["status"] == "present"
    assert entry["artifact_type"] is None
    assert entry["sha256"] is not None
