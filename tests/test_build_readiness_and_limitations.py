"""Tests for the consolidated readiness-and-limitations builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_readiness_and_limitations import (
    TELCO_LIMITATIONS,
    build_readiness_and_limitations,
    write_readiness_and_limitations,
)


def test_default_artifact_has_ten_limitations() -> None:
    artifact = build_readiness_and_limitations()

    assert len(artifact["limitations"]) == 10
    assert artifact["dataset_slug"] == "telco-customer-churn"
    assert artifact["schema_version"] == "readiness-and-limitations.v1"


def test_limitations_and_readiness_are_separate_top_level_concepts() -> None:
    artifact = build_readiness_and_limitations()

    assert "limitations" in artifact
    assert "readiness" in artifact
    assert isinstance(artifact["limitations"], list)
    assert isinstance(artifact["readiness"], dict)
    assert not isinstance(artifact["readiness"], bool)


def test_readiness_is_not_collapsed_into_one_boolean() -> None:
    artifact = build_readiness_and_limitations()
    readiness = artifact["readiness"]

    required_fields = {
        "analysis_completeness",
        "educational_research_scope",
        "modeling_completeness",
        "final_test_status",
        "inference_demo_status",
        "operational_deployment_readiness",
        "known_limitations_ref",
        "known_unsupported_uses",
        "remaining_blockers",
    }
    assert required_fields.issubset(readiness.keys())
    assert isinstance(readiness["analysis_completeness"], dict)
    assert isinstance(readiness["remaining_blockers"], list)


def test_operational_deployment_readiness_defaults_to_false() -> None:
    artifact = build_readiness_and_limitations()

    assert artifact["readiness"]["operational_deployment_readiness"] is False


def test_operational_deployment_readiness_cannot_be_elevated_to_true() -> None:
    readiness = {
        "analysis_completeness": {},
        "educational_research_scope": {},
        "modeling_completeness": {},
        "final_test_status": {},
        "inference_demo_status": {},
        "operational_deployment_readiness": True,
        "known_limitations_ref": {},
        "known_unsupported_uses": [],
        "remaining_blockers": [],
    }

    with pytest.raises(ValueError, match="must not be elevated"):
        build_readiness_and_limitations(readiness=readiness)


def test_readiness_requires_explicit_operational_deployment_readiness_field() -> None:
    readiness = {
        "analysis_completeness": {},
        "educational_research_scope": {},
        "modeling_completeness": {},
        "final_test_status": {},
        "inference_demo_status": {},
        "known_limitations_ref": {},
        "known_unsupported_uses": [],
        "remaining_blockers": [],
    }

    with pytest.raises(ValueError, match="operational_deployment_readiness"):
        build_readiness_and_limitations(readiness=readiness)


def test_readiness_pointers_reference_artifacts_by_path_and_field() -> None:
    artifact = build_readiness_and_limitations()
    readiness = artifact["readiness"]

    completeness_ref = readiness["analysis_completeness"]["evidence_ref"]
    assert completeness_ref["artifact_path"].endswith("quality-evidence.json")
    assert completeness_ref["field"]

    blocker_refs = readiness["remaining_blockers"]
    assert all("evidence_ref" in blocker for blocker in blocker_refs)


def test_readiness_pointers_do_not_duplicate_full_payloads() -> None:
    artifact = build_readiness_and_limitations()
    readiness = artifact["readiness"]

    # Pointer objects stay small (path + field + short note), never a full
    # copy of an upstream artifact's payload.
    for key in (
        "analysis_completeness",
        "modeling_completeness",
        "final_test_status",
        "inference_demo_status",
    ):
        assert set(readiness[key].keys()) <= {"status", "evidence_ref", "note"}


def test_limitation_ids_are_unique_and_referenced_by_readiness() -> None:
    artifact = build_readiness_and_limitations()
    limitation_ids = [item["limitation_id"] for item in artifact["limitations"]]

    assert len(limitation_ids) == len(set(limitation_ids))
    referenced = set(
        artifact["readiness"]["known_limitations_ref"]["limitation_ids"]
    )
    assert referenced == set(limitation_ids)


def test_custom_limitations_can_be_supplied() -> None:
    custom = [
        {
            "limitation_id": "LIM-CUSTOM-1",
            "statement": "Synthetic test limitation.",
            "category": "test",
            "blocking": False,
        }
    ]
    artifact = build_readiness_and_limitations(limitations=custom)

    assert artifact["limitations"] == custom


def test_default_limitations_constant_matches_readme_bullet_count() -> None:
    assert len(TELCO_LIMITATIONS) == 10


def test_artifact_is_json_serializable() -> None:
    artifact = build_readiness_and_limitations()

    json.dumps(artifact)


def test_write_readiness_and_limitations_produces_valid_json(tmp_path: Path) -> None:
    artifact = build_readiness_and_limitations()
    output_path = tmp_path / "readiness-and-limitations.json"

    write_readiness_and_limitations(artifact, output_path=output_path)

    reloaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert reloaded == artifact
