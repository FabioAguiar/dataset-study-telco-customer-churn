"""Tests for reusable initial data-quality finding consolidation."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from scripts.consolidate_quality_findings import (
    InitialDataQualityConsolidationError,
    consolidate_initial_data_quality_findings,
)


AVAILABLE_FIELDS = (
    "customerID",
    "tenure",
    "TotalCharges",
    "Churn",
)


def _findings() -> list[dict[str, object]]:
    return [
        {
            "finding_id": "DQ-001",
            "domain": "Data type and completeness",
            "title": "TotalCharges requires materialization",
            "affected_fields": ("TotalCharges", "tenure"),
            "severity": "High",
            "status": "Open",
            "disposition": "Must fix",
            "blocking_scope": "Preparation",
            "source_stages": ("6", "8", "11"),
            "required_action": "Materialize validated blanks and convert.",
            "verification": "No blank or non-numeric values remain.",
        },
        {
            "finding_id": "DQ-002",
            "domain": "Leakage governance",
            "title": "Temporal contract is unresolved",
            "affected_fields": (),
            "severity": "Critical",
            "status": "Open",
            "disposition": "External prerequisite",
            "blocking_scope": "External contract",
            "source_stages": ("15",),
            "required_action": "Define the temporal prediction contract.",
            "verification": "All required context fields are resolved.",
        },
        {
            "finding_id": "DQ-003",
            "domain": "Record integrity",
            "title": "Repeated profiles are valid accounts",
            "affected_fields": ("customerID",),
            "severity": "Medium",
            "status": "Accepted",
            "disposition": "Preserve",
            "blocking_scope": "None",
            "source_stages": ("9",),
            "required_action": "Retain all distinct accounts.",
            "verification": "No profile-based deduplication is applied.",
        },
    ]


def _evidence() -> list[dict[str, object]]:
    return [
        {
            "evidence_id": "EVD-001",
            "finding_id": "DQ-001",
            "source_report": "value_quality_report",
            "source_metric": "Blank count",
            "observed_value": 11,
            "expected_value": 0,
            "interpretation": "Hidden blank values require handling.",
        },
        {
            "evidence_id": "EVD-002",
            "finding_id": "DQ-002",
            "source_report": "leakage_report",
            "source_metric": "Modeling ready",
            "observed_value": False,
            "expected_value": True,
            "interpretation": "Temporal context remains incomplete.",
        },
        {
            "evidence_id": "EVD-003",
            "finding_id": "DQ-003",
            "source_report": "duplicate_report",
            "source_metric": "Duplicate identifiers",
            "observed_value": 0,
            "expected_value": 0,
            "interpretation": "Distinct identifiers must be retained.",
        },
    ]


def _actions() -> list[dict[str, object]]:
    return [
        {
            "action_id": "ACT-001",
            "finding_ids": ("DQ-001",),
            "phase": "Before split",
            "action": "Convert TotalCharges under the declared rule.",
            "fit_scope": "Deterministic",
            "blocking": True,
            "status": "Pending",
            "acceptance_criteria": "The prepared column is numeric.",
        },
        {
            "action_id": "ACT-002",
            "finding_ids": ("DQ-002",),
            "phase": "External contract",
            "action": "Resolve observation time and prediction horizon.",
            "fit_scope": "External",
            "blocking": True,
            "status": "Blocked",
            "acceptance_criteria": "Temporal ordering is documented.",
        },
    ]


def _non_issues() -> list[dict[str, object]]:
    return [
        {
            "non_issue_id": "NI-001",
            "domain": "Numerical quality",
            "title": "No IQR outlier treatment is required",
            "affected_fields": ("tenure", "TotalCharges"),
            "source_stages": ("11",),
            "evidence": "No IQR candidate outliers were identified.",
            "disposition": "No action",
            "interpretation": "Do not remove or clip numerical values.",
        }
    ]


def _report(**overrides):
    parameters = {
        "available_fields": AVAILABLE_FIELDS,
        "row_count": 7043,
        "findings": _findings(),
        "evidence": _evidence(),
        "preparation_actions": _actions(),
        "validated_non_issues": _non_issues(),
    }
    parameters.update(overrides)
    return consolidate_initial_data_quality_findings(**parameters)


def test_valid_consolidation_exposes_all_tables() -> None:
    report = _report()

    assert report.is_structurally_valid
    assert report.is_safe_preparation_scope_defined
    assert not report.is_modeling_ready
    assert len(report.findings_frame()) == 3
    assert len(report.evidence_frame()) == 3
    assert len(report.preparation_actions_frame()) == 2
    assert len(report.validated_non_issues_frame()) == 1
    assert len(report.readiness_frame()) == 3


def test_summary_reports_expected_readiness() -> None:
    summary = _report().summary_frame().set_index("Metric")

    assert summary.loc["Dataset rows", "Value"] == 7043
    assert summary.loc["Must-fix findings", "Value"] == 1
    assert bool(summary.loc["Structurally valid", "Value"])
    assert bool(summary.loc["Safe preparation scope defined", "Value"])
    assert not bool(summary.loc["Modeling ready", "Value"])


def test_findings_are_sorted_by_severity_then_id() -> None:
    assert list(_report().findings_frame()["Finding ID"]) == [
        "DQ-002",
        "DQ-001",
        "DQ-003",
    ]


def test_actions_are_sorted_by_phase_then_id() -> None:
    assert list(_report().preparation_actions_frame()["Action ID"]) == [
        "ACT-001",
        "ACT-002",
    ]


def test_multiple_affected_fields_are_preserved() -> None:
    row = _report().findings_frame().set_index("Finding ID").loc["DQ-001"]

    assert row["Affected fields"] == ("TotalCharges", "tenure")
    assert row["Affected field count"] == 2


def test_general_finding_may_have_no_affected_field() -> None:
    row = _report().findings_frame().set_index("Finding ID").loc["DQ-002"]

    assert row["Affected fields"] == ()
    assert row["Affected field count"] == 0


def test_duplicate_finding_id_is_invalid() -> None:
    findings = _findings()
    findings.append(copy.deepcopy(findings[0]))
    report = _report(findings=findings)

    assert "Duplicate finding ID" in set(report.issues_frame()["Issue"])
    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="declared more than once",
    ):
        report.raise_if_invalid()


def test_duplicate_evidence_id_is_invalid() -> None:
    evidence = _evidence()
    evidence.append(copy.deepcopy(evidence[0]))
    report = _report(evidence=evidence)

    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="Evidence ID",
    ):
        report.raise_if_invalid()


def test_duplicate_action_id_is_invalid() -> None:
    actions = _actions()
    actions.append(copy.deepcopy(actions[0]))
    report = _report(preparation_actions=actions)

    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="Action ID",
    ):
        report.raise_if_invalid()


def test_unknown_affected_field_is_invalid() -> None:
    findings = _findings()
    findings[0]["affected_fields"] = ("unknown",)
    report = _report(findings=findings)

    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="is not available",
    ):
        report.raise_if_invalid()


def test_finding_without_evidence_is_invalid() -> None:
    evidence = [row for row in _evidence() if row["finding_id"] != "DQ-001"]
    report = _report(evidence=evidence)

    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="No evidence record",
    ):
        report.raise_if_invalid()


def test_evidence_with_unknown_finding_is_reported() -> None:
    evidence = _evidence()
    evidence[0]["finding_id"] = "DQ-999"
    report = _report(evidence=evidence)

    assert "Unknown finding reference" in set(report.issues_frame()["Issue"])


def test_action_with_unknown_finding_is_reported() -> None:
    actions = _actions()
    actions[0]["finding_ids"] = ("DQ-999",)
    report = _report(preparation_actions=actions)

    assert "Unknown finding reference" in set(report.issues_frame()["Issue"])


def test_actionable_finding_without_action_is_invalid() -> None:
    actions = [row for row in _actions() if "DQ-001" not in row["finding_ids"]]
    report = _report(preparation_actions=actions)

    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="No preparation action",
    ):
        report.raise_if_invalid()


def test_missing_acceptance_criteria_is_invalid() -> None:
    actions = _actions()
    actions[0]["acceptance_criteria"] = ""
    report = _report(preparation_actions=actions)

    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="no measurable acceptance criteria",
    ):
        report.raise_if_invalid()


@pytest.mark.parametrize("severity", ["Urgent", "", None])
def test_invalid_severity_is_reported(severity: object) -> None:
    findings = _findings()
    findings[0]["severity"] = severity
    report = _report(findings=findings)

    assert "Invalid severity" in set(report.issues_frame()["Issue"])


@pytest.mark.parametrize("status", ["Pending", "Done", ""])
def test_invalid_finding_status_is_rejected(status: str) -> None:
    findings = _findings()
    findings[0]["status"] = status
    report = _report(findings=findings)

    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="Unsupported status",
    ):
        report.raise_if_invalid()


def test_invalid_disposition_is_rejected() -> None:
    findings = _findings()
    findings[0]["disposition"] = "Maybe"
    report = _report(findings=findings)

    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="Unsupported disposition",
    ):
        report.raise_if_invalid()


def test_invalid_action_status_is_rejected() -> None:
    actions = _actions()
    actions[0]["status"] = "Open"
    report = _report(preparation_actions=actions)

    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="Unsupported action status",
    ):
        report.raise_if_invalid()


def test_invalid_non_issue_disposition_is_rejected() -> None:
    non_issues = _non_issues()
    non_issues[0]["disposition"] = "Must fix"
    report = _report(validated_non_issues=non_issues)

    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="Unsupported disposition",
    ):
        report.raise_if_invalid()


def test_blockers_frame_links_actions() -> None:
    blockers = _report().blockers_frame().set_index("Finding ID")

    assert blockers.loc["DQ-001", "Linked actions"] == ("ACT-001",)
    assert blockers.loc["DQ-002", "Linked actions"] == ("ACT-002",)


def test_report_exposes_expected_condition_properties() -> None:
    report = _report()

    assert report.has_open_findings
    assert report.has_must_fix_actions
    assert report.has_external_blockers
    assert report.has_modeling_blockers
    assert report.has_validated_non_issues


def test_completed_blocking_actions_and_closed_findings_clear_modeling() -> None:
    findings = _findings()
    findings[0]["status"] = "Controlled"
    findings[1]["status"] = "Controlled"
    actions = _actions()
    for action in actions:
        action["status"] = "Complete"

    report = _report(findings=findings, preparation_actions=actions)

    assert report.is_modeling_ready
    report.raise_if_modeling_not_ready()


def test_modeling_gate_reports_open_blockers() -> None:
    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="unresolved modeling blockers",
    ):
        _report().raise_if_modeling_not_ready()


def test_external_prerequisite_can_be_checked_independently() -> None:
    with pytest.raises(
        InitialDataQualityConsolidationError,
        match="external prerequisites",
    ):
        _report().raise_if_modeling_not_ready(
            require_no_modeling_blockers=False,
            require_blocking_actions_complete=False,
        )


def test_deterministic_and_train_only_actions_are_supported() -> None:
    findings = _findings()
    findings.append(
        {
            "finding_id": "DQ-004",
            "domain": "Pipeline policy",
            "title": "Scaling must be train-only",
            "affected_fields": ("tenure",),
            "severity": "Medium",
            "status": "Open",
            "disposition": "Train-only",
            "blocking_scope": "Training pipeline",
            "source_stages": ("15",),
            "required_action": "Fit scaling on training data.",
            "verification": "No full-dataset fit occurs.",
        }
    )
    evidence = _evidence()
    evidence.append(
        {
            "evidence_id": "EVD-004",
            "finding_id": "DQ-004",
            "source_report": "leakage_report",
            "source_metric": "Pipeline policy",
            "observed_value": True,
            "expected_value": True,
            "interpretation": "Train-only fitting is mandatory.",
        }
    )
    actions = _actions()
    actions.append(
        {
            "action_id": "ACT-003",
            "finding_ids": ("DQ-004",),
            "phase": "Train-only transformation",
            "action": "Fit the scaler inside the training pipeline.",
            "fit_scope": "Training only",
            "blocking": True,
            "status": "Pending",
            "acceptance_criteria": "Validation data is transform-only.",
        }
    )

    report = _report(
        findings=findings,
        evidence=evidence,
        preparation_actions=actions,
    )

    assert report.is_structurally_valid
    assert report.is_safe_preparation_scope_defined


def test_validated_non_issue_without_action_is_valid() -> None:
    report = _report()

    assert report.issues_frame().empty
    assert report.validated_non_issues_frame().iloc[0]["Disposition"] == (
        "No action"
    )


def test_returned_frames_are_defensive_copies() -> None:
    report = _report()
    findings = report.findings_frame()
    findings.loc[0, "Title"] = "changed"

    assert report.findings_frame().loc[0, "Title"] != "changed"


def test_input_declarations_are_not_mutated() -> None:
    findings = _findings()
    evidence = _evidence()
    actions = _actions()
    non_issues = _non_issues()
    before = copy.deepcopy((findings, evidence, actions, non_issues))

    _report(
        findings=findings,
        evidence=evidence,
        preparation_actions=actions,
        validated_non_issues=non_issues,
    )

    assert (findings, evidence, actions, non_issues) == before


def test_results_are_deterministic() -> None:
    first = _report()
    second = _report()

    pd.testing.assert_frame_equal(
        first.findings_frame(),
        second.findings_frame(),
    )
    pd.testing.assert_frame_equal(
        first.evidence_frame(),
        second.evidence_frame(),
    )
    pd.testing.assert_frame_equal(
        first.preparation_actions_frame(),
        second.preparation_actions_frame(),
    )


def test_string_values_are_normalized_without_mutation() -> None:
    findings = _findings()
    findings[0]["finding_id"] = " DQ-001 "
    findings[0]["affected_fields"] = "TotalCharges"

    report = _report(findings=findings)
    row = report.findings_frame().set_index("Finding ID").loc["DQ-001"]

    assert row["Affected fields"] == ("TotalCharges",)


def test_raise_if_invalid_can_disable_selected_requirement() -> None:
    evidence = [row for row in _evidence() if row["finding_id"] != "DQ-003"]
    report = _report(evidence=evidence)

    report.raise_if_invalid(
        require_evidence_for_findings=False,
    )


def test_no_findings_is_structurally_valid_and_modeling_ready() -> None:
    report = consolidate_initial_data_quality_findings(
        available_fields=AVAILABLE_FIELDS,
        row_count=0,
        findings=(),
        evidence=(),
        preparation_actions=(),
        validated_non_issues=(),
    )

    assert report.is_structurally_valid
    assert report.is_safe_preparation_scope_defined
    assert report.is_modeling_ready
    assert report.blockers_frame().empty
