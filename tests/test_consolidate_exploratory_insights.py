"""Tests for reusable key-exploratory-insight consolidation."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from scripts.consolidate_exploratory_insights import (
    ExploratoryInsightsConsolidationError,
    consolidate_key_exploratory_insights,
)


AVAILABLE_FIELDS = (
    "tenure",
    "Contract",
    "MonthlyCharges",
    "InternetService",
    "Churn",
)


def _insights() -> list[dict[str, object]]:
    return [
        {
            "insight_id": "INS-001",
            "theme": "Customer lifecycle",
            "title": "Early tenure is associated with higher churn",
            "insight_type": "Pattern",
            "affected_fields": ("tenure", "Churn"),
            "relevance": "High",
            "status": "Observed",
            "source_stages": ("10", "14"),
            "summary": "Churn decreases across tenure quantiles.",
            "modeling_implication": "Evaluate non-linear tenure effects.",
            "interpretation_boundary": "Association does not prove causation.",
        },
        {
            "insight_id": "INS-002",
            "theme": "Contract and retention",
            "title": "Contract categories show strong churn contrasts",
            "insight_type": "Contrast",
            "affected_fields": ("Contract", "Churn"),
            "relevance": "High",
            "status": "Observed",
            "source_stages": ("14",),
            "summary": "Month-to-month customers churn more frequently.",
            "modeling_implication": "Preserve contract categories.",
            "interpretation_boundary": "Contract may be confounded by tenure.",
        },
        {
            "insight_id": "INS-003",
            "theme": "Governance",
            "title": "Temporal availability remains unresolved",
            "insight_type": "Governance limitation",
            "affected_fields": (),
            "relevance": "Contextual",
            "status": "Unresolved",
            "source_stages": ("15", "16"),
            "summary": "The feature snapshot cutoff is not documented.",
            "modeling_implication": "Do not clear modeling yet.",
            "interpretation_boundary": "Data cleaning cannot resolve timing.",
        },
    ]


def _evidence() -> list[dict[str, object]]:
    return [
        {
            "evidence_id": "EVI-001",
            "insight_id": "INS-001",
            "evidence_kind": "Feature-to-target",
            "source_report": "feature_target_report",
            "source_metric": "tenure Cohen's d",
            "observed_value": -0.85,
            "comparison_value": 0.0,
            "direction": "Negative",
            "interpretation": "Higher tenure is associated with lower churn.",
        },
        {
            "evidence_id": "EVI-002",
            "insight_id": "INS-002",
            "evidence_kind": "Categorical contrast",
            "source_report": "feature_target_report",
            "source_metric": "Contract churn-rate spread",
            "observed_value": 0.40,
            "comparison_value": 0.0,
            "direction": "Positive spread",
            "interpretation": "Contract groups have different churn rates.",
        },
        {
            "evidence_id": "EVI-003",
            "insight_id": "INS-003",
            "evidence_kind": "Leakage/governance",
            "source_report": "leakage_report",
            "source_metric": "Modeling ready",
            "observed_value": False,
            "comparison_value": True,
            "direction": "Blocked",
            "interpretation": "Temporal context remains unresolved.",
        },
    ]


def _hypotheses() -> list[dict[str, object]]:
    return [
        {
            "hypothesis_id": "HYP-001",
            "linked_insight_ids": ("INS-001", "INS-002"),
            "title": "Early-tenure monthly customers form a high-risk group",
            "hypothesis": "Tenure and contract interact in churn risk.",
            "status": "Unvalidated",
            "confounding_risks": (
                "MonthlyCharges",
                "InternetService",
            ),
            "required_validation": "Test tenure by Contract interactions.",
            "decision_stage": "Model selection",
        }
    ]


def _actions() -> list[dict[str, object]]:
    return [
        {
            "action_id": "VAL-001",
            "hypothesis_ids": ("HYP-001",),
            "validation_type": "Interaction test",
            "action": "Evaluate a tenure by Contract interaction.",
            "stage": "Model selection",
            "blocking": False,
            "status": "Planned",
            "acceptance_criteria": "The interaction is stable across folds.",
        },
        {
            "action_id": "VAL-002",
            "hypothesis_ids": ("HYP-001",),
            "validation_type": "Cross-validation",
            "action": "Compare segment performance across folds.",
            "stage": "Model evaluation",
            "blocking": False,
            "status": "Planned",
            "acceptance_criteria": "Segment metrics are reproducible.",
        },
    ]


def _limitations() -> list[dict[str, object]]:
    return [
        {
            "limitation_id": "LIM-001",
            "theme": "Governance",
            "title": "Temporal contract is incomplete",
            "limitation_type": "Temporal",
            "affected_fields": (),
            "severity": "Critical",
            "status": "Unresolved",
            "source_stages": ("15", "16"),
            "implication": "Leakage-safe modeling cannot be confirmed.",
            "required_resolution": "Define observation time and horizon.",
        },
        {
            "limitation_id": "LIM-002",
            "theme": "Data sufficiency",
            "title": "Observed features do not determine churn",
            "limitation_type": "Data sufficiency",
            "affected_fields": ("Churn",),
            "severity": "Medium",
            "status": "Accepted",
            "source_stages": ("9", "17"),
            "implication": "Model outputs must remain probabilistic.",
            "required_resolution": "Use calibration and error analysis.",
        },
    ]


def _report(**overrides):
    parameters = {
        "available_fields": AVAILABLE_FIELDS,
        "insights": _insights(),
        "evidence": _evidence(),
        "hypotheses": _hypotheses(),
        "validation_actions": _actions(),
        "limitations": _limitations(),
    }
    parameters.update(overrides)
    return consolidate_key_exploratory_insights(**parameters)


def test_valid_consolidation_exposes_all_tables() -> None:
    report = _report()

    assert report.is_structurally_valid
    assert report.is_ready_for_preparation_decisions
    assert not report.is_ready_for_modeling
    assert len(report.insights_frame()) == 3
    assert len(report.evidence_frame()) == 3
    assert len(report.hypotheses_frame()) == 1
    assert len(report.validation_actions_frame()) == 2
    assert len(report.limitations_frame()) == 2
    assert len(report.readiness_frame()) == 3


def test_summary_reports_expected_counts_and_readiness() -> None:
    summary = _report().summary_frame().set_index("Metric")

    assert summary.loc["Consolidated insights", "Value"] == 3
    assert summary.loc["High-relevance insights", "Value"] == 2
    assert summary.loc["Evidence records", "Value"] == 3
    assert summary.loc["Exploratory hypotheses", "Value"] == 1
    assert not bool(summary.loc["Ready for modeling", "Value"])


def test_insights_are_sorted_by_relevance_then_id() -> None:
    assert list(_report().insights_frame()["Insight ID"]) == [
        "INS-001",
        "INS-002",
        "INS-003",
    ]


def test_hypotheses_include_link_and_action_counts() -> None:
    row = _report().hypotheses_frame().iloc[0]

    assert row["Linked insight count"] == 2
    assert row["Validation action count"] == 2
    assert row["Confounding risk count"] == 2


def test_insights_include_evidence_and_hypothesis_counts() -> None:
    rows = _report().insights_frame().set_index("Insight ID")

    assert rows.loc["INS-001", "Evidence count"] == 1
    assert rows.loc["INS-001", "Hypothesis count"] == 1
    assert rows.loc["INS-003", "Hypothesis count"] == 0


def test_general_insight_may_have_no_affected_field() -> None:
    row = _report().insights_frame().set_index("Insight ID").loc["INS-003"]

    assert row["Affected fields"] == ()
    assert row["Affected field count"] == 0


def test_duplicate_insight_id_is_invalid() -> None:
    insights = _insights()
    insights.append(copy.deepcopy(insights[0]))
    report = _report(insights=insights)

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="declared more than once",
    ):
        report.raise_if_invalid()


def test_duplicate_evidence_id_is_invalid() -> None:
    evidence = _evidence()
    evidence.append(copy.deepcopy(evidence[0]))

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="Evidence ID",
    ):
        _report(evidence=evidence).raise_if_invalid()


def test_duplicate_hypothesis_id_is_invalid() -> None:
    hypotheses = _hypotheses()
    hypotheses.append(copy.deepcopy(hypotheses[0]))

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="Hypothesis ID",
    ):
        _report(hypotheses=hypotheses).raise_if_invalid()


def test_duplicate_action_id_is_invalid() -> None:
    actions = _actions()
    actions.append(copy.deepcopy(actions[0]))

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="Action ID",
    ):
        _report(validation_actions=actions).raise_if_invalid()


def test_duplicate_limitation_id_is_invalid() -> None:
    limitations = _limitations()
    limitations.append(copy.deepcopy(limitations[0]))

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="Limitation ID",
    ):
        _report(limitations=limitations).raise_if_invalid()


def test_unknown_affected_field_is_invalid() -> None:
    insights = _insights()
    insights[0]["affected_fields"] = ("unknown",)

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="is not available",
    ):
        _report(insights=insights).raise_if_invalid()


def test_unknown_limitation_field_is_invalid() -> None:
    limitations = _limitations()
    limitations[0]["affected_fields"] = ("unknown",)

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="is not available",
    ):
        _report(limitations=limitations).raise_if_invalid()


def test_insight_without_evidence_is_invalid() -> None:
    evidence = [row for row in _evidence() if row["insight_id"] != "INS-001"]

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="No evidence record",
    ):
        _report(evidence=evidence).raise_if_invalid()


def test_evidence_with_unknown_insight_is_invalid() -> None:
    evidence = _evidence()
    evidence[0]["insight_id"] = "INS-999"

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="is not declared",
    ):
        _report(evidence=evidence).raise_if_invalid()


def test_hypothesis_without_linked_insight_is_invalid() -> None:
    hypotheses = _hypotheses()
    hypotheses[0]["linked_insight_ids"] = ()

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="No observed insight supports",
    ):
        _report(hypotheses=hypotheses).raise_if_invalid()


def test_hypothesis_with_unknown_insight_is_invalid() -> None:
    hypotheses = _hypotheses()
    hypotheses[0]["linked_insight_ids"] = ("INS-999",)

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="is not declared",
    ):
        _report(hypotheses=hypotheses).raise_if_invalid()


def test_hypothesis_without_validation_action_is_invalid() -> None:
    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="No validation action",
    ):
        _report(validation_actions=()).raise_if_invalid()


def test_action_with_unknown_hypothesis_is_invalid() -> None:
    actions = _actions()
    actions[0]["hypothesis_ids"] = ("HYP-999",)

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="is not declared",
    ):
        _report(validation_actions=actions).raise_if_invalid()


def test_missing_interpretation_boundary_is_invalid() -> None:
    insights = _insights()
    insights[0]["interpretation_boundary"] = ""

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="No interpretation boundary",
    ):
        _report(insights=insights).raise_if_invalid()


@pytest.mark.parametrize("relevance", ["Critical", "", None])
def test_invalid_relevance_is_rejected(relevance: object) -> None:
    insights = _insights()
    insights[0]["relevance"] = relevance

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="Unsupported relevance",
    ):
        _report(insights=insights).raise_if_invalid()


@pytest.mark.parametrize("status", ["Pending", "Done", ""])
def test_invalid_insight_status_is_rejected(status: str) -> None:
    insights = _insights()
    insights[0]["status"] = status

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="Unsupported insight status",
    ):
        _report(insights=insights).raise_if_invalid()


def test_invalid_insight_type_is_rejected() -> None:
    insights = _insights()
    insights[0]["insight_type"] = "Conclusion"

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="Unsupported insight type",
    ):
        _report(insights=insights).raise_if_invalid()


def test_invalid_evidence_kind_is_rejected() -> None:
    evidence = _evidence()
    evidence[0]["evidence_kind"] = "Unknown"

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="Unsupported evidence kind",
    ):
        _report(evidence=evidence).raise_if_invalid()


def test_missing_required_validation_is_invalid() -> None:
    hypotheses = _hypotheses()
    hypotheses[0]["required_validation"] = ""

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="No required validation",
    ):
        _report(hypotheses=hypotheses).raise_if_invalid()


def test_invalid_validation_type_is_rejected() -> None:
    actions = _actions()
    actions[0]["validation_type"] = "Experiment"

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="Unsupported validation type",
    ):
        _report(validation_actions=actions).raise_if_invalid()


def test_missing_acceptance_criteria_is_invalid() -> None:
    actions = _actions()
    actions[0]["acceptance_criteria"] = ""

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="No measurable acceptance criteria",
    ):
        _report(validation_actions=actions).raise_if_invalid()


def test_invalid_limitation_contract_is_rejected() -> None:
    limitations = _limitations()
    limitations[0]["limitation_type"] = "Unknown"
    limitations[0]["severity"] = "Urgent"
    limitations[0]["status"] = "Pending"
    report = _report(limitations=limitations)

    issues = set(report.issues_frame()["Issue"])
    assert "Invalid limitation type" in issues
    assert "Invalid limitation severity" in issues
    assert "Invalid limitation status" in issues


def test_report_exposes_expected_condition_properties() -> None:
    report = _report()

    assert report.has_high_relevance_insights
    assert report.has_unvalidated_hypotheses
    assert report.has_confounding_risks
    assert not report.has_structural_dependencies
    assert not report.has_modeling_limitations
    assert report.has_unresolved_governance_limits
    assert report.has_unresolved_temporal_limits


def test_dependency_and_modeling_limitation_properties() -> None:
    insights = _insights()
    insights[0]["insight_type"] = "Dependency"
    insights[1]["insight_type"] = "Modeling limitation"
    report = _report(insights=insights)

    assert report.has_structural_dependencies
    assert report.has_modeling_limitations


def test_modeling_gate_reports_temporal_and_governance_limits() -> None:
    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="temporal or inference-time contract",
    ):
        _report().raise_if_modeling_not_ready()


def test_resolved_governance_limit_clears_modeling() -> None:
    insights = _insights()
    insights[2]["status"] = "Controlled"
    limitations = _limitations()
    limitations[0]["status"] = "Resolved"
    report = _report(insights=insights, limitations=limitations)

    assert report.is_ready_for_modeling
    report.raise_if_modeling_not_ready()


def test_high_relevance_hypothesis_gate_can_be_enabled() -> None:
    insights = _insights()
    insights[2]["status"] = "Controlled"
    limitations = _limitations()
    limitations[0]["status"] = "Resolved"
    report = _report(insights=insights, limitations=limitations)

    with pytest.raises(
        ExploratoryInsightsConsolidationError,
        match="high-relevance hypotheses",
    ):
        report.raise_if_modeling_not_ready(
            require_no_unvalidated_critical_hypotheses=True,
        )


def test_returned_frames_are_defensive_copies() -> None:
    report = _report()
    insights = report.insights_frame()
    insights.loc[0, "Title"] = "changed"

    assert report.insights_frame().loc[0, "Title"] != "changed"


def test_input_declarations_are_not_mutated() -> None:
    inputs = (
        _insights(),
        _evidence(),
        _hypotheses(),
        _actions(),
        _limitations(),
    )
    before = copy.deepcopy(inputs)

    _report(
        insights=inputs[0],
        evidence=inputs[1],
        hypotheses=inputs[2],
        validation_actions=inputs[3],
        limitations=inputs[4],
    )

    assert inputs == before


def test_results_are_deterministic() -> None:
    first = _report()
    second = _report()

    pd.testing.assert_frame_equal(
        first.insights_frame(),
        second.insights_frame(),
    )
    pd.testing.assert_frame_equal(
        first.hypotheses_frame(),
        second.hypotheses_frame(),
    )
    pd.testing.assert_frame_equal(
        first.validation_actions_frame(),
        second.validation_actions_frame(),
    )


def test_string_values_are_normalized_without_mutation() -> None:
    insights = _insights()
    insights[0]["insight_id"] = " INS-001 "
    insights[0]["affected_fields"] = "tenure"

    row = _report(insights=insights).insights_frame().set_index(
        "Insight ID"
    ).loc["INS-001"]

    assert row["Affected fields"] == ("tenure",)


def test_selected_validation_requirement_can_be_disabled() -> None:
    evidence = [row for row in _evidence() if row["insight_id"] != "INS-001"]
    report = _report(evidence=evidence)

    report.raise_if_invalid(require_evidence_for_insights=False)


def test_empty_consolidation_is_valid_and_modeling_ready() -> None:
    report = consolidate_key_exploratory_insights(
        available_fields=AVAILABLE_FIELDS,
        insights=(),
        evidence=(),
        hypotheses=(),
        validation_actions=(),
        limitations=(),
    )

    assert report.is_structurally_valid
    assert report.is_ready_for_preparation_decisions
    assert report.is_ready_for_modeling
    assert report.insights_frame().empty
