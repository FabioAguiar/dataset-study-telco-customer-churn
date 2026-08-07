"""Reusable, non-mutating consolidation of key exploratory insights."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_INSIGHT_COLUMNS: Final[list[str]] = [
    "Insight ID",
    "Theme",
    "Title",
    "Insight type",
    "Affected fields",
    "Affected field count",
    "Relevance",
    "Status",
    "Source stages",
    "Summary",
    "Modeling implication",
    "Interpretation boundary",
    "Evidence count",
    "Hypothesis count",
]

_EVIDENCE_COLUMNS: Final[list[str]] = [
    "Evidence ID",
    "Insight ID",
    "Evidence kind",
    "Source report",
    "Source metric",
    "Observed value",
    "Comparison value",
    "Direction",
    "Interpretation",
]

_HYPOTHESIS_COLUMNS: Final[list[str]] = [
    "Hypothesis ID",
    "Linked insight IDs",
    "Linked insight count",
    "Title",
    "Hypothesis",
    "Status",
    "Confounding risks",
    "Confounding risk count",
    "Required validation",
    "Decision stage",
    "Validation action count",
]

_ACTION_COLUMNS: Final[list[str]] = [
    "Action ID",
    "Hypothesis IDs",
    "Hypothesis count",
    "Validation type",
    "Action",
    "Stage",
    "Blocking",
    "Status",
    "Acceptance criteria",
]

_LIMITATION_COLUMNS: Final[list[str]] = [
    "Limitation ID",
    "Theme",
    "Title",
    "Limitation type",
    "Affected fields",
    "Affected field count",
    "Severity",
    "Status",
    "Source stages",
    "Implication",
    "Required resolution",
]

_READINESS_COLUMNS: Final[list[str]] = [
    "Readiness check",
    "Ready",
    "Interpretation",
]

_ISSUE_COLUMNS: Final[list[str]] = [
    "Scope",
    "Item",
    "Issue",
    "Details",
    "Potential impact",
]

_ALLOWED_INSIGHT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Pattern",
        "Contrast",
        "Dependency",
        "Data-quality condition",
        "Modeling limitation",
        "Governance limitation",
    }
)

_ALLOWED_RELEVANCE: Final[frozenset[str]] = frozenset(
    {
        "High",
        "Medium",
        "Low",
        "Contextual",
    }
)

_ALLOWED_INSIGHT_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Observed",
        "Exploratory",
        "Controlled",
        "Unresolved",
    }
)

_ALLOWED_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "Data quality",
        "Target distribution",
        "Numerical pattern",
        "Categorical contrast",
        "Feature dependency",
        "Feature-to-target",
        "Leakage/governance",
    }
)

_ALLOWED_HYPOTHESIS_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Unvalidated",
        "Partially validated",
        "Validated",
        "Rejected",
        "Deferred",
    }
)

_ALLOWED_DECISION_STAGES: Final[frozenset[str]] = frozenset(
    {
        "Data preparation",
        "Model selection",
        "Model evaluation",
        "External contract",
    }
)

_ALLOWED_VALIDATION_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Interaction test",
        "Ablation",
        "Cross-validation",
        "Calibration",
        "Confounder control",
        "Error analysis",
        "External data",
        "Temporal validation",
    }
)

_ALLOWED_ACTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Planned",
        "In progress",
        "Complete",
        "Blocked",
        "Deferred",
    }
)

_ALLOWED_LIMITATION_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Data sufficiency",
        "Modeling",
        "Governance",
        "Temporal",
        "Inference availability",
        "Data quality",
    }
)

_ALLOWED_LIMITATION_SEVERITIES: Final[frozenset[str]] = frozenset(
    {
        "Critical",
        "High",
        "Medium",
        "Low",
        "Contextual",
    }
)

_ALLOWED_LIMITATION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Open",
        "Controlled",
        "Accepted",
        "Resolved",
        "Unresolved",
    }
)

_RELEVANCE_ORDER: Final[dict[str, int]] = {
    "High": 0,
    "Medium": 1,
    "Low": 2,
    "Contextual": 3,
}

_HYPOTHESIS_STATUS_ORDER: Final[dict[str, int]] = {
    "Unvalidated": 0,
    "Partially validated": 1,
    "Deferred": 2,
    "Validated": 3,
    "Rejected": 4,
}

_VALIDATION_TYPE_ORDER: Final[dict[str, int]] = {
    "Interaction test": 0,
    "Ablation": 1,
    "Confounder control": 2,
    "Cross-validation": 3,
    "Calibration": 4,
    "Error analysis": 5,
    "Temporal validation": 6,
    "External data": 7,
}

_LIMITATION_SEVERITY_ORDER: Final[dict[str, int]] = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
    "Contextual": 4,
}

_UNRESOLVED_LIMITATION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Open",
        "Unresolved",
    }
)

_GOVERNANCE_LIMITATION_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Governance",
        "Temporal",
        "Inference availability",
    }
)

_TEMPORAL_LIMITATION_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Temporal",
        "Inference availability",
    }
)


class ExploratoryInsightsConsolidationError(ValueError):
    """Raised when insight declarations or readiness requirements fail."""


@dataclass(frozen=True, slots=True)
class KeyExploratoryInsightsReport:
    """Consolidate exploratory evidence without recomputing analysis."""

    available_fields: tuple[str, ...]
    insights: pd.DataFrame
    evidence: pd.DataFrame
    hypotheses: pd.DataFrame
    validation_actions: pd.DataFrame
    limitations: pd.DataFrame
    issues: pd.DataFrame

    @property
    def has_high_relevance_insights(self) -> bool:
        """Return whether at least one insight has high relevance."""
        if self.insights.empty:
            return False
        return bool(self.insights["Relevance"].eq("High").any())

    @property
    def has_unvalidated_hypotheses(self) -> bool:
        """Return whether hypotheses still require validation."""
        if self.hypotheses.empty:
            return False
        return bool(
            self.hypotheses["Status"].isin(
                {"Unvalidated", "Partially validated", "Deferred"}
            ).any()
        )

    @property
    def has_confounding_risks(self) -> bool:
        """Return whether hypotheses declare possible confounders."""
        if self.hypotheses.empty:
            return False
        return bool(self.hypotheses["Confounding risk count"].gt(0).any())

    @property
    def has_structural_dependencies(self) -> bool:
        """Return whether dependency insights are documented."""
        if self.insights.empty:
            return False
        return bool(self.insights["Insight type"].eq("Dependency").any())

    @property
    def has_modeling_limitations(self) -> bool:
        """Return whether modeling limitations are documented."""
        insight_limit = (
            not self.insights.empty
            and self.insights["Insight type"].eq("Modeling limitation").any()
        )
        declared_limit = (
            not self.limitations.empty
            and self.limitations["Limitation type"].eq("Modeling").any()
        )
        return bool(insight_limit or declared_limit)

    @property
    def has_unresolved_governance_limits(self) -> bool:
        """Return whether governance, timing, or availability remains open."""
        insight_limit = False
        if not self.insights.empty:
            insight_limit = bool(
                (
                    self.insights["Insight type"].eq("Governance limitation")
                    & self.insights["Status"].eq("Unresolved")
                ).any()
            )

        declared_limit = False
        if not self.limitations.empty:
            declared_limit = bool(
                (
                    self.limitations["Limitation type"].isin(
                        _GOVERNANCE_LIMITATION_TYPES
                    )
                    & self.limitations["Status"].isin(
                        _UNRESOLVED_LIMITATION_STATUSES
                    )
                ).any()
            )

        return insight_limit or declared_limit

    @property
    def has_unresolved_temporal_limits(self) -> bool:
        """Return whether temporal or inference availability remains open."""
        if self.limitations.empty:
            return False
        return bool(
            (
                self.limitations["Limitation type"].isin(
                    _TEMPORAL_LIMITATION_TYPES
                )
                & self.limitations["Status"].isin(
                    _UNRESOLVED_LIMITATION_STATUSES
                )
            ).any()
        )

    @property
    def is_structurally_valid(self) -> bool:
        """Return whether identifiers, references, and values are valid."""
        return self.issues.empty

    @property
    def is_ready_for_preparation_decisions(self) -> bool:
        """Return whether insights can safely inform preparation decisions."""
        return self.is_structurally_valid

    @property
    def is_ready_for_modeling(self) -> bool:
        """Return whether no unresolved governance limitation blocks modeling."""
        return (
            self.is_ready_for_preparation_decisions
            and not self.has_unresolved_governance_limits
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic counts and readiness indicators."""
        rows = [
            {
                "Metric": "Declared fields",
                "Value": len(self.available_fields),
                "Interpretation": "Fields available to referenced insights",
            },
            {
                "Metric": "Consolidated insights",
                "Value": len(self.insights),
                "Interpretation": "Prioritized exploratory conclusions",
            },
            {
                "Metric": "High-relevance insights",
                "Value": int(
                    self.insights["Relevance"].eq("High").sum()
                )
                if not self.insights.empty
                else 0,
                "Interpretation": "Insights prioritized for downstream review",
            },
            {
                "Metric": "Evidence records",
                "Value": len(self.evidence),
                "Interpretation": "Traceable support from prior analytical stages",
            },
            {
                "Metric": "Exploratory hypotheses",
                "Value": len(self.hypotheses),
                "Interpretation": "Explanations that still require validation",
            },
            {
                "Metric": "Unvalidated hypotheses",
                "Value": int(self.has_unvalidated_hypotheses),
                "Interpretation": "At least one hypothesis remains unvalidated",
            },
            {
                "Metric": "Validation actions",
                "Value": len(self.validation_actions),
                "Interpretation": "Traceable future validation work",
            },
            {
                "Metric": "Declared limitations",
                "Value": len(self.limitations),
                "Interpretation": "Known interpretation and modeling constraints",
            },
            {
                "Metric": "Unresolved governance limits",
                "Value": int(self.has_unresolved_governance_limits),
                "Interpretation": "Open timing, governance, or inference constraints",
            },
            {
                "Metric": "Structurally valid",
                "Value": self.is_structurally_valid,
                "Interpretation": "Insight, evidence, and validation contracts are coherent",
            },
            {
                "Metric": "Ready for preparation decisions",
                "Value": self.is_ready_for_preparation_decisions,
                "Interpretation": "Insights can inform preliminary preparation scope",
            },
            {
                "Metric": "Ready for modeling",
                "Value": self.is_ready_for_modeling,
                "Interpretation": "No unresolved governance limitation remains",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def insights_frame(self) -> pd.DataFrame:
        """Return a defensive copy of consolidated insights."""
        return self.insights.copy(deep=True)

    def evidence_frame(self) -> pd.DataFrame:
        """Return a defensive copy of insight evidence."""
        return self.evidence.copy(deep=True)

    def hypotheses_frame(self) -> pd.DataFrame:
        """Return a defensive copy of exploratory hypotheses."""
        return self.hypotheses.copy(deep=True)

    def validation_actions_frame(self) -> pd.DataFrame:
        """Return a defensive copy of validation actions."""
        return self.validation_actions.copy(deep=True)

    def limitations_frame(self) -> pd.DataFrame:
        """Return a defensive copy of exploratory limitations."""
        return self.limitations.copy(deep=True)

    def readiness_frame(self) -> pd.DataFrame:
        """Return preparation and modeling readiness checks."""
        rows = [
            {
                "Readiness check": "Structural contract",
                "Ready": self.is_structurally_valid,
                "Interpretation": (
                    "Insight declarations and references are valid"
                    if self.is_structurally_valid
                    else "Structural issues must be corrected"
                ),
            },
            {
                "Readiness check": "Preparation decisions",
                "Ready": self.is_ready_for_preparation_decisions,
                "Interpretation": (
                    "Exploratory evidence can inform preliminary preparation"
                    if self.is_ready_for_preparation_decisions
                    else "Preparation decisions are not safely traceable"
                ),
            },
            {
                "Readiness check": "Modeling clearance",
                "Ready": self.is_ready_for_modeling,
                "Interpretation": (
                    "No unresolved governance limitation remains"
                    if self.is_ready_for_modeling
                    else "Governance or temporal limits still block modeling"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_READINESS_COLUMNS)

    def issues_frame(self) -> pd.DataFrame:
        """Return a defensive copy of structural consolidation issues."""
        return self.issues.copy(deep=True)

    def raise_if_invalid(
        self,
        *,
        require_unique_insight_ids: bool = True,
        require_unique_evidence_ids: bool = True,
        require_unique_hypothesis_ids: bool = True,
        require_unique_action_ids: bool = True,
        require_unique_limitation_ids: bool = True,
        require_known_fields: bool = True,
        require_evidence_for_insights: bool = True,
        require_validation_for_hypotheses: bool = True,
        require_interpretation_boundaries: bool = True,
        require_valid_relevance: bool = True,
        require_valid_statuses: bool = True,
        require_valid_types: bool = True,
        require_valid_references: bool = True,
        require_acceptance_criteria: bool = True,
    ) -> None:
        """Raise when selected structural requirements are not satisfied."""
        selected: set[str] = set()
        if require_unique_insight_ids:
            selected.add("Duplicate insight ID")
        if require_unique_evidence_ids:
            selected.add("Duplicate evidence ID")
        if require_unique_hypothesis_ids:
            selected.add("Duplicate hypothesis ID")
        if require_unique_action_ids:
            selected.add("Duplicate action ID")
        if require_unique_limitation_ids:
            selected.add("Duplicate limitation ID")
        if require_known_fields:
            selected.add("Unknown affected field")
        if require_evidence_for_insights:
            selected.add("Insight without evidence")
        if require_validation_for_hypotheses:
            selected.update(
                {
                    "Hypothesis without linked insight",
                    "Hypothesis without validation action",
                    "Missing required validation",
                }
            )
        if require_interpretation_boundaries:
            selected.add("Missing interpretation boundary")
        if require_valid_relevance:
            selected.add("Invalid relevance")
        if require_valid_statuses:
            selected.update(
                {
                    "Invalid insight status",
                    "Invalid hypothesis status",
                    "Invalid validation status",
                    "Invalid limitation status",
                }
            )
        if require_valid_types:
            selected.update(
                {
                    "Invalid insight type",
                    "Invalid evidence kind",
                    "Invalid validation type",
                    "Invalid decision stage",
                    "Invalid limitation type",
                    "Invalid limitation severity",
                }
            )
        if require_valid_references:
            selected.update(
                {
                    "Unknown insight reference",
                    "Unknown hypothesis reference",
                }
            )
        if require_acceptance_criteria:
            selected.add("Missing acceptance criteria")

        failures = self.issues.loc[self.issues["Issue"].isin(selected)]
        if failures.empty:
            return

        details = "; ".join(
            str(value) for value in failures["Details"].tolist()
        )
        raise ExploratoryInsightsConsolidationError(
            "Invalid exploratory-insight consolidation: " + details
        )

    def raise_if_modeling_not_ready(
        self,
        *,
        require_no_unvalidated_critical_hypotheses: bool = False,
        require_temporal_contract_complete: bool = True,
        require_no_unresolved_governance_limits: bool = True,
    ) -> None:
        """Raise when selected downstream-modeling gates remain open."""
        reasons: list[str] = []

        if not self.is_structurally_valid:
            reasons.append("the structural insight contract is invalid")

        if (
            require_no_unvalidated_critical_hypotheses
            and self.has_unvalidated_hypotheses
            and self.has_high_relevance_insights
        ):
            reasons.append("high-relevance hypotheses remain unvalidated")

        if require_temporal_contract_complete and self.has_unresolved_temporal_limits:
            reasons.append("the temporal or inference-time contract is incomplete")

        if (
            require_no_unresolved_governance_limits
            and self.has_unresolved_governance_limits
        ):
            reasons.append("unresolved governance limitations remain")

        if reasons:
            raise ExploratoryInsightsConsolidationError(
                "Exploratory insights do not clear modeling: "
                + "; ".join(reasons)
            )


def consolidate_key_exploratory_insights(
    *,
    available_fields: Sequence[object],
    insights: Sequence[Mapping[str, object]],
    evidence: Sequence[Mapping[str, object]],
    hypotheses: Sequence[Mapping[str, object]],
    validation_actions: Sequence[Mapping[str, object]],
    limitations: Sequence[Mapping[str, object]],
) -> KeyExploratoryInsightsReport:
    """Normalize and validate exploratory insight declarations."""
    fields = _unique_text_tuple(available_fields)
    insight_declarations = deepcopy(list(insights))
    evidence_declarations = deepcopy(list(evidence))
    hypothesis_declarations = deepcopy(list(hypotheses))
    action_declarations = deepcopy(list(validation_actions))
    limitation_declarations = deepcopy(list(limitations))

    issues: list[dict[str, object]] = []

    insight_rows = _normalize_insights(
        insight_declarations,
        available_fields=fields,
        issues=issues,
    )
    evidence_rows = _normalize_evidence(
        evidence_declarations,
        issues=issues,
    )
    hypothesis_rows = _normalize_hypotheses(
        hypothesis_declarations,
        issues=issues,
    )
    action_rows = _normalize_actions(
        action_declarations,
        issues=issues,
    )
    limitation_rows = _normalize_limitations(
        limitation_declarations,
        available_fields=fields,
        issues=issues,
    )

    insight_ids = {str(row["Insight ID"]) for row in insight_rows}
    hypothesis_ids = {str(row["Hypothesis ID"]) for row in hypothesis_rows}

    _validate_references_and_coverage(
        insight_rows=insight_rows,
        evidence_rows=evidence_rows,
        hypothesis_rows=hypothesis_rows,
        action_rows=action_rows,
        insight_ids=insight_ids,
        hypothesis_ids=hypothesis_ids,
        issues=issues,
    )

    evidence_count_by_insight: dict[str, int] = {}
    for row in evidence_rows:
        insight_id = str(row["Insight ID"])
        evidence_count_by_insight[insight_id] = (
            evidence_count_by_insight.get(insight_id, 0) + 1
        )

    hypothesis_count_by_insight: dict[str, int] = {}
    for row in hypothesis_rows:
        for insight_id in row["Linked insight IDs"]:
            key = str(insight_id)
            hypothesis_count_by_insight[key] = (
                hypothesis_count_by_insight.get(key, 0) + 1
            )

    action_count_by_hypothesis: dict[str, int] = {}
    for row in action_rows:
        for hypothesis_id in row["Hypothesis IDs"]:
            key = str(hypothesis_id)
            action_count_by_hypothesis[key] = (
                action_count_by_hypothesis.get(key, 0) + 1
            )

    for row in insight_rows:
        insight_id = str(row["Insight ID"])
        row["Evidence count"] = evidence_count_by_insight.get(insight_id, 0)
        row["Hypothesis count"] = hypothesis_count_by_insight.get(insight_id, 0)

    for row in hypothesis_rows:
        hypothesis_id = str(row["Hypothesis ID"])
        row["Validation action count"] = action_count_by_hypothesis.get(
            hypothesis_id, 0
        )

    insights_frame = pd.DataFrame(insight_rows, columns=_INSIGHT_COLUMNS)
    if not insights_frame.empty:
        insights_frame["_order"] = insights_frame["Relevance"].map(
            _RELEVANCE_ORDER
        ).fillna(99)
        insights_frame = (
            insights_frame.sort_values(["_order", "Insight ID"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    evidence_frame = pd.DataFrame(evidence_rows, columns=_EVIDENCE_COLUMNS)
    if not evidence_frame.empty:
        evidence_frame = evidence_frame.sort_values(
            ["Insight ID", "Evidence ID"]
        ).reset_index(drop=True)

    hypotheses_frame = pd.DataFrame(hypothesis_rows, columns=_HYPOTHESIS_COLUMNS)
    if not hypotheses_frame.empty:
        hypotheses_frame["_order"] = hypotheses_frame["Status"].map(
            _HYPOTHESIS_STATUS_ORDER
        ).fillna(99)
        hypotheses_frame = (
            hypotheses_frame.sort_values(["_order", "Hypothesis ID"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    actions_frame = pd.DataFrame(action_rows, columns=_ACTION_COLUMNS)
    if not actions_frame.empty:
        actions_frame["_order"] = actions_frame["Validation type"].map(
            _VALIDATION_TYPE_ORDER
        ).fillna(99)
        actions_frame = (
            actions_frame.sort_values(["_order", "Action ID"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    limitations_frame = pd.DataFrame(
        limitation_rows, columns=_LIMITATION_COLUMNS
    )
    if not limitations_frame.empty:
        limitations_frame["_order"] = limitations_frame["Severity"].map(
            _LIMITATION_SEVERITY_ORDER
        ).fillna(99)
        limitations_frame = (
            limitations_frame.sort_values(["_order", "Limitation ID"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    issues_frame = pd.DataFrame(issues, columns=_ISSUE_COLUMNS)
    if not issues_frame.empty:
        issues_frame = issues_frame.sort_values(
            ["Scope", "Item", "Issue"]
        ).reset_index(drop=True)

    return KeyExploratoryInsightsReport(
        available_fields=fields,
        insights=insights_frame,
        evidence=evidence_frame,
        hypotheses=hypotheses_frame,
        validation_actions=actions_frame,
        limitations=limitations_frame,
        issues=issues_frame,
    )


def _normalize_insights(
    declarations: Sequence[Mapping[str, object]],
    *,
    available_fields: tuple[str, ...],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    available = set(available_fields)

    for position, declaration in enumerate(declarations):
        item = f"insight[{position}]"
        insight_id = _text(declaration.get("insight_id")) or item
        if insight_id in seen:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Duplicate insight ID",
                    f"Insight ID {insight_id!r} is declared more than once",
                    "Evidence and hypotheses cannot be linked deterministically",
                )
            )
        seen.add(insight_id)

        affected_fields = _tuple_values(declaration.get("affected_fields", ()))
        for field in affected_fields:
            if field not in available:
                issues.append(
                    _issue(
                        "Insight",
                        insight_id,
                        "Unknown affected field",
                        f"Affected field {field!r} is not available",
                        "The insight may refer to a non-existent variable",
                    )
                )

        insight_type = _text(declaration.get("insight_type"))
        if insight_type not in _ALLOWED_INSIGHT_TYPES:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Invalid insight type",
                    f"Unsupported insight type {insight_type!r}",
                    "Theme summaries and readiness are unreliable",
                )
            )

        relevance = _text(declaration.get("relevance"))
        if relevance not in _ALLOWED_RELEVANCE:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Invalid relevance",
                    f"Unsupported relevance {relevance!r}",
                    "Priority ordering is not reliable",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_INSIGHT_STATUSES:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Invalid insight status",
                    f"Unsupported insight status {status!r}",
                    "Observed facts and unresolved hypotheses may be mixed",
                )
            )

        boundary = _text(declaration.get("interpretation_boundary"))
        if not boundary:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Missing interpretation boundary",
                    "No interpretation boundary was declared",
                    "Correlation may be presented as causation or a final decision",
                )
            )

        rows.append(
            {
                "Insight ID": insight_id,
                "Theme": _text(declaration.get("theme")),
                "Title": _text(declaration.get("title")),
                "Insight type": insight_type,
                "Affected fields": affected_fields,
                "Affected field count": len(affected_fields),
                "Relevance": relevance,
                "Status": status,
                "Source stages": _tuple_values(
                    declaration.get("source_stages", ())
                ),
                "Summary": _text(declaration.get("summary")),
                "Modeling implication": _text(
                    declaration.get("modeling_implication")
                ),
                "Interpretation boundary": boundary,
                "Evidence count": 0,
                "Hypothesis count": 0,
            }
        )

    return rows


def _normalize_evidence(
    declarations: Sequence[Mapping[str, object]],
    *,
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, declaration in enumerate(declarations):
        item = f"evidence[{position}]"
        evidence_id = _text(declaration.get("evidence_id")) or item
        if evidence_id in seen:
            issues.append(
                _issue(
                    "Evidence",
                    evidence_id,
                    "Duplicate evidence ID",
                    f"Evidence ID {evidence_id!r} is declared more than once",
                    "Insight support cannot be traced uniquely",
                )
            )
        seen.add(evidence_id)

        evidence_kind = _text(declaration.get("evidence_kind"))
        if evidence_kind not in _ALLOWED_EVIDENCE_KINDS:
            issues.append(
                _issue(
                    "Evidence",
                    evidence_id,
                    "Invalid evidence kind",
                    f"Unsupported evidence kind {evidence_kind!r}",
                    "Evidence matrices cannot be constructed consistently",
                )
            )

        rows.append(
            {
                "Evidence ID": evidence_id,
                "Insight ID": _text(declaration.get("insight_id")),
                "Evidence kind": evidence_kind,
                "Source report": _text(declaration.get("source_report")),
                "Source metric": _text(declaration.get("source_metric")),
                "Observed value": deepcopy(declaration.get("observed_value")),
                "Comparison value": deepcopy(
                    declaration.get("comparison_value")
                ),
                "Direction": _text(declaration.get("direction")),
                "Interpretation": _text(declaration.get("interpretation")),
            }
        )

    return rows


def _normalize_hypotheses(
    declarations: Sequence[Mapping[str, object]],
    *,
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, declaration in enumerate(declarations):
        item = f"hypothesis[{position}]"
        hypothesis_id = _text(declaration.get("hypothesis_id")) or item
        if hypothesis_id in seen:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Duplicate hypothesis ID",
                    f"Hypothesis ID {hypothesis_id!r} is declared more than once",
                    "Validation actions cannot be linked uniquely",
                )
            )
        seen.add(hypothesis_id)

        linked_ids = _tuple_values(declaration.get("linked_insight_ids", ()))
        if not linked_ids:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Hypothesis without linked insight",
                    "No observed insight supports this hypothesis",
                    "The hypothesis is detached from exploratory evidence",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_HYPOTHESIS_STATUSES:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Invalid hypothesis status",
                    f"Unsupported hypothesis status {status!r}",
                    "Validation readiness cannot be evaluated consistently",
                )
            )

        required_validation = _text(declaration.get("required_validation"))
        if not required_validation:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Missing required validation",
                    "No required validation was declared",
                    "The hypothesis may be treated as a conclusion",
                )
            )

        decision_stage = _text(declaration.get("decision_stage"))
        if decision_stage not in _ALLOWED_DECISION_STAGES:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Invalid decision stage",
                    f"Unsupported decision stage {decision_stage!r}",
                    "Future work cannot be routed consistently",
                )
            )

        confounding_risks = _tuple_values(
            declaration.get("confounding_risks", ())
        )

        rows.append(
            {
                "Hypothesis ID": hypothesis_id,
                "Linked insight IDs": linked_ids,
                "Linked insight count": len(linked_ids),
                "Title": _text(declaration.get("title")),
                "Hypothesis": _text(declaration.get("hypothesis")),
                "Status": status,
                "Confounding risks": confounding_risks,
                "Confounding risk count": len(confounding_risks),
                "Required validation": required_validation,
                "Decision stage": decision_stage,
                "Validation action count": 0,
            }
        )

    return rows


def _normalize_actions(
    declarations: Sequence[Mapping[str, object]],
    *,
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, declaration in enumerate(declarations):
        item = f"action[{position}]"
        action_id = _text(declaration.get("action_id")) or item
        if action_id in seen:
            issues.append(
                _issue(
                    "Validation action",
                    action_id,
                    "Duplicate action ID",
                    f"Action ID {action_id!r} is declared more than once",
                    "Validation work cannot be referenced uniquely",
                )
            )
        seen.add(action_id)

        validation_type = _text(declaration.get("validation_type"))
        if validation_type not in _ALLOWED_VALIDATION_TYPES:
            issues.append(
                _issue(
                    "Validation action",
                    action_id,
                    "Invalid validation type",
                    f"Unsupported validation type {validation_type!r}",
                    "The validation plan cannot be summarized consistently",
                )
            )

        stage = _text(declaration.get("stage"))
        if stage not in _ALLOWED_DECISION_STAGES:
            issues.append(
                _issue(
                    "Validation action",
                    action_id,
                    "Invalid decision stage",
                    f"Unsupported validation stage {stage!r}",
                    "Future work cannot be routed consistently",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_ACTION_STATUSES:
            issues.append(
                _issue(
                    "Validation action",
                    action_id,
                    "Invalid validation status",
                    f"Unsupported validation status {status!r}",
                    "Validation progress cannot be evaluated consistently",
                )
            )

        acceptance = _text(declaration.get("acceptance_criteria"))
        if not acceptance:
            issues.append(
                _issue(
                    "Validation action",
                    action_id,
                    "Missing acceptance criteria",
                    "No measurable acceptance criteria were declared",
                    "The validation cannot be completed objectively",
                )
            )

        hypothesis_ids = _tuple_values(
            declaration.get("hypothesis_ids", ())
        )

        rows.append(
            {
                "Action ID": action_id,
                "Hypothesis IDs": hypothesis_ids,
                "Hypothesis count": len(hypothesis_ids),
                "Validation type": validation_type,
                "Action": _text(declaration.get("action")),
                "Stage": stage,
                "Blocking": bool(declaration.get("blocking", False)),
                "Status": status,
                "Acceptance criteria": acceptance,
            }
        )

    return rows


def _normalize_limitations(
    declarations: Sequence[Mapping[str, object]],
    *,
    available_fields: tuple[str, ...],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    available = set(available_fields)

    for position, declaration in enumerate(declarations):
        item = f"limitation[{position}]"
        limitation_id = _text(declaration.get("limitation_id")) or item
        if limitation_id in seen:
            issues.append(
                _issue(
                    "Limitation",
                    limitation_id,
                    "Duplicate limitation ID",
                    f"Limitation ID {limitation_id!r} is declared more than once",
                    "Readiness may count the same limitation multiple times",
                )
            )
        seen.add(limitation_id)

        affected_fields = _tuple_values(declaration.get("affected_fields", ()))
        for field in affected_fields:
            if field not in available:
                issues.append(
                    _issue(
                        "Limitation",
                        limitation_id,
                        "Unknown affected field",
                        f"Affected field {field!r} is not available",
                        "The limitation may refer to a non-existent variable",
                    )
                )

        limitation_type = _text(declaration.get("limitation_type"))
        if limitation_type not in _ALLOWED_LIMITATION_TYPES:
            issues.append(
                _issue(
                    "Limitation",
                    limitation_id,
                    "Invalid limitation type",
                    f"Unsupported limitation type {limitation_type!r}",
                    "Readiness gates cannot classify the limitation",
                )
            )

        severity = _text(declaration.get("severity"))
        if severity not in _ALLOWED_LIMITATION_SEVERITIES:
            issues.append(
                _issue(
                    "Limitation",
                    limitation_id,
                    "Invalid limitation severity",
                    f"Unsupported limitation severity {severity!r}",
                    "Limitation priority ordering is unreliable",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_LIMITATION_STATUSES:
            issues.append(
                _issue(
                    "Limitation",
                    limitation_id,
                    "Invalid limitation status",
                    f"Unsupported limitation status {status!r}",
                    "Modeling readiness cannot be evaluated consistently",
                )
            )

        rows.append(
            {
                "Limitation ID": limitation_id,
                "Theme": _text(declaration.get("theme")),
                "Title": _text(declaration.get("title")),
                "Limitation type": limitation_type,
                "Affected fields": affected_fields,
                "Affected field count": len(affected_fields),
                "Severity": severity,
                "Status": status,
                "Source stages": _tuple_values(
                    declaration.get("source_stages", ())
                ),
                "Implication": _text(declaration.get("implication")),
                "Required resolution": _text(
                    declaration.get("required_resolution")
                ),
            }
        )

    return rows


def _validate_references_and_coverage(
    *,
    insight_rows: list[dict[str, object]],
    evidence_rows: list[dict[str, object]],
    hypothesis_rows: list[dict[str, object]],
    action_rows: list[dict[str, object]],
    insight_ids: set[str],
    hypothesis_ids: set[str],
    issues: list[dict[str, object]],
) -> None:
    evidence_by_insight: dict[str, int] = {}
    for row in evidence_rows:
        insight_id = str(row["Insight ID"])
        if insight_id not in insight_ids:
            issues.append(
                _issue(
                    "Evidence",
                    str(row["Evidence ID"]),
                    "Unknown insight reference",
                    f"Insight ID {insight_id!r} is not declared",
                    "Evidence cannot be traced to a known insight",
                )
            )
        evidence_by_insight[insight_id] = evidence_by_insight.get(insight_id, 0) + 1

    for row in insight_rows:
        insight_id = str(row["Insight ID"])
        if evidence_by_insight.get(insight_id, 0) == 0:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Insight without evidence",
                    "No evidence record references this insight",
                    "The insight is not traceable to prior analysis",
                )
            )

    for row in hypothesis_rows:
        hypothesis_id = str(row["Hypothesis ID"])
        for insight_id in row["Linked insight IDs"]:
            if str(insight_id) not in insight_ids:
                issues.append(
                    _issue(
                        "Hypothesis",
                        hypothesis_id,
                        "Unknown insight reference",
                        f"Insight ID {insight_id!r} is not declared",
                        "The hypothesis is linked to unavailable evidence",
                    )
                )

    actions_by_hypothesis: dict[str, int] = {}
    for row in action_rows:
        action_id = str(row["Action ID"])
        for hypothesis_id in row["Hypothesis IDs"]:
            key = str(hypothesis_id)
            if key not in hypothesis_ids:
                issues.append(
                    _issue(
                        "Validation action",
                        action_id,
                        "Unknown hypothesis reference",
                        f"Hypothesis ID {key!r} is not declared",
                        "Validation work cannot be traced to a hypothesis",
                    )
                )
            actions_by_hypothesis[key] = actions_by_hypothesis.get(key, 0) + 1

    for row in hypothesis_rows:
        hypothesis_id = str(row["Hypothesis ID"])
        if actions_by_hypothesis.get(hypothesis_id, 0) == 0:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Hypothesis without validation action",
                    "No validation action references this hypothesis",
                    "The hypothesis has no executable validation plan",
                )
            )


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tuple_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, Sequence):
        values = value
    else:
        values = (value,)

    result: list[str] = []
    for item in values:
        text = _text(item)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _json_safe(value: Any) -> Any:
    """Recursively replace non-JSON-compliant float NaN with ``None``.

    Some pandas dataframe columns render a missing entry as float ``nan``
    rather than ``None`` once serialized through ``to_dict("records")``.
    This normalizes those values for a strict JSON artifact without
    altering any analysis result.
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


EXPLORATORY_ANALYSIS_EVIDENCE_CONTRACT_VERSION: Final[str] = (
    "exploratory-analysis-evidence.v1"
)


def build_exploratory_analysis_evidence(
    report: KeyExploratoryInsightsReport,
    *,
    dataset_slug: str,
    visual_ids_by_insight: Mapping[str, Sequence[str]] | None = None,
    producer_revision: str | None = None,
    producer_revision_unavailable_reason: str | None = None,
    input_artifact_references: Mapping[str, Any] | None = None,
    contract_version: str = EXPLORATORY_ANALYSIS_EVIDENCE_CONTRACT_VERSION,
) -> dict[str, Any]:
    """Materialize an already-computed ``KeyExploratoryInsightsReport``.

    This performs no new exploratory analysis: it transcribes the insight,
    evidence, hypothesis, validation-action, and limitation records already
    produced by :func:`consolidate_key_exploratory_insights`. Each insight
    may carry an explicit ``visual_ids`` cross-reference into
    ``docs/images/visual-evidence-index.json`` when the caller supplies one;
    the artifact itself never depends on notebook runtime state.
    """
    if producer_revision is None and not producer_revision_unavailable_reason:
        raise ValueError(
            "producer_revision_unavailable_reason is required when "
            "producer_revision is not supplied."
        )

    visual_map = dict(visual_ids_by_insight or {})
    insight_rows: list[dict[str, Any]] = []
    for row in report.insights_frame().to_dict("records"):
        insight_id = str(row["Insight ID"])
        insight_rows.append(
            {
                "insight_id": insight_id,
                "theme": row["Theme"],
                "title": row["Title"],
                "insight_type": row["Insight type"],
                "affected_fields": list(row["Affected fields"]),
                "relevance": row["Relevance"],
                "status": row["Status"],
                "source_stages": list(row["Source stages"]),
                "summary": row["Summary"],
                "modeling_implication": row["Modeling implication"],
                "interpretation_boundary": row["Interpretation boundary"],
                "evidence_count": int(row["Evidence count"]),
                "hypothesis_count": int(row["Hypothesis count"]),
                "visual_ids": list(visual_map.get(insight_id, ())),
            }
        )

    return _json_safe({
        "schema_version": EXPLORATORY_ANALYSIS_EVIDENCE_CONTRACT_VERSION,
        "artifact_type": "exploratory_analysis_evidence",
        "contract_version": contract_version,
        "dataset_slug": dataset_slug,
        "available_fields": list(report.available_fields),
        "insights": insight_rows,
        "evidence": report.evidence_frame().to_dict("records"),
        "hypotheses": report.hypotheses_frame().to_dict("records"),
        "validation_actions": report.validation_actions_frame().to_dict("records"),
        "limitations": report.limitations_frame().to_dict("records"),
        "readiness": {
            "is_structurally_valid": report.is_structurally_valid,
            "is_ready_for_preparation_decisions": (
                report.is_ready_for_preparation_decisions
            ),
            "is_ready_for_modeling": report.is_ready_for_modeling,
            "has_high_relevance_insights": report.has_high_relevance_insights,
            "has_unvalidated_hypotheses": report.has_unvalidated_hypotheses,
            "has_unresolved_governance_limits": (
                report.has_unresolved_governance_limits
            ),
            "has_unresolved_temporal_limits": report.has_unresolved_temporal_limits,
        },
        "producer_revision": producer_revision,
        "producer_revision_unavailable_reason": (
            None
            if producer_revision is not None
            else producer_revision_unavailable_reason
        ),
        "input_artifact_hashes_or_references": (
            dict(input_artifact_references) if input_artifact_references else {}
        ),
    })


def _unique_text_tuple(values: Sequence[object]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _issue(
    scope: str,
    item: str,
    issue: str,
    details: str,
    potential_impact: str,
) -> dict[str, object]:
    return {
        "Scope": scope,
        "Item": item,
        "Issue": issue,
        "Details": details,
        "Potential impact": potential_impact,
    }
