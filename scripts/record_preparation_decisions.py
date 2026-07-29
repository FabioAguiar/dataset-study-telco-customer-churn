"""Reusable, non-mutating registry of preliminary data-preparation decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_DECISION_COLUMNS: Final[list[str]] = [
    "Decision ID",
    "Domain",
    "Title",
    "Affected fields",
    "Affected field count",
    "Status",
    "Phase",
    "Fit scope",
    "Operation",
    "Rationale",
    "Prerequisites",
    "Prerequisite count",
    "Acceptance criteria",
    "Source stages",
    "Evidence count",
    "Execution step count",
]

_EVIDENCE_COLUMNS: Final[list[str]] = [
    "Evidence ID",
    "Decision ID",
    "Source report",
    "Source item",
    "Observed value",
    "Expected or reference",
    "Interpretation",
]

_EXECUTION_COLUMNS: Final[list[str]] = [
    "Step ID",
    "Sequence",
    "Decision IDs",
    "Decision count",
    "Phase",
    "Action",
    "Blocking",
    "Status",
    "Temporal dependency",
    "Acceptance criteria",
]

_GUARDRAIL_COLUMNS: Final[list[str]] = [
    "Guardrail ID",
    "Domain",
    "Title",
    "Affected fields",
    "Affected field count",
    "Severity",
    "Status",
    "Prohibited operation",
    "Rationale",
    "Verification",
]

_SPLIT_POLICY_COLUMNS: Final[list[str]] = [
    "Policy item",
    "Value",
    "Status",
    "Interpretation",
]

_BLOCKER_COLUMNS: Final[list[str]] = [
    "Decision ID",
    "Title",
    "Domain",
    "Phase",
    "Status",
    "Fit scope",
    "Prerequisites",
    "Operation",
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

_ALLOWED_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Approved",
        "Conditional",
        "Deferred",
        "Prohibited",
        "Blocked",
    }
)

_ALLOWED_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "Cleaning",
        "Feature role",
        "Target governance",
        "Transformation",
        "Feature engineering",
        "Dataset splitting",
        "Leakage governance",
    }
)

_ALLOWED_PHASES: Final[frozenset[str]] = frozenset(
    {
        "Before split",
        "Split",
        "Train-only transformation",
        "Model selection",
        "External contract",
    }
)

_ALLOWED_FIT_SCOPES: Final[frozenset[str]] = frozenset(
    {
        "Deterministic",
        "Train only",
        "Evaluation only",
        "None",
        "External",
    }
)

_ALLOWED_STEP_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Planned",
        "Ready",
        "Blocked",
        "Complete",
        "Deferred",
    }
)

_ALLOWED_GUARDRAIL_SEVERITIES: Final[frozenset[str]] = frozenset(
    {
        "Critical",
        "High",
        "Medium",
        "Low",
    }
)

_ALLOWED_GUARDRAIL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Active",
        "Planned",
        "Controlled",
    }
)

_ALLOWED_TEMPORAL_POLICY_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Resolved temporal split",
        "Resolved snapshot fallback",
        "Unresolved",
    }
)

_STATUS_ORDER: Final[dict[str, int]] = {
    "Blocked": 0,
    "Prohibited": 1,
    "Approved": 2,
    "Conditional": 3,
    "Deferred": 4,
}

_PHASE_ORDER: Final[dict[str, int]] = {
    "Before split": 0,
    "Split": 1,
    "Train-only transformation": 2,
    "Model selection": 3,
    "External contract": 4,
}

_SPLIT_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "train_fraction",
    "validation_fraction",
    "test_fraction",
    "stratify_by",
    "random_seed",
    "shuffle",
    "temporal_priority",
    "temporal_policy_status",
    "random_split_fallback",
    "test_holdout_untouched",
    "disjoint_partitions_required",
    "group_by_identifiers",
)

_SPLIT_LABELS: Final[dict[str, str]] = {
    "train_fraction": "Train fraction",
    "validation_fraction": "Validation fraction",
    "test_fraction": "Test fraction",
    "stratify_by": "Stratification field",
    "random_seed": "Random seed",
    "shuffle": "Shuffle random split",
    "temporal_priority": "Temporal split priority",
    "temporal_policy_status": "Temporal policy status",
    "random_split_fallback": "Random-split fallback",
    "test_holdout_untouched": "Final test holdout",
    "disjoint_partitions_required": "Disjoint partitions",
    "group_by_identifiers": "Identifier grouping",
}


class PreparationDecisionContractError(ValueError):
    """Raised when preparation-decision declarations or gates are invalid."""


@dataclass(frozen=True, slots=True)
class PreparationDecisionReport:
    """Record preparation decisions without applying any transformation."""

    available_fields: tuple[str, ...]
    decisions: pd.DataFrame
    evidence: pd.DataFrame
    execution_steps: pd.DataFrame
    guardrails: pd.DataFrame
    split_policy: dict[str, object]
    issues: pd.DataFrame

    @property
    def has_approved_decisions(self) -> bool:
        return bool(
            not self.decisions.empty
            and self.decisions["Status"].eq("Approved").any()
        )

    @property
    def has_conditional_decisions(self) -> bool:
        return bool(
            not self.decisions.empty
            and self.decisions["Status"].eq("Conditional").any()
        )

    @property
    def has_deferred_decisions(self) -> bool:
        return bool(
            not self.decisions.empty
            and self.decisions["Status"].eq("Deferred").any()
        )

    @property
    def has_prohibited_operations(self) -> bool:
        return bool(
            (
                not self.decisions.empty
                and self.decisions["Status"].eq("Prohibited").any()
            )
            or not self.guardrails.empty
        )

    @property
    def has_external_blockers(self) -> bool:
        if self.decisions.empty:
            return False
        return bool(
            (
                self.decisions["Status"].eq("Blocked")
                & (
                    self.decisions["Phase"].eq("External contract")
                    | self.decisions["Fit scope"].eq("External")
                )
            ).any()
        )

    @property
    def has_train_only_operations(self) -> bool:
        if self.decisions.empty:
            return False
        return bool(
            self.decisions["Fit scope"].eq("Train only").any()
        )

    @property
    def has_deterministic_cleaning_scope(self) -> bool:
        if self.decisions.empty:
            return False
        selected = self.decisions.loc[
            self.decisions["Domain"].eq("Cleaning")
            & self.decisions["Status"].eq("Approved")
            & self.decisions["Phase"].eq("Before split")
            & self.decisions["Fit scope"].eq("Deterministic")
        ]
        return not selected.empty

    @property
    def is_structurally_valid(self) -> bool:
        return self.issues.empty

    @property
    def is_ready_for_deterministic_preparation(self) -> bool:
        if not self.is_structurally_valid:
            return False
        if not self.has_deterministic_cleaning_scope:
            return False

        deterministic = self.decisions.loc[
            self.decisions["Status"].eq("Approved")
            & self.decisions["Fit scope"].eq("Deterministic")
            & self.decisions["Phase"].eq("Before split")
        ]
        if deterministic.empty:
            return False

        linked_evidence = set(self.evidence["Decision ID"].astype(str))
        return set(deterministic["Decision ID"].astype(str)).issubset(
            linked_evidence
        )

    @property
    def is_ready_for_split_execution(self) -> bool:
        if not self.is_structurally_valid:
            return False
        if not self.is_ready_for_deterministic_preparation:
            return False

        temporal_status = str(
            self.split_policy.get("temporal_policy_status", "")
        )
        if temporal_status == "Unresolved":
            return False

        blocked_split = self.decisions.loc[
            self.decisions["Status"].eq("Blocked")
            & self.decisions["Domain"].isin(
                {"Dataset splitting", "Leakage governance"}
            )
        ]
        return blocked_split.empty

    @property
    def is_ready_for_modeling(self) -> bool:
        if not self.is_ready_for_split_execution:
            return False
        if self.has_external_blockers:
            return False

        blocking_steps = self.execution_steps.loc[
            self.execution_steps["Blocking"]
        ]
        if blocking_steps.empty:
            return True
        return bool(blocking_steps["Status"].eq("Complete").all())

    def summary_frame(self) -> pd.DataFrame:
        status_counts = self.decisions["Status"].value_counts()
        rows = [
            {
                "Metric": "Declared fields",
                "Value": len(self.available_fields),
                "Interpretation": "Fields available to preparation decisions",
            },
            {
                "Metric": "Preparation decisions",
                "Value": len(self.decisions),
                "Interpretation": "Versioned cleaning, transformation, engineering, and split decisions",
            },
            {
                "Metric": "Approved decisions",
                "Value": int(status_counts.get("Approved", 0)),
                "Interpretation": "Decisions authorized within their declared scope",
            },
            {
                "Metric": "Conditional decisions",
                "Value": int(status_counts.get("Conditional", 0)),
                "Interpretation": "Decisions requiring a later model or contract condition",
            },
            {
                "Metric": "Deferred decisions",
                "Value": int(status_counts.get("Deferred", 0)),
                "Interpretation": "Candidates intentionally postponed to evaluation",
            },
            {
                "Metric": "Prohibited decisions",
                "Value": int(status_counts.get("Prohibited", 0)),
                "Interpretation": "Operations explicitly excluded from preparation",
            },
            {
                "Metric": "Blocked decisions",
                "Value": int(status_counts.get("Blocked", 0)),
                "Interpretation": "Decisions awaiting unresolved prerequisites",
            },
            {
                "Metric": "Evidence records",
                "Value": len(self.evidence),
                "Interpretation": "Traceable support for decisions",
            },
            {
                "Metric": "Execution steps",
                "Value": len(self.execution_steps),
                "Interpretation": "Ordered future implementation plan",
            },
            {
                "Metric": "Guardrails",
                "Value": len(self.guardrails),
                "Interpretation": "Explicit prohibitions protecting preparation integrity",
            },
            {
                "Metric": "Structurally valid",
                "Value": self.is_structurally_valid,
                "Interpretation": "Decisions, evidence, steps, guardrails, and split policy are coherent",
            },
            {
                "Metric": "Ready for deterministic preparation",
                "Value": self.is_ready_for_deterministic_preparation,
                "Interpretation": "Approved deterministic cleaning may be implemented",
            },
            {
                "Metric": "Ready for split execution",
                "Value": self.is_ready_for_split_execution,
                "Interpretation": "Temporal policy and split blockers are resolved",
            },
            {
                "Metric": "Ready for modeling",
                "Value": self.is_ready_for_modeling,
                "Interpretation": "All blocking execution and governance conditions are complete",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def decisions_frame(self) -> pd.DataFrame:
        return _defensive_frame(self.decisions)

    def evidence_frame(self) -> pd.DataFrame:
        return _defensive_frame(self.evidence)

    def execution_plan_frame(self) -> pd.DataFrame:
        return _defensive_frame(self.execution_steps)

    def guardrails_frame(self) -> pd.DataFrame:
        return _defensive_frame(self.guardrails)

    def split_policy_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        temporal_status = str(
            self.split_policy.get("temporal_policy_status", "")
        )
        for key in _SPLIT_REQUIRED_KEYS:
            value = deepcopy(self.split_policy.get(key))
            status = "Resolved"
            if key == "temporal_policy_status" and value == "Unresolved":
                status = "Unresolved"
            elif key in {"temporal_priority", "test_holdout_untouched", "disjoint_partitions_required"} and value is not True:
                status = "Invalid"
            elif key == "random_split_fallback" and not _text(value):
                status = "Missing"

            interpretation = _split_interpretation(key, value, temporal_status)
            rows.append(
                {
                    "Policy item": _SPLIT_LABELS[key],
                    "Value": value,
                    "Status": status,
                    "Interpretation": interpretation,
                }
            )
        return pd.DataFrame(rows, columns=_SPLIT_POLICY_COLUMNS)

    def blockers_frame(self) -> pd.DataFrame:
        if self.decisions.empty:
            return pd.DataFrame(columns=_BLOCKER_COLUMNS)
        selected = self.decisions.loc[
            self.decisions["Status"].eq("Blocked")
        ]
        rows = [
            {
                "Decision ID": row["Decision ID"],
                "Title": row["Title"],
                "Domain": row["Domain"],
                "Phase": row["Phase"],
                "Status": row["Status"],
                "Fit scope": row["Fit scope"],
                "Prerequisites": deepcopy(row["Prerequisites"]),
                "Operation": row["Operation"],
            }
            for _, row in selected.iterrows()
        ]
        return pd.DataFrame(rows, columns=_BLOCKER_COLUMNS)

    def readiness_frame(self) -> pd.DataFrame:
        rows = [
            {
                "Readiness check": "Structural contract",
                "Ready": self.is_structurally_valid,
                "Interpretation": (
                    "All declarations and references are valid"
                    if self.is_structurally_valid
                    else "Structural contract issues must be corrected"
                ),
            },
            {
                "Readiness check": "Deterministic preparation",
                "Ready": self.is_ready_for_deterministic_preparation,
                "Interpretation": (
                    "Approved deterministic cleaning is traceable and bounded"
                    if self.is_ready_for_deterministic_preparation
                    else "Deterministic preparation scope is incomplete"
                ),
            },
            {
                "Readiness check": "Split execution",
                "Ready": self.is_ready_for_split_execution,
                "Interpretation": (
                    "Split policy is operationally resolved"
                    if self.is_ready_for_split_execution
                    else "Temporal precedence or split blockers remain unresolved"
                ),
            },
            {
                "Readiness check": "Modeling clearance",
                "Ready": self.is_ready_for_modeling,
                "Interpretation": (
                    "All blocking preparation and governance steps are complete"
                    if self.is_ready_for_modeling
                    else "Blocking steps or external prerequisites remain open"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_READINESS_COLUMNS)

    def issues_frame(self) -> pd.DataFrame:
        return _defensive_frame(self.issues)

    def raise_if_invalid(
        self,
        *,
        require_unique_decision_ids: bool = True,
        require_unique_evidence_ids: bool = True,
        require_unique_step_ids: bool = True,
        require_unique_guardrail_ids: bool = True,
        require_known_fields: bool = True,
        require_evidence_for_decisions: bool = True,
        require_acceptance_criteria: bool = True,
        require_valid_statuses: bool = True,
        require_valid_domains: bool = True,
        require_valid_phases: bool = True,
        require_valid_fit_scopes: bool = True,
        require_valid_references: bool = True,
        require_complete_split_policy: bool = True,
    ) -> None:
        selected: set[str] = set()
        if require_unique_decision_ids:
            selected.add("Duplicate decision ID")
        if require_unique_evidence_ids:
            selected.add("Duplicate evidence ID")
        if require_unique_step_ids:
            selected.add("Duplicate step ID")
        if require_unique_guardrail_ids:
            selected.add("Duplicate guardrail ID")
        if require_known_fields:
            selected.add("Unknown affected field")
        if require_evidence_for_decisions:
            selected.add("Decision without evidence")
        if require_acceptance_criteria:
            selected.update(
                {
                    "Missing decision acceptance criteria",
                    "Missing step acceptance criteria",
                }
            )
        if require_valid_statuses:
            selected.update(
                {
                    "Invalid decision status",
                    "Invalid step status",
                    "Invalid guardrail status",
                }
            )
        if require_valid_domains:
            selected.update(
                {
                    "Invalid decision domain",
                    "Invalid guardrail domain",
                }
            )
        if require_valid_phases:
            selected.update(
                {
                    "Invalid decision phase",
                    "Invalid step phase",
                }
            )
        if require_valid_fit_scopes:
            selected.add("Invalid fit scope")
        if require_valid_references:
            selected.update(
                {
                    "Unknown decision reference",
                    "Unknown prerequisite reference",
                }
            )
        if require_complete_split_policy:
            selected.update(
                {
                    "Incomplete split policy",
                    "Invalid split proportion",
                    "Invalid split total",
                    "Unknown stratification field",
                    "Invalid random seed",
                    "Invalid temporal policy status",
                    "Missing random-split fallback",
                    "Invalid temporal priority",
                    "Invalid test holdout contract",
                    "Invalid disjoint partition contract",
                    "Unknown grouping identifier",
                }
            )

        failures = self.issues.loc[self.issues["Issue"].isin(selected)]
        if failures.empty:
            return

        details = "; ".join(str(value) for value in failures["Details"])
        raise PreparationDecisionContractError(
            "Invalid preparation-decision contract: " + details
        )

    def raise_if_split_not_ready(
        self,
        *,
        require_temporal_policy_resolved: bool = True,
        require_no_blocked_split_decisions: bool = True,
        require_stratification_contract: bool = True,
        require_disjoint_partition_contract: bool = True,
    ) -> None:
        reasons: list[str] = []

        if not self.is_structurally_valid:
            reasons.append("the structural preparation-decision contract is invalid")

        if (
            require_temporal_policy_resolved
            and self.split_policy.get("temporal_policy_status") == "Unresolved"
        ):
            reasons.append("the temporal split policy remains unresolved")

        if require_no_blocked_split_decisions:
            blocked = self.decisions.loc[
                self.decisions["Status"].eq("Blocked")
                & self.decisions["Domain"].isin(
                    {"Dataset splitting", "Leakage governance"}
                )
            ]
            if not blocked.empty:
                reasons.append("blocked split or leakage-governance decisions remain")

        if require_stratification_contract:
            stratify_by = _text(self.split_policy.get("stratify_by"))
            if not stratify_by or stratify_by not in self.available_fields:
                reasons.append("the stratification contract is incomplete")

        if (
            require_disjoint_partition_contract
            and self.split_policy.get("disjoint_partitions_required") is not True
        ):
            reasons.append("the disjoint-partition contract is not enforced")

        if reasons:
            raise PreparationDecisionContractError(
                "Preparation split is not ready: " + "; ".join(reasons)
            )


def record_preparation_decisions(
    *,
    available_fields: Sequence[object],
    decisions: Sequence[Mapping[str, object]],
    evidence: Sequence[Mapping[str, object]],
    execution_steps: Sequence[Mapping[str, object]],
    guardrails: Sequence[Mapping[str, object]],
    split_policy: Mapping[str, object],
) -> PreparationDecisionReport:
    """Normalize and validate preliminary preparation decisions."""
    fields = _unique_text_tuple(available_fields)
    decision_declarations = deepcopy(list(decisions))
    evidence_declarations = deepcopy(list(evidence))
    step_declarations = deepcopy(list(execution_steps))
    guardrail_declarations = deepcopy(list(guardrails))
    split_declaration = deepcopy(dict(split_policy))

    issues: list[dict[str, object]] = []

    decision_rows = _normalize_decisions(
        decision_declarations,
        available_fields=fields,
        issues=issues,
    )
    evidence_rows = _normalize_evidence(evidence_declarations, issues=issues)
    step_rows = _normalize_steps(step_declarations, issues=issues)
    guardrail_rows = _normalize_guardrails(
        guardrail_declarations,
        available_fields=fields,
        issues=issues,
    )

    decision_ids = {str(row["Decision ID"]) for row in decision_rows}
    _validate_references_and_coverage(
        decision_rows=decision_rows,
        evidence_rows=evidence_rows,
        step_rows=step_rows,
        decision_ids=decision_ids,
        issues=issues,
    )
    _validate_split_policy(
        split_declaration,
        available_fields=fields,
        issues=issues,
    )

    evidence_counts: dict[str, int] = {}
    for row in evidence_rows:
        decision_id = str(row["Decision ID"])
        evidence_counts[decision_id] = evidence_counts.get(decision_id, 0) + 1

    step_counts: dict[str, int] = {}
    for row in step_rows:
        for decision_id in row["Decision IDs"]:
            key = str(decision_id)
            step_counts[key] = step_counts.get(key, 0) + 1

    for row in decision_rows:
        decision_id = str(row["Decision ID"])
        row["Evidence count"] = evidence_counts.get(decision_id, 0)
        row["Execution step count"] = step_counts.get(decision_id, 0)

    decisions_frame = pd.DataFrame(decision_rows, columns=_DECISION_COLUMNS)
    if not decisions_frame.empty:
        decisions_frame["_phase"] = decisions_frame["Phase"].map(
            _PHASE_ORDER
        ).fillna(99)
        decisions_frame["_status"] = decisions_frame["Status"].map(
            _STATUS_ORDER
        ).fillna(99)
        decisions_frame = (
            decisions_frame.sort_values(["_phase", "_status", "Decision ID"])
            .drop(columns=["_phase", "_status"])
            .reset_index(drop=True)
        )

    evidence_frame = pd.DataFrame(evidence_rows, columns=_EVIDENCE_COLUMNS)
    if not evidence_frame.empty:
        evidence_frame = evidence_frame.sort_values(
            ["Decision ID", "Evidence ID"]
        ).reset_index(drop=True)

    steps_frame = pd.DataFrame(step_rows, columns=_EXECUTION_COLUMNS)
    if not steps_frame.empty:
        steps_frame = steps_frame.sort_values(
            ["Sequence", "Step ID"]
        ).reset_index(drop=True)

    guardrails_frame = pd.DataFrame(guardrail_rows, columns=_GUARDRAIL_COLUMNS)
    if not guardrails_frame.empty:
        guardrails_frame = guardrails_frame.sort_values(
            ["Severity", "Guardrail ID"],
            key=lambda series: series.map(
                {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
            ) if series.name == "Severity" else series,
        ).reset_index(drop=True)

    issues_frame = pd.DataFrame(issues, columns=_ISSUE_COLUMNS)
    if not issues_frame.empty:
        issues_frame = issues_frame.sort_values(
            ["Scope", "Item", "Issue"]
        ).reset_index(drop=True)

    return PreparationDecisionReport(
        available_fields=fields,
        decisions=decisions_frame,
        evidence=evidence_frame,
        execution_steps=steps_frame,
        guardrails=guardrails_frame,
        split_policy=split_declaration,
        issues=issues_frame,
    )


def _normalize_decisions(
    declarations: Sequence[Mapping[str, object]],
    *,
    available_fields: tuple[str, ...],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    available = set(available_fields)

    for position, declaration in enumerate(declarations):
        fallback = f"decision[{position}]"
        decision_id = _text(declaration.get("decision_id")) or fallback
        if decision_id in seen:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Duplicate decision ID",
                    f"Decision ID {decision_id!r} is declared more than once",
                    "Evidence and execution steps cannot be linked deterministically",
                )
            )
        seen.add(decision_id)

        fields = _tuple_values(declaration.get("affected_fields", ()))
        for field in fields:
            if field not in available:
                issues.append(
                    _issue(
                        "Decision",
                        decision_id,
                        "Unknown affected field",
                        f"Affected field {field!r} is not available",
                        "The decision may target a non-existent variable",
                    )
                )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_STATUSES:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Invalid decision status",
                    f"Unsupported decision status {status!r}",
                    "Readiness cannot be determined reliably",
                )
            )

        domain = _text(declaration.get("domain"))
        if domain not in _ALLOWED_DOMAINS:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Invalid decision domain",
                    f"Unsupported decision domain {domain!r}",
                    "Scope summaries become unreliable",
                )
            )

        phase = _text(declaration.get("phase"))
        if phase not in _ALLOWED_PHASES:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Invalid decision phase",
                    f"Unsupported decision phase {phase!r}",
                    "Execution ordering becomes ambiguous",
                )
            )

        fit_scope = _text(declaration.get("fit_scope"))
        if fit_scope not in _ALLOWED_FIT_SCOPES:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Invalid fit scope",
                    f"Unsupported fit scope {fit_scope!r}",
                    "Train/test isolation cannot be audited",
                )
            )

        acceptance = _text(declaration.get("acceptance_criteria"))
        if not acceptance:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Missing decision acceptance criteria",
                    f"Decision {decision_id!r} has no measurable acceptance criteria",
                    "Implementation cannot be verified",
                )
            )

        prerequisites = _tuple_values(declaration.get("prerequisites", ()))
        rows.append(
            {
                "Decision ID": decision_id,
                "Domain": domain,
                "Title": _text(declaration.get("title")),
                "Affected fields": fields,
                "Affected field count": len(fields),
                "Status": status,
                "Phase": phase,
                "Fit scope": fit_scope,
                "Operation": _text(declaration.get("operation")),
                "Rationale": _text(declaration.get("rationale")),
                "Prerequisites": prerequisites,
                "Prerequisite count": len(prerequisites),
                "Acceptance criteria": acceptance,
                "Source stages": _tuple_values(
                    declaration.get("source_stages", ())
                ),
                "Evidence count": 0,
                "Execution step count": 0,
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
        fallback = f"evidence[{position}]"
        evidence_id = _text(declaration.get("evidence_id")) or fallback
        if evidence_id in seen:
            issues.append(
                _issue(
                    "Evidence",
                    evidence_id,
                    "Duplicate evidence ID",
                    f"Evidence ID {evidence_id!r} is declared more than once",
                    "Decision support cannot be traced deterministically",
                )
            )
        seen.add(evidence_id)

        rows.append(
            {
                "Evidence ID": evidence_id,
                "Decision ID": _text(declaration.get("decision_id")),
                "Source report": _text(declaration.get("source_report")),
                "Source item": _text(declaration.get("source_item")),
                "Observed value": deepcopy(declaration.get("observed_value")),
                "Expected or reference": deepcopy(
                    declaration.get("expected_or_reference")
                ),
                "Interpretation": _text(declaration.get("interpretation")),
            }
        )

    return rows


def _normalize_steps(
    declarations: Sequence[Mapping[str, object]],
    *,
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, declaration in enumerate(declarations):
        fallback = f"step[{position}]"
        step_id = _text(declaration.get("step_id")) or fallback
        if step_id in seen:
            issues.append(
                _issue(
                    "Execution step",
                    step_id,
                    "Duplicate step ID",
                    f"Step ID {step_id!r} is declared more than once",
                    "Execution order cannot be traced deterministically",
                )
            )
        seen.add(step_id)

        phase = _text(declaration.get("phase"))
        if phase not in _ALLOWED_PHASES:
            issues.append(
                _issue(
                    "Execution step",
                    step_id,
                    "Invalid step phase",
                    f"Unsupported execution phase {phase!r}",
                    "Execution ordering becomes ambiguous",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_STEP_STATUSES:
            issues.append(
                _issue(
                    "Execution step",
                    step_id,
                    "Invalid step status",
                    f"Unsupported execution-step status {status!r}",
                    "Readiness cannot be determined reliably",
                )
            )

        acceptance = _text(declaration.get("acceptance_criteria"))
        if not acceptance:
            issues.append(
                _issue(
                    "Execution step",
                    step_id,
                    "Missing step acceptance criteria",
                    f"Execution step {step_id!r} has no acceptance criteria",
                    "Future implementation cannot be verified",
                )
            )

        decision_ids = _tuple_values(declaration.get("decision_ids", ()))
        rows.append(
            {
                "Step ID": step_id,
                "Sequence": _integer(declaration.get("sequence"), default=position + 1),
                "Decision IDs": decision_ids,
                "Decision count": len(decision_ids),
                "Phase": phase,
                "Action": _text(declaration.get("action")),
                "Blocking": bool(declaration.get("blocking", False)),
                "Status": status,
                "Temporal dependency": bool(
                    declaration.get("temporal_dependency", False)
                ),
                "Acceptance criteria": acceptance,
            }
        )

    return rows


def _normalize_guardrails(
    declarations: Sequence[Mapping[str, object]],
    *,
    available_fields: tuple[str, ...],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    available = set(available_fields)

    for position, declaration in enumerate(declarations):
        fallback = f"guardrail[{position}]"
        guardrail_id = _text(declaration.get("guardrail_id")) or fallback
        if guardrail_id in seen:
            issues.append(
                _issue(
                    "Guardrail",
                    guardrail_id,
                    "Duplicate guardrail ID",
                    f"Guardrail ID {guardrail_id!r} is declared more than once",
                    "Prohibition coverage cannot be traced deterministically",
                )
            )
        seen.add(guardrail_id)

        fields = _tuple_values(declaration.get("affected_fields", ()))
        for field in fields:
            if field not in available:
                issues.append(
                    _issue(
                        "Guardrail",
                        guardrail_id,
                        "Unknown affected field",
                        f"Affected field {field!r} is not available",
                        "The guardrail may not protect the intended variable",
                    )
                )

        domain = _text(declaration.get("domain"))
        if domain not in _ALLOWED_DOMAINS:
            issues.append(
                _issue(
                    "Guardrail",
                    guardrail_id,
                    "Invalid guardrail domain",
                    f"Unsupported guardrail domain {domain!r}",
                    "Guardrail summaries become unreliable",
                )
            )

        severity = _text(declaration.get("severity"))
        if severity not in _ALLOWED_GUARDRAIL_SEVERITIES:
            issues.append(
                _issue(
                    "Guardrail",
                    guardrail_id,
                    "Invalid guardrail severity",
                    f"Unsupported guardrail severity {severity!r}",
                    "Risk priority cannot be interpreted",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_GUARDRAIL_STATUSES:
            issues.append(
                _issue(
                    "Guardrail",
                    guardrail_id,
                    "Invalid guardrail status",
                    f"Unsupported guardrail status {status!r}",
                    "Protection state cannot be interpreted",
                )
            )

        rows.append(
            {
                "Guardrail ID": guardrail_id,
                "Domain": domain,
                "Title": _text(declaration.get("title")),
                "Affected fields": fields,
                "Affected field count": len(fields),
                "Severity": severity,
                "Status": status,
                "Prohibited operation": _text(
                    declaration.get("prohibited_operation")
                ),
                "Rationale": _text(declaration.get("rationale")),
                "Verification": _text(declaration.get("verification")),
            }
        )

    return rows


def _validate_references_and_coverage(
    *,
    decision_rows: list[dict[str, object]],
    evidence_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
    decision_ids: set[str],
    issues: list[dict[str, object]],
) -> None:
    evidence_links: set[str] = set()
    for row in evidence_rows:
        decision_id = str(row["Decision ID"])
        if decision_id not in decision_ids:
            issues.append(
                _issue(
                    "Evidence",
                    str(row["Evidence ID"]),
                    "Unknown decision reference",
                    f"Decision ID {decision_id!r} does not exist",
                    "Evidence cannot support a declared decision",
                )
            )
        else:
            evidence_links.add(decision_id)

    for row in decision_rows:
        decision_id = str(row["Decision ID"])
        if decision_id not in evidence_links:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Decision without evidence",
                    f"Decision {decision_id!r} has no linked evidence record",
                    "The decision lacks traceable support",
                )
            )

        for prerequisite in row["Prerequisites"]:
            key = str(prerequisite)
            if key not in decision_ids:
                issues.append(
                    _issue(
                        "Decision",
                        decision_id,
                        "Unknown prerequisite reference",
                        f"Prerequisite decision {key!r} does not exist",
                        "Decision dependency order cannot be validated",
                    )
                )

    for row in step_rows:
        for decision_id in row["Decision IDs"]:
            key = str(decision_id)
            if key not in decision_ids:
                issues.append(
                    _issue(
                        "Execution step",
                        str(row["Step ID"]),
                        "Unknown decision reference",
                        f"Decision ID {key!r} does not exist",
                        "Execution step cannot be traced to a decision",
                    )
                )


def _validate_split_policy(
    split_policy: Mapping[str, object],
    *,
    available_fields: tuple[str, ...],
    issues: list[dict[str, object]],
) -> None:
    missing = [key for key in _SPLIT_REQUIRED_KEYS if key not in split_policy]
    if missing:
        issues.append(
            _issue(
                "Split policy",
                "policy",
                "Incomplete split policy",
                f"Missing required split-policy keys: {tuple(missing)!r}",
                "Partition behavior cannot be reproduced",
            )
        )

    fractions: list[float] = []
    for key in ("train_fraction", "validation_fraction", "test_fraction"):
        value = split_policy.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = -1.0
        if not 0.0 < numeric < 1.0:
            issues.append(
                _issue(
                    "Split policy",
                    key,
                    "Invalid split proportion",
                    f"{key} must be a number strictly between 0 and 1",
                    "A valid train/validation/test partition cannot be formed",
                )
            )
        fractions.append(numeric)

    if all(value > 0.0 for value in fractions) and abs(sum(fractions) - 1.0) > 1e-9:
        issues.append(
            _issue(
                "Split policy",
                "fractions",
                "Invalid split total",
                f"Split fractions sum to {sum(fractions):.12g}, not 1.0",
                "Rows may be omitted or assigned more than once",
            )
        )

    stratify_by = _text(split_policy.get("stratify_by"))
    if stratify_by not in set(available_fields):
        issues.append(
            _issue(
                "Split policy",
                "stratify_by",
                "Unknown stratification field",
                f"Stratification field {stratify_by!r} is not available",
                "Class representation cannot be guaranteed",
            )
        )

    seed = split_policy.get("random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        issues.append(
            _issue(
                "Split policy",
                "random_seed",
                "Invalid random seed",
                "Random seed must be an integer",
                "Random fallback cannot be reproduced",
            )
        )

    temporal_status = _text(split_policy.get("temporal_policy_status"))
    if temporal_status not in _ALLOWED_TEMPORAL_POLICY_STATUSES:
        issues.append(
            _issue(
                "Split policy",
                "temporal_policy_status",
                "Invalid temporal policy status",
                f"Unsupported temporal policy status {temporal_status!r}",
                "Temporal precedence cannot be evaluated",
            )
        )

    if not _text(split_policy.get("random_split_fallback")):
        issues.append(
            _issue(
                "Split policy",
                "random_split_fallback",
                "Missing random-split fallback",
                "The snapshot fallback rule is empty",
                "Random splitting may be used without source justification",
            )
        )

    if split_policy.get("temporal_priority") is not True:
        issues.append(
            _issue(
                "Split policy",
                "temporal_priority",
                "Invalid temporal priority",
                "Temporal split must explicitly take precedence when valid timing exists",
                "Evaluation may fail to represent future inference",
            )
        )

    if split_policy.get("test_holdout_untouched") is not True:
        issues.append(
            _issue(
                "Split policy",
                "test_holdout_untouched",
                "Invalid test holdout contract",
                "The final test partition must remain untouched until final evaluation",
                "Model selection may leak into final evaluation",
            )
        )

    if split_policy.get("disjoint_partitions_required") is not True:
        issues.append(
            _issue(
                "Split policy",
                "disjoint_partitions_required",
                "Invalid disjoint partition contract",
                "Train, validation, and test partitions must be disjoint",
                "The same observation may appear in multiple partitions",
            )
        )

    identifiers = _tuple_values(split_policy.get("group_by_identifiers", ()))
    for identifier in identifiers:
        if identifier not in set(available_fields):
            issues.append(
                _issue(
                    "Split policy",
                    "group_by_identifiers",
                    "Unknown grouping identifier",
                    f"Grouping identifier {identifier!r} is not available",
                    "Entity overlap cannot be checked reliably",
                )
            )



def _defensive_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame copy with independently copied object values."""
    result = frame.copy(deep=True)
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(deepcopy)
    return result

def _split_interpretation(
    key: str,
    value: object,
    temporal_status: str,
) -> str:
    interpretations = {
        "train_fraction": "Provisional share allocated to model fitting",
        "validation_fraction": "Provisional share reserved for model selection",
        "test_fraction": "Final holdout share reserved for one-time evaluation",
        "stratify_by": "Field used to preserve class representation in random fallback",
        "random_seed": "Seed used only for reproducible random fallback",
        "shuffle": "Random fallback shuffles rows before partitioning",
        "temporal_priority": "Chronological partitioning overrides random fallback when timing is valid",
        "temporal_policy_status": (
            "Temporal nature is unresolved and split execution remains blocked"
            if temporal_status == "Unresolved"
            else "Temporal partition strategy has an explicit resolution"
        ),
        "random_split_fallback": "Condition required before using the provisional random split",
        "test_holdout_untouched": "The final test set is excluded from all fitting and selection",
        "disjoint_partitions_required": "Partition indices and identifiers may not overlap",
        "group_by_identifiers": "Entity identifiers used to verify cross-partition isolation",
    }
    return interpretations.get(key, f"Declared value: {value!r}")


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
    try:
        values = list(value)  # type: ignore[arg-type]
    except TypeError:
        values = [value]
    return tuple(
        text
        for text in (_text(item) for item in values)
        if text
    )


def _unique_text_tuple(values: Sequence[object]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


def _integer(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
