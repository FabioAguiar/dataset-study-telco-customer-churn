"""Reusable, non-mutating consolidation of initial data-quality findings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_FINDING_COLUMNS: Final[list[str]] = [
    "Finding ID",
    "Domain",
    "Title",
    "Affected fields",
    "Affected field count",
    "Severity",
    "Status",
    "Disposition",
    "Blocking scope",
    "Source stages",
    "Required action",
    "Verification",
]

_EVIDENCE_COLUMNS: Final[list[str]] = [
    "Evidence ID",
    "Finding ID",
    "Source report",
    "Source metric",
    "Observed value",
    "Expected value",
    "Interpretation",
]

_ACTION_COLUMNS: Final[list[str]] = [
    "Action ID",
    "Finding IDs",
    "Finding count",
    "Phase",
    "Action",
    "Fit scope",
    "Blocking",
    "Status",
    "Acceptance criteria",
]

_NON_ISSUE_COLUMNS: Final[list[str]] = [
    "Non-issue ID",
    "Domain",
    "Title",
    "Affected fields",
    "Affected field count",
    "Source stages",
    "Evidence",
    "Disposition",
    "Interpretation",
]

_BLOCKER_COLUMNS: Final[list[str]] = [
    "Finding ID",
    "Title",
    "Severity",
    "Status",
    "Disposition",
    "Blocking scope",
    "Linked actions",
    "Required action",
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

_ALLOWED_SEVERITIES: Final[frozenset[str]] = frozenset(
    {
        "Critical",
        "High",
        "Medium",
        "Low",
        "Informational",
    }
)

_ALLOWED_FINDING_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Open",
        "Controlled",
        "Accepted",
        "Monitor",
        "Review",
        "Validated",
    }
)

_ALLOWED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "Must fix",
        "Preserve",
        "Monitor",
        "Train-only",
        "Evaluate",
        "Exclude",
        "Prohibited",
        "External prerequisite",
        "No action",
    }
)

_ALLOWED_BLOCKING_SCOPES: Final[frozenset[str]] = frozenset(
    {
        "None",
        "Preparation",
        "Split",
        "Training pipeline",
        "Model evaluation",
        "Modeling clearance",
        "External contract",
    }
)

_ALLOWED_PHASES: Final[frozenset[str]] = frozenset(
    {
        "Before split",
        "Split",
        "Train-only transformation",
        "Model evaluation",
        "External contract",
    }
)

_ALLOWED_FIT_SCOPES: Final[frozenset[str]] = frozenset(
    {
        "Deterministic",
        "Split policy",
        "Training only",
        "Evaluation only",
        "External",
        "None",
    }
)

_ALLOWED_ACTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Pending",
        "Controlled",
        "Complete",
        "Blocked",
        "Accepted",
    }
)

_ALLOWED_NON_ISSUE_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "No action",
        "Preserve",
        "Monitor",
    }
)

_SEVERITY_ORDER: Final[dict[str, int]] = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
    "Informational": 4,
}

_PHASE_ORDER: Final[dict[str, int]] = {
    "Before split": 0,
    "Split": 1,
    "Train-only transformation": 2,
    "Model evaluation": 3,
    "External contract": 4,
}

_OPEN_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Open",
        "Monitor",
        "Review",
    }
)

_ACTION_REQUIRED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "Must fix",
        "Train-only",
        "Evaluate",
        "External prerequisite",
    }
)

_MODELING_BLOCKING_SCOPES: Final[frozenset[str]] = frozenset(
    {
        "Preparation",
        "Split",
        "Training pipeline",
        "Model evaluation",
        "Modeling clearance",
        "External contract",
    }
)

_EXTERNAL_BLOCKING_SCOPES: Final[frozenset[str]] = frozenset(
    {
        "External contract",
    }
)


class InitialDataQualityConsolidationError(ValueError):
    """Raised when finding structure or readiness requirements are invalid."""


@dataclass(frozen=True, slots=True)
class InitialDataQualityReport:
    """Consolidate data-quality evidence without applying data preparation."""

    available_fields: tuple[str, ...]
    row_count: int
    findings: pd.DataFrame
    evidence: pd.DataFrame
    preparation_actions: pd.DataFrame
    validated_non_issues: pd.DataFrame
    issues: pd.DataFrame

    @property
    def has_open_findings(self) -> bool:
        """Return whether any finding remains open, monitored, or in review."""
        if self.findings.empty:
            return False
        return bool(self.findings["Status"].isin(_OPEN_STATUSES).any())

    @property
    def has_must_fix_actions(self) -> bool:
        """Return whether an unresolved finding requires deterministic repair."""
        if self.findings.empty:
            return False
        unresolved = self.findings["Status"].isin(_OPEN_STATUSES)
        return bool(
            (
                unresolved
                & self.findings["Disposition"].eq("Must fix")
            ).any()
        )

    @property
    def has_external_blockers(self) -> bool:
        """Return whether an unresolved external prerequisite is recorded."""
        if self.findings.empty:
            return False
        return bool(
            (
                self.findings["Status"].isin(_OPEN_STATUSES)
                & self.findings["Blocking scope"].isin(
                    _EXTERNAL_BLOCKING_SCOPES
                )
            ).any()
        )

    @property
    def has_modeling_blockers(self) -> bool:
        """Return whether unresolved findings block safe model development."""
        if self.findings.empty:
            return False
        return bool(
            (
                self.findings["Status"].isin(_OPEN_STATUSES)
                & self.findings["Blocking scope"].isin(
                    _MODELING_BLOCKING_SCOPES
                )
            ).any()
        )

    @property
    def has_validated_non_issues(self) -> bool:
        """Return whether validated non-issues are documented."""
        return not self.validated_non_issues.empty

    @property
    def is_structurally_valid(self) -> bool:
        """Return whether declarations and references are structurally valid."""
        return self.issues.empty

    @property
    def is_safe_preparation_scope_defined(self) -> bool:
        """Return whether every actionable finding has a traceable action."""
        if not self.is_structurally_valid:
            return False

        actionable_findings = self.findings.loc[
            self.findings["Disposition"].isin(
                _ACTION_REQUIRED_DISPOSITIONS
            )
        ]
        if actionable_findings.empty:
            return True

        linked_findings: set[str] = set()
        for finding_ids in self.preparation_actions["Finding IDs"]:
            linked_findings.update(str(value) for value in finding_ids)

        required_ids = set(
            actionable_findings["Finding ID"].astype(str)
        )
        return required_ids.issubset(linked_findings)

    @property
    def is_modeling_ready(self) -> bool:
        """Return whether no unresolved finding or action blocks modeling."""
        if not self.is_safe_preparation_scope_defined:
            return False
        if self.has_modeling_blockers:
            return False

        if self.preparation_actions.empty:
            return True

        blocking_actions = self.preparation_actions.loc[
            self.preparation_actions["Blocking"]
        ]
        if blocking_actions.empty:
            return True

        unresolved_action_statuses = {"Pending", "Blocked"}
        return not bool(
            blocking_actions["Status"].isin(
                unresolved_action_statuses
            ).any()
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic counts and readiness indicators."""
        severity_counts = self.findings["Severity"].value_counts()
        rows = [
            {
                "Metric": "Dataset rows",
                "Value": self.row_count,
                "Interpretation": "Rows represented by the consolidated audit",
            },
            {
                "Metric": "Declared fields",
                "Value": len(self.available_fields),
                "Interpretation": "Fields available to referenced findings",
            },
            {
                "Metric": "Consolidated findings",
                "Value": len(self.findings),
                "Interpretation": "Quality, policy, and governance findings",
            },
            {
                "Metric": "Open findings",
                "Value": int(
                    self.findings["Status"].isin(_OPEN_STATUSES).sum()
                ),
                "Interpretation": "Findings still requiring action or review",
            },
            {
                "Metric": "Critical findings",
                "Value": int(severity_counts.get("Critical", 0)),
                "Interpretation": "Highest-priority governance conditions",
            },
            {
                "Metric": "High findings",
                "Value": int(severity_counts.get("High", 0)),
                "Interpretation": "High-priority preparation conditions",
            },
            {
                "Metric": "Must-fix findings",
                "Value": int(
                    self.findings["Disposition"].eq("Must fix").sum()
                ),
                "Interpretation": "Deterministic corrections required",
            },
            {
                "Metric": "Preparation actions",
                "Value": len(self.preparation_actions),
                "Interpretation": "Traceable actions with acceptance criteria",
            },
            {
                "Metric": "Validated non-issues",
                "Value": len(self.validated_non_issues),
                "Interpretation": "Conditions explicitly protected from over-cleaning",
            },
            {
                "Metric": "External blockers",
                "Value": int(self.has_external_blockers),
                "Interpretation": "Unresolved prerequisites outside data cleaning",
            },
            {
                "Metric": "Structurally valid",
                "Value": self.is_structurally_valid,
                "Interpretation": "Finding, evidence, and action contracts are coherent",
            },
            {
                "Metric": "Safe preparation scope defined",
                "Value": self.is_safe_preparation_scope_defined,
                "Interpretation": "Every actionable finding has a linked action",
            },
            {
                "Metric": "Modeling ready",
                "Value": self.is_modeling_ready,
                "Interpretation": "No unresolved blocker remains before modeling",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def findings_frame(self) -> pd.DataFrame:
        """Return a defensive copy of consolidated findings."""
        return self.findings.copy(deep=True)

    def evidence_frame(self) -> pd.DataFrame:
        """Return a defensive copy of finding evidence."""
        return self.evidence.copy(deep=True)

    def preparation_actions_frame(self) -> pd.DataFrame:
        """Return a defensive copy of preparation actions."""
        return self.preparation_actions.copy(deep=True)

    def blockers_frame(self) -> pd.DataFrame:
        """Return unresolved findings with a non-empty blocking scope."""
        if self.findings.empty:
            return pd.DataFrame(columns=_BLOCKER_COLUMNS)

        action_links: dict[str, list[str]] = {}
        for _, action in self.preparation_actions.iterrows():
            action_id = str(action["Action ID"])
            for finding_id in action["Finding IDs"]:
                action_links.setdefault(str(finding_id), []).append(action_id)

        selected = self.findings.loc[
            self.findings["Status"].isin(_OPEN_STATUSES)
            & self.findings["Blocking scope"].ne("None")
        ]

        rows = []
        for _, finding in selected.iterrows():
            finding_id = str(finding["Finding ID"])
            rows.append(
                {
                    "Finding ID": finding_id,
                    "Title": finding["Title"],
                    "Severity": finding["Severity"],
                    "Status": finding["Status"],
                    "Disposition": finding["Disposition"],
                    "Blocking scope": finding["Blocking scope"],
                    "Linked actions": tuple(action_links.get(finding_id, [])),
                    "Required action": finding["Required action"],
                }
            )

        return pd.DataFrame(rows, columns=_BLOCKER_COLUMNS)

    def validated_non_issues_frame(self) -> pd.DataFrame:
        """Return a defensive copy of validated non-issues."""
        return self.validated_non_issues.copy(deep=True)

    def readiness_frame(self) -> pd.DataFrame:
        """Return structural, preparation-scope, and modeling readiness."""
        rows = [
            {
                "Readiness check": "Structural contract",
                "Ready": self.is_structurally_valid,
                "Interpretation": (
                    "Finding, evidence, action, and field references are valid"
                    if self.is_structurally_valid
                    else "Structural issues must be corrected"
                ),
            },
            {
                "Readiness check": "Safe preparation scope",
                "Ready": self.is_safe_preparation_scope_defined,
                "Interpretation": (
                    "Actionable findings have linked implementation steps"
                    if self.is_safe_preparation_scope_defined
                    else "At least one actionable finding lacks a safe action"
                ),
            },
            {
                "Readiness check": "Modeling clearance",
                "Ready": self.is_modeling_ready,
                "Interpretation": (
                    "No unresolved preparation or external blocker remains"
                    if self.is_modeling_ready
                    else "Open blockers still prevent modeling clearance"
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
        require_unique_finding_ids: bool = True,
        require_unique_evidence_ids: bool = True,
        require_unique_action_ids: bool = True,
        require_known_fields: bool = True,
        require_evidence_for_findings: bool = True,
        require_action_for_open_findings: bool = True,
        require_acceptance_criteria: bool = True,
        require_valid_statuses: bool = True,
        require_valid_dispositions: bool = True,
        require_valid_severities: bool = True,
        require_valid_blocking_scopes: bool = True,
        require_valid_action_contracts: bool = True,
        require_valid_references: bool = True,
        require_unique_non_issue_ids: bool = True,
    ) -> None:
        """Raise when selected structural requirements are not satisfied."""
        selected_issue_names: set[str] = set()

        if require_unique_finding_ids:
            selected_issue_names.add("Duplicate finding ID")
        if require_unique_evidence_ids:
            selected_issue_names.add("Duplicate evidence ID")
        if require_unique_action_ids:
            selected_issue_names.add("Duplicate action ID")
        if require_known_fields:
            selected_issue_names.add("Unknown affected field")
        if require_evidence_for_findings:
            selected_issue_names.add("Finding without evidence")
        if require_action_for_open_findings:
            selected_issue_names.add("Actionable finding without action")
        if require_acceptance_criteria:
            selected_issue_names.add("Missing acceptance criteria")
        if require_valid_statuses:
            selected_issue_names.update(
                {
                    "Invalid finding status",
                    "Invalid action status",
                }
            )
        if require_valid_dispositions:
            selected_issue_names.update(
                {
                    "Invalid finding disposition",
                    "Invalid non-issue disposition",
                }
            )
        if require_valid_severities:
            selected_issue_names.add("Invalid severity")
        if require_valid_blocking_scopes:
            selected_issue_names.add("Invalid blocking scope")
        if require_valid_action_contracts:
            selected_issue_names.update(
                {
                    "Invalid action phase",
                    "Invalid fit scope",
                }
            )
        if require_valid_references:
            selected_issue_names.add("Unknown finding reference")
        if require_unique_non_issue_ids:
            selected_issue_names.add("Duplicate non-issue ID")

        relevant = self.issues.loc[
            self.issues["Issue"].isin(selected_issue_names)
        ]
        if relevant.empty:
            return

        descriptions = "; ".join(
            f"{row['Item']}: {row['Details']}"
            for _, row in relevant.iterrows()
        )
        raise InitialDataQualityConsolidationError(
            "Initial data-quality consolidation is invalid: " + descriptions
        )

    def raise_if_modeling_not_ready(
        self,
        *,
        require_no_modeling_blockers: bool = True,
        require_no_unresolved_external_prerequisites: bool = True,
        require_blocking_actions_complete: bool = True,
    ) -> None:
        """Raise when selected conditions still block modeling clearance."""
        failures: list[str] = []

        if not self.is_structurally_valid:
            failures.append("the consolidation structure is invalid")
        if not self.is_safe_preparation_scope_defined:
            failures.append("safe preparation actions are incomplete")
        if require_no_modeling_blockers and self.has_modeling_blockers:
            failures.append("unresolved modeling blockers remain")
        if (
            require_no_unresolved_external_prerequisites
            and self.has_external_blockers
        ):
            failures.append("external prerequisites remain unresolved")
        if require_blocking_actions_complete:
            blocking_actions = self.preparation_actions.loc[
                self.preparation_actions["Blocking"]
            ]
            incomplete = blocking_actions["Status"].isin(
                {"Pending", "Blocked"}
            )
            if bool(incomplete.any()):
                failures.append("blocking preparation actions are incomplete")

        if failures:
            raise InitialDataQualityConsolidationError(
                "Initial data-quality findings do not clear modeling: "
                + "; ".join(failures)
            )


def consolidate_initial_data_quality_findings(
    *,
    available_fields: Sequence[str],
    row_count: int,
    findings: Sequence[Mapping[str, object]],
    evidence: Sequence[Mapping[str, object]],
    preparation_actions: Sequence[Mapping[str, object]],
    validated_non_issues: Sequence[Mapping[str, object]] = (),
) -> InitialDataQualityReport:
    """Normalize and validate initial data-quality findings and actions."""
    fields = tuple(str(value) for value in available_fields)
    finding_rows, finding_issues = _normalize_findings(findings, fields)
    evidence_rows, evidence_issues = _normalize_evidence(evidence)
    action_rows, action_issues = _normalize_actions(preparation_actions)
    non_issue_rows, non_issue_issues = _normalize_non_issues(
        validated_non_issues,
        fields,
    )

    findings_frame = pd.DataFrame(finding_rows, columns=_FINDING_COLUMNS)
    evidence_frame = pd.DataFrame(evidence_rows, columns=_EVIDENCE_COLUMNS)
    action_frame = pd.DataFrame(action_rows, columns=_ACTION_COLUMNS)
    non_issue_frame = pd.DataFrame(
        non_issue_rows,
        columns=_NON_ISSUE_COLUMNS,
    )

    issues = [
        *finding_issues,
        *evidence_issues,
        *action_issues,
        *non_issue_issues,
    ]
    issues.extend(
        _reference_issues(
            findings_frame,
            evidence_frame,
            action_frame,
        )
    )

    findings_frame = _sort_findings(findings_frame)
    evidence_frame = _sort_evidence(evidence_frame, findings_frame)
    action_frame = _sort_actions(action_frame)
    non_issue_frame = _sort_non_issues(non_issue_frame)
    issues_frame = pd.DataFrame(issues, columns=_ISSUE_COLUMNS)
    if not issues_frame.empty:
        issues_frame = issues_frame.sort_values(
            by=["Scope", "Item", "Issue"],
            kind="stable",
        ).reset_index(drop=True)

    return InitialDataQualityReport(
        available_fields=fields,
        row_count=int(row_count),
        findings=findings_frame,
        evidence=evidence_frame,
        preparation_actions=action_frame,
        validated_non_issues=non_issue_frame,
        issues=issues_frame,
    )


def _normalize_findings(
    declarations: Sequence[Mapping[str, object]],
    available_fields: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    seen: set[str] = set()
    available = set(available_fields)

    for position, declaration in enumerate(declarations):
        item = f"finding[{position}]"
        finding_id = _text(declaration.get("finding_id"))
        if not finding_id:
            finding_id = item
        if finding_id in seen:
            issues.append(
                _issue(
                    "Finding",
                    finding_id,
                    "Duplicate finding ID",
                    f"Finding ID {finding_id!r} is declared more than once",
                    "Evidence and actions cannot be linked deterministically",
                )
            )
        seen.add(finding_id)

        affected_fields = _tuple_values(
            declaration.get("affected_fields", ())
        )
        for field in affected_fields:
            if field not in available:
                issues.append(
                    _issue(
                        "Finding",
                        finding_id,
                        "Unknown affected field",
                        f"Affected field {field!r} is not available",
                        "Preparation could target a non-existent column",
                    )
                )

        severity = _text(declaration.get("severity"))
        if severity not in _ALLOWED_SEVERITIES:
            issues.append(
                _issue(
                    "Finding",
                    finding_id,
                    "Invalid severity",
                    f"Unsupported severity {severity!r}",
                    "Priority ordering is not reliable",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_FINDING_STATUSES:
            issues.append(
                _issue(
                    "Finding",
                    finding_id,
                    "Invalid finding status",
                    f"Unsupported status {status!r}",
                    "Readiness cannot be evaluated consistently",
                )
            )

        disposition = _text(declaration.get("disposition"))
        if disposition not in _ALLOWED_DISPOSITIONS:
            issues.append(
                _issue(
                    "Finding",
                    finding_id,
                    "Invalid finding disposition",
                    f"Unsupported disposition {disposition!r}",
                    "Preparation behavior is ambiguous",
                )
            )

        blocking_scope = _text(
            declaration.get("blocking_scope", "None")
        )
        if blocking_scope not in _ALLOWED_BLOCKING_SCOPES:
            issues.append(
                _issue(
                    "Finding",
                    finding_id,
                    "Invalid blocking scope",
                    f"Unsupported blocking scope {blocking_scope!r}",
                    "Readiness blockers cannot be classified",
                )
            )

        rows.append(
            {
                "Finding ID": finding_id,
                "Domain": _text(declaration.get("domain")),
                "Title": _text(declaration.get("title")),
                "Affected fields": affected_fields,
                "Affected field count": len(affected_fields),
                "Severity": severity,
                "Status": status,
                "Disposition": disposition,
                "Blocking scope": blocking_scope,
                "Source stages": _tuple_values(
                    declaration.get("source_stages", ())
                ),
                "Required action": _text(
                    declaration.get("required_action")
                ),
                "Verification": _text(
                    declaration.get("verification")
                ),
            }
        )

    return rows, issues


def _normalize_evidence(
    declarations: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, declaration in enumerate(declarations):
        evidence_id = _text(declaration.get("evidence_id"))
        if not evidence_id:
            evidence_id = f"evidence[{position}]"
        if evidence_id in seen:
            issues.append(
                _issue(
                    "Evidence",
                    evidence_id,
                    "Duplicate evidence ID",
                    f"Evidence ID {evidence_id!r} is declared more than once",
                    "Evidence traceability is ambiguous",
                )
            )
        seen.add(evidence_id)

        rows.append(
            {
                "Evidence ID": evidence_id,
                "Finding ID": _text(declaration.get("finding_id")),
                "Source report": _text(
                    declaration.get("source_report")
                ),
                "Source metric": _text(
                    declaration.get("source_metric")
                ),
                "Observed value": declaration.get("observed_value"),
                "Expected value": declaration.get("expected_value"),
                "Interpretation": _text(
                    declaration.get("interpretation")
                ),
            }
        )

    return rows, issues


def _normalize_actions(
    declarations: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, declaration in enumerate(declarations):
        action_id = _text(declaration.get("action_id"))
        if not action_id:
            action_id = f"action[{position}]"
        if action_id in seen:
            issues.append(
                _issue(
                    "Action",
                    action_id,
                    "Duplicate action ID",
                    f"Action ID {action_id!r} is declared more than once",
                    "Implementation sequencing is ambiguous",
                )
            )
        seen.add(action_id)

        phase = _text(declaration.get("phase"))
        if phase not in _ALLOWED_PHASES:
            issues.append(
                _issue(
                    "Action",
                    action_id,
                    "Invalid action phase",
                    f"Unsupported phase {phase!r}",
                    "Preparation order is not deterministic",
                )
            )

        fit_scope = _text(declaration.get("fit_scope"))
        if fit_scope not in _ALLOWED_FIT_SCOPES:
            issues.append(
                _issue(
                    "Action",
                    action_id,
                    "Invalid fit scope",
                    f"Unsupported fit scope {fit_scope!r}",
                    "Leakage controls cannot be evaluated",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_ACTION_STATUSES:
            issues.append(
                _issue(
                    "Action",
                    action_id,
                    "Invalid action status",
                    f"Unsupported action status {status!r}",
                    "Action readiness cannot be evaluated",
                )
            )

        acceptance_criteria = _text(
            declaration.get("acceptance_criteria")
        )
        if not acceptance_criteria:
            issues.append(
                _issue(
                    "Action",
                    action_id,
                    "Missing acceptance criteria",
                    "The action has no measurable acceptance criteria",
                    "Completion cannot be verified",
                )
            )

        finding_ids = _tuple_values(
            declaration.get("finding_ids", ())
        )
        rows.append(
            {
                "Action ID": action_id,
                "Finding IDs": finding_ids,
                "Finding count": len(finding_ids),
                "Phase": phase,
                "Action": _text(declaration.get("action")),
                "Fit scope": fit_scope,
                "Blocking": bool(declaration.get("blocking", False)),
                "Status": status,
                "Acceptance criteria": acceptance_criteria,
            }
        )

    return rows, issues


def _normalize_non_issues(
    declarations: Sequence[Mapping[str, object]],
    available_fields: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    seen: set[str] = set()
    available = set(available_fields)

    for position, declaration in enumerate(declarations):
        non_issue_id = _text(declaration.get("non_issue_id"))
        if not non_issue_id:
            non_issue_id = f"non_issue[{position}]"
        if non_issue_id in seen:
            issues.append(
                _issue(
                    "Non-issue",
                    non_issue_id,
                    "Duplicate non-issue ID",
                    f"Non-issue ID {non_issue_id!r} is repeated",
                    "Validated conditions cannot be referenced uniquely",
                )
            )
        seen.add(non_issue_id)

        affected_fields = _tuple_values(
            declaration.get("affected_fields", ())
        )
        for field in affected_fields:
            if field not in available:
                issues.append(
                    _issue(
                        "Non-issue",
                        non_issue_id,
                        "Unknown affected field",
                        f"Affected field {field!r} is not available",
                        "A non-issue may refer to a non-existent column",
                    )
                )

        disposition = _text(
            declaration.get("disposition", "No action")
        )
        if disposition not in _ALLOWED_NON_ISSUE_DISPOSITIONS:
            issues.append(
                _issue(
                    "Non-issue",
                    non_issue_id,
                    "Invalid non-issue disposition",
                    f"Unsupported disposition {disposition!r}",
                    "Protection against over-cleaning is ambiguous",
                )
            )

        rows.append(
            {
                "Non-issue ID": non_issue_id,
                "Domain": _text(declaration.get("domain")),
                "Title": _text(declaration.get("title")),
                "Affected fields": affected_fields,
                "Affected field count": len(affected_fields),
                "Source stages": _tuple_values(
                    declaration.get("source_stages", ())
                ),
                "Evidence": _text(declaration.get("evidence")),
                "Disposition": disposition,
                "Interpretation": _text(
                    declaration.get("interpretation")
                ),
            }
        )

    return rows, issues


def _reference_issues(
    findings: pd.DataFrame,
    evidence: pd.DataFrame,
    actions: pd.DataFrame,
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    finding_ids = set(findings["Finding ID"].astype(str))
    evidence_ids = set(evidence["Finding ID"].astype(str))

    for finding_id in sorted(evidence_ids - finding_ids):
        issues.append(
            _issue(
                "Evidence",
                finding_id,
                "Unknown finding reference",
                f"Evidence references unknown finding {finding_id!r}",
                "Evidence cannot be attached to a finding",
            )
        )

    for _, action in actions.iterrows():
        action_id = str(action["Action ID"])
        for finding_id in action["Finding IDs"]:
            if str(finding_id) not in finding_ids:
                issues.append(
                    _issue(
                        "Action",
                        action_id,
                        "Unknown finding reference",
                        f"Action references unknown finding {finding_id!r}",
                        "The action cannot be traced to a finding",
                    )
                )

    for _, finding in findings.iterrows():
        finding_id = str(finding["Finding ID"])
        if finding_id not in evidence_ids:
            issues.append(
                _issue(
                    "Finding",
                    finding_id,
                    "Finding without evidence",
                    "No evidence record references this finding",
                    "The finding is not traceable to observed evidence",
                )
            )

    actionable = findings.loc[
        findings["Disposition"].isin(_ACTION_REQUIRED_DISPOSITIONS)
    ]
    linked_action_ids: set[str] = set()
    for finding_ids in actions["Finding IDs"]:
        linked_action_ids.update(str(value) for value in finding_ids)

    for finding_id in actionable["Finding ID"].astype(str):
        if finding_id not in linked_action_ids:
            issues.append(
                _issue(
                    "Finding",
                    finding_id,
                    "Actionable finding without action",
                    "No preparation action references this finding",
                    "The preparation scope is incomplete",
                )
            )

    return issues


def _sort_findings(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["_severity_order"] = result["Severity"].map(
        _SEVERITY_ORDER
    ).fillna(len(_SEVERITY_ORDER))
    result = result.sort_values(
        by=["_severity_order", "Finding ID"],
        kind="stable",
    ).drop(columns="_severity_order")
    return result.reset_index(drop=True)


def _sort_evidence(
    frame: pd.DataFrame,
    findings: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    order = {
        str(finding_id): position
        for position, finding_id in enumerate(findings["Finding ID"])
    }
    result = frame.copy()
    result["_finding_order"] = result["Finding ID"].map(order).fillna(
        len(order)
    )
    result = result.sort_values(
        by=["_finding_order", "Evidence ID"],
        kind="stable",
    ).drop(columns="_finding_order")
    return result.reset_index(drop=True)


def _sort_actions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["_phase_order"] = result["Phase"].map(_PHASE_ORDER).fillna(
        len(_PHASE_ORDER)
    )
    result = result.sort_values(
        by=["_phase_order", "Action ID"],
        kind="stable",
    ).drop(columns="_phase_order")
    return result.reset_index(drop=True)


def _sort_non_issues(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        by=["Domain", "Non-issue ID"],
        kind="stable",
    ).reset_index(drop=True)


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tuple_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


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
