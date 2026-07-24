"""Reusable, non-mutating analysis of categorical target distributions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_DISTRIBUTION_COLUMNS: Final[list[str]] = [
    "Class",
    "Count",
    "Percentage",
    "Expected",
    "Role",
]

_ISSUE_COLUMNS: Final[list[str]] = [
    "Issue",
    "Count",
    "Values",
    "Potential impact",
]


class TargetAnalysisError(ValueError):
    """Raised when target configuration or expectations are invalid."""


@dataclass(frozen=True, slots=True)
class TargetDistributionReport:
    """Summarize class counts, prevalence, imbalance, and target issues."""

    target: str
    row_count: int
    non_missing_count: int
    missing_count: int
    expected_classes: tuple[object, ...]
    positive_class: object | None
    unexpected_classes: tuple[object, ...]
    missing_expected_classes: tuple[object, ...]
    majority_classes: tuple[object, ...]
    minority_classes: tuple[object, ...]
    majority_count: int
    minority_count: int
    positive_class_count: int | None
    distribution: pd.DataFrame

    @property
    def class_count(self) -> int:
        """Return the number of classes observed at least once."""
        if self.distribution.empty:
            return 0
        return int((self.distribution["Count"] > 0).sum())

    @property
    def majority_class(self) -> object | None:
        """Return the unique majority class, or None when tied or absent."""
        if len(self.majority_classes) != 1:
            return None
        return self.majority_classes[0]

    @property
    def minority_class(self) -> object | None:
        """Return the unique minority class, or None when tied or absent."""
        if len(self.minority_classes) != 1:
            return None
        return self.minority_classes[0]

    @property
    def has_majority_tie(self) -> bool:
        """Return whether multiple classes share the largest count."""
        return len(self.majority_classes) > 1

    @property
    def has_minority_tie(self) -> bool:
        """Return whether multiple classes share the smallest positive count."""
        return len(self.minority_classes) > 1

    @property
    def majority_share(self) -> float | None:
        """Return the majority proportion among non-missing targets."""
        if self.non_missing_count == 0:
            return None
        return self.majority_count / self.non_missing_count

    @property
    def minority_share(self) -> float | None:
        """Return the minority proportion among non-missing targets."""
        if self.non_missing_count == 0:
            return None
        return self.minority_count / self.non_missing_count

    @property
    def imbalance_ratio(self) -> float | None:
        """Return majority count divided by smallest positive class count."""
        if self.minority_count <= 0:
            return None
        return self.majority_count / self.minority_count

    @property
    def positive_class_share(self) -> float | None:
        """Return positive-class prevalence among non-missing targets."""
        if (
            self.positive_class is None
            or self.positive_class_count is None
            or self.non_missing_count == 0
        ):
            return None
        return self.positive_class_count / self.non_missing_count

    @property
    def majority_baseline_accuracy(self) -> float | None:
        """Return the accuracy of always predicting a majority class."""
        return self.majority_share

    @property
    def has_missing_values(self) -> bool:
        """Return whether the target contains missing values."""
        return self.missing_count > 0

    @property
    def has_unexpected_classes(self) -> bool:
        """Return whether observed classes were not declared as expected."""
        return bool(self.unexpected_classes)

    @property
    def has_missing_expected_classes(self) -> bool:
        """Return whether declared classes were absent from observations."""
        return bool(self.missing_expected_classes)

    @property
    def has_issues(self) -> bool:
        """Return whether missing, absent, or unexpected target values exist."""
        return (
            self.has_missing_values
            or self.has_unexpected_classes
            or self.has_missing_expected_classes
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic target-distribution metrics."""

        def percent(value: float | None) -> str:
            if value is None:
                return "Not available"
            return f"{value:.4%}"

        def class_list(values: tuple[object, ...]) -> str:
            if not values:
                return "Not available"
            return ", ".join(repr(value) for value in values)

        rows = [
            {
                "Metric": "Total rows",
                "Value": self.row_count,
                "Interpretation": "All observations",
            },
            {
                "Metric": "Non-missing target values",
                "Value": self.non_missing_count,
                "Interpretation": "Rows used for class proportions",
            },
            {
                "Metric": "Missing target values",
                "Value": self.missing_count,
                "Interpretation": (
                    "Requires review"
                    if self.has_missing_values
                    else "No missing target values"
                ),
            },
            {
                "Metric": "Observed classes",
                "Value": self.class_count,
                "Interpretation": "Classes with at least one observation",
            },
            {
                "Metric": "Majority class",
                "Value": class_list(self.majority_classes),
                "Interpretation": (
                    "Tied majority"
                    if self.has_majority_tie
                    else "Most frequent observed class"
                ),
            },
            {
                "Metric": "Minority class",
                "Value": class_list(self.minority_classes),
                "Interpretation": (
                    "Tied minority"
                    if self.has_minority_tie
                    else "Least frequent observed class"
                ),
            },
            {
                "Metric": "Majority share",
                "Value": percent(self.majority_share),
                "Interpretation": "Majority-class baseline accuracy",
            },
            {
                "Metric": "Minority share",
                "Value": percent(self.minority_share),
                "Interpretation": "Smallest observed class prevalence",
            },
            {
                "Metric": "Majority-to-minority ratio",
                "Value": (
                    "Not available"
                    if self.imbalance_ratio is None
                    else round(self.imbalance_ratio, 4)
                ),
                "Interpretation": "Descriptive imbalance indicator",
            },
            {
                "Metric": "Positive class",
                "Value": (
                    "Not declared"
                    if self.positive_class is None
                    else repr(self.positive_class)
                ),
                "Interpretation": "Outcome treated as positive",
            },
            {
                "Metric": "Positive-class prevalence",
                "Value": percent(self.positive_class_share),
                "Interpretation": "Share of non-missing target values",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def distribution_frame(self) -> pd.DataFrame:
        """Return class counts and percentages in deterministic order."""
        return self.distribution.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return target conditions requiring review."""
        rows: list[dict[str, object]] = []

        if self.has_missing_values:
            rows.append(
                {
                    "Issue": "Missing target values",
                    "Count": self.missing_count,
                    "Values": "<missing>",
                    "Potential impact": (
                        "Rows without labels cannot support supervised "
                        "training or target-based evaluation."
                    ),
                }
            )

        if self.has_missing_expected_classes:
            rows.append(
                {
                    "Issue": "Missing expected classes",
                    "Count": len(self.missing_expected_classes),
                    "Values": ", ".join(
                        repr(value)
                        for value in self.missing_expected_classes
                    ),
                    "Potential impact": (
                        "The observed dataset does not cover every declared "
                        "outcome class."
                    ),
                }
            )

        if self.has_unexpected_classes:
            unexpected_count = int(
                self.distribution.loc[
                    self.distribution["Class"].isin(
                        self.unexpected_classes
                    ),
                    "Count",
                ].sum()
            )
            rows.append(
                {
                    "Issue": "Unexpected classes",
                    "Count": unexpected_count,
                    "Values": ", ".join(
                        repr(value) for value in self.unexpected_classes
                    ),
                    "Potential impact": (
                        "Unsupported labels may invalidate target encoding "
                        "and evaluation assumptions."
                    ),
                }
            )

        return pd.DataFrame(rows, columns=_ISSUE_COLUMNS)

    def raise_if_invalid(
        self,
        *,
        require_no_missing_target: bool = True,
        require_expected_classes_present: bool = True,
        require_no_unexpected_classes: bool = True,
    ) -> None:
        """Raise when configured target expectations are not satisfied."""
        failures: list[str] = []

        if require_no_missing_target and self.has_missing_values:
            failures.append(
                f"missing_target_values:{self.missing_count}"
            )

        if (
            require_expected_classes_present
            and self.has_missing_expected_classes
        ):
            failures.append(
                "missing_expected_classes:"
                + ",".join(
                    repr(value)
                    for value in self.missing_expected_classes
                )
            )

        if require_no_unexpected_classes and self.has_unexpected_classes:
            failures.append(
                "unexpected_classes:"
                + ",".join(
                    repr(value) for value in self.unexpected_classes
                )
            )

        if failures:
            raise TargetAnalysisError(
                "Target distribution validation failed: "
                + "; ".join(failures)
            )


def analyze_target_distribution(
    dataframe: pd.DataFrame,
    *,
    target: str,
    expected_classes: Sequence[object] | None = None,
    positive_class: object | None = None,
) -> TargetDistributionReport:
    """Analyze a categorical target without modifying the source DataFrame."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    target_name = _normalize_column_name(target, field="target")
    _require_unique_columns(dataframe)

    if target_name not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_name!r}")

    normalized_expected = _normalize_expected_classes(expected_classes)

    if positive_class is not None and _is_missing_scalar(positive_class):
        raise TargetAnalysisError(
            "positive_class must not be a missing value."
        )

    if (
        positive_class is not None
        and normalized_expected
        and not _contains_value(normalized_expected, positive_class)
    ):
        raise TargetAnalysisError(
            "positive_class must be included in expected_classes."
        )

    target_series = dataframe[target_name]
    missing_mask = target_series.isna()
    observed = target_series.loc[~missing_mask]

    observed_classes = tuple(pd.unique(observed))
    class_order = list(normalized_expected)
    class_order.extend(
        value
        for value in observed_classes
        if not _contains_value(class_order, value)
    )

    unexpected_classes = tuple(
        value
        for value in observed_classes
        if normalized_expected
        and not _contains_value(normalized_expected, value)
    )
    missing_expected_classes = tuple(
        value
        for value in normalized_expected
        if not _contains_value(observed_classes, value)
    )

    counts = [
        int(_value_mask(observed, class_value).sum())
        for class_value in class_order
    ]
    positive_counts = [count for count in counts if count > 0]

    if positive_counts:
        majority_count = max(positive_counts)
        minority_count = min(positive_counts)
        majority_classes = tuple(
            class_value
            for class_value, count in zip(
                class_order,
                counts,
                strict=True,
            )
            if count == majority_count
        )
        minority_classes = tuple(
            class_value
            for class_value, count in zip(
                class_order,
                counts,
                strict=True,
            )
            if count == minority_count
        )
    else:
        majority_count = 0
        minority_count = 0
        majority_classes = ()
        minority_classes = ()

    non_missing_count = int((~missing_mask).sum())
    distribution_rows: list[dict[str, object]] = []

    for class_value, count in zip(class_order, counts, strict=True):
        distribution_rows.append(
            {
                "Class": class_value,
                "Count": count,
                "Percentage": (
                    count / non_missing_count
                    if non_missing_count
                    else 0.0
                ),
                "Expected": (
                    True
                    if not normalized_expected
                    else _contains_value(
                        normalized_expected,
                        class_value,
                    )
                ),
                "Role": _class_role(
                    class_value=class_value,
                    count=count,
                    positive_class=positive_class,
                    majority_classes=majority_classes,
                    minority_classes=minority_classes,
                ),
            }
        )

    positive_class_count = None
    if positive_class is not None:
        positive_class_count = int(
            _value_mask(observed, positive_class).sum()
        )

    distribution = pd.DataFrame(
        distribution_rows,
        columns=_DISTRIBUTION_COLUMNS,
    )

    return TargetDistributionReport(
        target=target_name,
        row_count=len(dataframe),
        non_missing_count=non_missing_count,
        missing_count=int(missing_mask.sum()),
        expected_classes=normalized_expected,
        positive_class=positive_class,
        unexpected_classes=unexpected_classes,
        missing_expected_classes=missing_expected_classes,
        majority_classes=majority_classes,
        minority_classes=minority_classes,
        majority_count=majority_count,
        minority_count=minority_count,
        positive_class_count=positive_class_count,
        distribution=distribution,
    )


def _class_role(
    *,
    class_value: object,
    count: int,
    positive_class: object | None,
    majority_classes: tuple[object, ...],
    minority_classes: tuple[object, ...],
) -> str:
    roles: list[str] = []

    if count == 0:
        roles.append("Absent expected")
    elif _contains_value(majority_classes, class_value):
        if len(majority_classes) > 1:
            roles.append("Tied")
        else:
            roles.append("Majority")
    elif _contains_value(minority_classes, class_value):
        if len(minority_classes) > 1:
            roles.append("Tied minority")
        else:
            roles.append("Minority")
    else:
        roles.append("Intermediate")

    if positive_class is not None:
        roles.append(
            "Positive"
            if _values_equal(class_value, positive_class)
            else "Negative"
        )

    return " / ".join(roles)


def _normalize_column_name(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise TargetAnalysisError(f"{field} must not be empty.")
    return normalized


def _normalize_expected_classes(
    values: Sequence[object] | None,
) -> tuple[object, ...]:
    if values is None:
        return ()

    if isinstance(values, (str, bytes)):
        raise TypeError(
            "expected_classes must be a sequence of class values, "
            "not a string."
        )

    normalized = tuple(values)
    for value in normalized:
        if _is_missing_scalar(value):
            raise TargetAnalysisError(
                "expected_classes must not contain missing values."
            )

    for index, value in enumerate(normalized):
        if _contains_value(normalized[:index], value):
            raise TargetAnalysisError(
                f"expected_classes contains a duplicate value: {value!r}"
            )

    return normalized


def _require_unique_columns(dataframe: pd.DataFrame) -> None:
    duplicated = dataframe.columns[
        dataframe.columns.duplicated(keep=False)
    ].tolist()
    if duplicated:
        labels = ", ".join(repr(value) for value in duplicated)
        raise TargetAnalysisError(
            f"DataFrame contains duplicated column labels: {labels}"
        )


def _value_mask(series: pd.Series, value: object) -> pd.Series:
    try:
        mask = series.eq(value)
    except (TypeError, ValueError):
        mask = series.map(lambda item: _values_equal(item, value))

    return mask.fillna(False).astype(bool)


def _contains_value(values: Sequence[object], candidate: object) -> bool:
    return any(_values_equal(value, candidate) for value in values)


def _values_equal(left: object, right: object) -> bool:
    try:
        result = left == right
    except (TypeError, ValueError):
        return False

    if not pd.api.types.is_scalar(result):
        return False

    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _is_missing_scalar(value: object) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False

    if not pd.api.types.is_scalar(result):
        return False

    try:
        return bool(result)
    except (TypeError, ValueError):
        return False
