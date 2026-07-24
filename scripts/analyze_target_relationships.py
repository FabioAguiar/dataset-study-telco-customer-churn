"""Reusable, non-mutating analysis of feature-to-target relationships."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log, sqrt
from numbers import Integral, Real
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_NUMERICAL_RELATIONSHIP_COLUMNS: Final[list[str]] = [
    "Feature",
    "Valid paired rows",
    "Missing paired rows",
    "Positive-class count",
    "Negative-class count",
    "Positive-class mean",
    "Negative-class mean",
    "Mean difference",
    "Positive-class median",
    "Negative-class median",
    "Median difference",
    "Point-biserial correlation",
    "Absolute point-biserial correlation",
    "Cohen's d",
    "Absolute Cohen's d",
    "Eta squared",
    "Quantile positive-class rate spread",
    "Review flag",
    "Interpretation",
]

_NUMERICAL_CLASS_STATISTICS_COLUMNS: Final[list[str]] = [
    "Feature",
    "Target class",
    "Row count",
    "Valid numeric count",
    "Mean",
    "Median",
    "Standard deviation",
    "Minimum",
    "Maximum",
    "IQR",
]

_NUMERICAL_BIN_COLUMNS: Final[list[str]] = [
    "Feature",
    "Bin",
    "Bin order",
    "Lower bound",
    "Upper bound",
    "Row count",
    "Positive count",
    "Negative count",
    "Positive-class rate",
    "Overall positive-class rate",
    "Rate difference",
    "Lift",
    "Low-support flag",
]

_CATEGORICAL_RELATIONSHIP_COLUMNS: Final[list[str]] = [
    "Feature",
    "Valid paired rows",
    "Missing paired rows",
    "Observed categories",
    "Cramer's V",
    "U(Target | Feature)",
    "Minimum positive-class rate",
    "Maximum positive-class rate",
    "Positive-class rate spread",
    "Weighted absolute rate difference",
    "Low-support category count",
    "Review flag",
    "Interpretation",
]

_CATEGORICAL_RATE_COLUMNS: Final[list[str]] = [
    "Feature",
    "Category",
    "Category order",
    "Row count",
    "Positive count",
    "Negative count",
    "Positive-class rate",
    "Overall positive-class rate",
    "Rate difference",
    "Lift",
    "Odds ratio versus remaining categories",
    "Wilson interval lower",
    "Wilson interval upper",
    "Low-support flag",
    "Expected category",
]

_ISSUE_COLUMNS: Final[list[str]] = [
    "Scope",
    "Feature",
    "Issue",
    "Details",
    "Potential impact",
]


class FeatureTargetAnalysisError(ValueError):
    """Raised when feature-to-target configuration or data are invalid."""


@dataclass(frozen=True, slots=True)
class FeatureTargetRelationshipReport:
    """Summarize univariate numerical and categorical target evidence."""

    requested_numerical_features: tuple[str, ...]
    available_numerical_features: tuple[str, ...]
    missing_numerical_features: tuple[str, ...]
    requested_categorical_features: tuple[str, ...]
    available_categorical_features: tuple[str, ...]
    missing_categorical_features: tuple[str, ...]
    expected_target_classes: tuple[object, ...]
    observed_target_classes: tuple[object, ...]
    unexpected_target_classes: tuple[object, ...]
    missing_expected_target_classes: tuple[object, ...]
    positive_class: object
    row_count: int
    missing_target_count: int
    indices_aligned: bool
    numerical_effect_review_threshold: float
    categorical_association_review_threshold: float
    rate_difference_review_threshold: float
    minimum_group_count: int
    numerical_relationships: pd.DataFrame
    numerical_class_statistics: pd.DataFrame
    numerical_bins: pd.DataFrame
    categorical_relationships: pd.DataFrame
    categorical_rates: pd.DataFrame
    issues: pd.DataFrame

    @property
    def has_alignment_issues(self) -> bool:
        """Return whether feature projections and target indices differ."""
        return not self.indices_aligned

    @property
    def has_missing_features(self) -> bool:
        """Return whether any requested feature is absent."""
        return bool(
            self.missing_numerical_features
            or self.missing_categorical_features
        )

    @property
    def has_missing_target_values(self) -> bool:
        """Return whether the target contains missing values."""
        return self.missing_target_count > 0

    @property
    def has_unexpected_target_classes(self) -> bool:
        """Return whether undeclared target classes were observed."""
        return bool(self.unexpected_target_classes)

    @property
    def has_missing_expected_target_classes(self) -> bool:
        """Return whether a declared target class was absent."""
        return bool(self.missing_expected_target_classes)

    @property
    def has_constant_features(self) -> bool:
        """Return whether any analyzed feature lacks sufficient variation."""
        if self.issues.empty:
            return False
        return bool(
            self.issues["Issue"].isin(
                {
                    "Constant numerical feature",
                    "Constant categorical feature",
                }
            ).any()
        )

    @property
    def has_low_support_groups(self) -> bool:
        """Return whether any category or quantile has low support."""
        numerical = (
            False
            if self.numerical_bins.empty
            else bool(self.numerical_bins["Low-support flag"].any())
        )
        categorical = (
            False
            if self.categorical_rates.empty
            else bool(self.categorical_rates["Low-support flag"].any())
        )
        return numerical or categorical

    @property
    def has_numerical_review_candidates(self) -> bool:
        """Return whether any numerical feature meets a review criterion."""
        if self.numerical_relationships.empty:
            return False
        return bool(self.numerical_relationships["Review flag"].any())

    @property
    def has_categorical_review_candidates(self) -> bool:
        """Return whether any categorical feature meets a review criterion."""
        if self.categorical_relationships.empty:
            return False
        return bool(self.categorical_relationships["Review flag"].any())

    @property
    def is_analysis_ready(self) -> bool:
        """Return whether no structural issue blocks the analysis."""
        return not (
            self.has_alignment_issues
            or self.has_missing_features
            or self.has_missing_target_values
            or self.has_unexpected_target_classes
            or self.has_missing_expected_target_classes
            or len(self.expected_target_classes) != 2
            or self.positive_class not in self.expected_target_classes
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic feature-to-target summary metrics."""
        rows = [
            {
                "Metric": "Total rows",
                "Value": self.row_count,
                "Interpretation": "Observations supplied to the analysis",
            },
            {
                "Metric": "Requested numerical features",
                "Value": len(self.requested_numerical_features),
                "Interpretation": "Numerical features declared by the study",
            },
            {
                "Metric": "Available numerical features",
                "Value": len(self.available_numerical_features),
                "Interpretation": "Numerical features found in the projection",
            },
            {
                "Metric": "Requested categorical features",
                "Value": len(self.requested_categorical_features),
                "Interpretation": "Categorical features declared by the study",
            },
            {
                "Metric": "Available categorical features",
                "Value": len(self.available_categorical_features),
                "Interpretation": "Categorical features found in the projection",
            },
            {
                "Metric": "Projection indices aligned",
                "Value": self.indices_aligned,
                "Interpretation": (
                    "Features and target refer to matching observations"
                    if self.indices_aligned
                    else "Target relationships were not calculated"
                ),
            },
            {
                "Metric": "Missing target values",
                "Value": self.missing_target_count,
                "Interpretation": (
                    "Requires review"
                    if self.has_missing_target_values
                    else "No missing target values"
                ),
            },
            {
                "Metric": "Observed target classes",
                "Value": len(self.observed_target_classes),
                "Interpretation": ", ".join(
                    repr(value) for value in self.observed_target_classes
                ),
            },
            {
                "Metric": "Numerical review candidates",
                "Value": (
                    0
                    if self.numerical_relationships.empty
                    else int(self.numerical_relationships["Review flag"].sum())
                ),
                "Interpretation": (
                    "Effect size or churn-rate spread meets an exploratory "
                    "threshold"
                ),
            },
            {
                "Metric": "Categorical review candidates",
                "Value": (
                    0
                    if self.categorical_relationships.empty
                    else int(
                        self.categorical_relationships["Review flag"].sum()
                    )
                ),
                "Interpretation": (
                    "Association or churn-rate spread meets an exploratory "
                    "threshold"
                ),
            },
            {
                "Metric": "Low-support groups",
                "Value": (
                    int(self.numerical_bins["Low-support flag"].sum())
                    + int(self.categorical_rates["Low-support flag"].sum())
                ),
                "Interpretation": (
                    f"Groups with fewer than {self.minimum_group_count} rows"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def numerical_relationships_frame(self) -> pd.DataFrame:
        """Return consolidated numerical feature-to-target evidence."""
        return self.numerical_relationships.copy(deep=True)

    def numerical_class_statistics_frame(self) -> pd.DataFrame:
        """Return descriptive numerical statistics by target class."""
        return self.numerical_class_statistics.copy(deep=True)

    def numerical_bins_frame(self) -> pd.DataFrame:
        """Return positive-class rates across numerical quantile bins."""
        return self.numerical_bins.copy(deep=True)

    def categorical_relationships_frame(self) -> pd.DataFrame:
        """Return consolidated categorical feature-to-target evidence."""
        return self.categorical_relationships.copy(deep=True)

    def categorical_rates_frame(self) -> pd.DataFrame:
        """Return positive-class evidence for every category."""
        return self.categorical_rates.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return structural and variation conditions requiring review."""
        return self.issues.copy(deep=True)

    def raise_if_invalid(
        self,
        *,
        require_aligned_indices: bool = True,
        require_features_present: bool = True,
        require_unique_columns: bool = True,
        require_binary_target: bool = True,
        require_positive_class_present: bool = True,
        require_no_missing_target: bool = True,
        require_expected_target_classes: bool = True,
        require_no_unexpected_target_classes: bool = True,
        require_sufficient_variation: bool = False,
    ) -> None:
        """Raise when configured feature-to-target requirements fail."""
        failures: list[str] = []

        if require_aligned_indices and self.has_alignment_issues:
            failures.append("projection_indices_not_aligned")

        if require_features_present:
            if self.missing_numerical_features:
                failures.append(
                    "missing_numerical_features:"
                    + ",".join(self.missing_numerical_features)
                )
            if self.missing_categorical_features:
                failures.append(
                    "missing_categorical_features:"
                    + ",".join(self.missing_categorical_features)
                )

        if require_unique_columns and not self.issues.empty:
            if self.issues["Issue"].str.startswith("Duplicated ").any():
                failures.append("duplicated_column_labels")

        if require_binary_target and len(self.expected_target_classes) != 2:
            failures.append("target_contract_is_not_binary")

        if (
            require_positive_class_present
            and self.positive_class not in self.observed_target_classes
        ):
            failures.append("positive_class_not_observed")

        if require_no_missing_target and self.has_missing_target_values:
            failures.append(
                f"missing_target_values:{self.missing_target_count}"
            )

        if (
            require_expected_target_classes
            and self.has_missing_expected_target_classes
        ):
            failures.append(
                "missing_expected_target_classes:"
                + ",".join(
                    repr(value)
                    for value in self.missing_expected_target_classes
                )
            )

        if (
            require_no_unexpected_target_classes
            and self.has_unexpected_target_classes
        ):
            failures.append(
                "unexpected_target_classes:"
                + ",".join(
                    repr(value) for value in self.unexpected_target_classes
                )
            )

        if require_sufficient_variation and self.has_constant_features:
            failures.append("constant_features_detected")

        if failures:
            raise FeatureTargetAnalysisError(
                "Feature-to-target analysis is invalid: "
                + "; ".join(failures)
            )


def analyze_feature_target_relationships(
    numerical_frame: pd.DataFrame,
    categorical_frame: pd.DataFrame,
    target: pd.Series,
    *,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    expected_target_classes: Sequence[object],
    positive_class: object,
    expected_category_values: (
        Mapping[str, Sequence[object]] | None
    ) = None,
    numerical_bin_count: Integral = 10,
    minimum_group_count: Integral = 50,
    numerical_effect_review_threshold: Real = 0.50,
    categorical_association_review_threshold: Real = 0.20,
    rate_difference_review_threshold: Real = 0.10,
) -> FeatureTargetRelationshipReport:
    """Analyze every declared feature against a binary target."""
    _validate_dataframe(numerical_frame, name="numerical_frame")
    _validate_dataframe(categorical_frame, name="categorical_frame")
    _validate_series(target, name="target")

    requested_numerical = _normalize_feature_names(
        numerical_features,
        name="numerical_features",
    )
    requested_categorical = _normalize_feature_names(
        categorical_features,
        name="categorical_features",
    )
    expected_classes = _normalize_values(
        expected_target_classes,
        name="expected_target_classes",
    )
    if _is_missing_scalar(positive_class):
        raise FeatureTargetAnalysisError("positive_class cannot be missing.")
    if positive_class not in expected_classes:
        raise FeatureTargetAnalysisError(
            "positive_class must belong to expected_target_classes."
        )

    category_contract = _normalize_category_contract(
        expected_category_values or {},
        requested_categorical=requested_categorical,
    )

    if isinstance(numerical_bin_count, bool) or not isinstance(
        numerical_bin_count, Integral
    ):
        raise FeatureTargetAnalysisError(
            "numerical_bin_count must be an integer."
        )
    if int(numerical_bin_count) < 2:
        raise FeatureTargetAnalysisError(
            "numerical_bin_count must be at least 2."
        )
    if isinstance(minimum_group_count, bool) or not isinstance(
        minimum_group_count, Integral
    ):
        raise FeatureTargetAnalysisError(
            "minimum_group_count must be an integer."
        )
    if int(minimum_group_count) < 1:
        raise FeatureTargetAnalysisError(
            "minimum_group_count must be at least 1."
        )

    numerical_threshold = _validate_nonnegative_threshold(
        numerical_effect_review_threshold,
        name="numerical_effect_review_threshold",
    )
    categorical_threshold = _validate_unit_threshold(
        categorical_association_review_threshold,
        name="categorical_association_review_threshold",
    )
    rate_threshold = _validate_unit_threshold(
        rate_difference_review_threshold,
        name="rate_difference_review_threshold",
    )

    numerical_source = numerical_frame.copy(deep=True)
    categorical_source = categorical_frame.copy(deep=True)
    target_source = target.copy(deep=True)

    available_numerical = tuple(
        feature
        for feature in requested_numerical
        if feature in numerical_source.columns
    )
    missing_numerical = tuple(
        feature
        for feature in requested_numerical
        if feature not in numerical_source.columns
    )
    available_categorical = tuple(
        feature
        for feature in requested_categorical
        if feature in categorical_source.columns
    )
    missing_categorical = tuple(
        feature
        for feature in requested_categorical
        if feature not in categorical_source.columns
    )

    indices_aligned = (
        numerical_source.index.equals(categorical_source.index)
        and numerical_source.index.equals(target_source.index)
    )

    observed_classes = tuple(pd.unique(target_source.dropna()))
    unexpected_classes = tuple(
        value for value in observed_classes if value not in expected_classes
    )
    missing_expected_classes = tuple(
        value for value in expected_classes if value not in observed_classes
    )
    missing_target_count = int(target_source.isna().sum())

    issues: list[dict[str, object]] = []
    if not indices_aligned:
        issues.append(
            {
                "Scope": "Projection alignment",
                "Feature": None,
                "Issue": "Projection indices are not aligned",
                "Details": (
                    f"numerical rows={len(numerical_source)}, "
                    f"categorical rows={len(categorical_source)}, "
                    f"target rows={len(target_source)}"
                ),
                "Potential impact": (
                    "Features could be associated with target labels from "
                    "different observations."
                ),
            }
        )

    if numerical_source.columns.duplicated().any():
        issues.append(
            _duplicate_column_issue("numerical", numerical_source)
        )
    if categorical_source.columns.duplicated().any():
        issues.append(
            _duplicate_column_issue("categorical", categorical_source)
        )

    for feature in missing_numerical:
        issues.append(_missing_feature_issue("Numerical", feature))
    for feature in missing_categorical:
        issues.append(_missing_feature_issue("Categorical", feature))

    if missing_target_count:
        issues.append(
            {
                "Scope": "Target contract",
                "Feature": None,
                "Issue": "Missing target values",
                "Details": f"count={missing_target_count}",
                "Potential impact": (
                    "Unlabelled observations cannot support supervised "
                    "feature-to-target analysis."
                ),
            }
        )
    if unexpected_classes:
        issues.append(
            {
                "Scope": "Target contract",
                "Feature": None,
                "Issue": "Unexpected target classes",
                "Details": ", ".join(repr(v) for v in unexpected_classes),
                "Potential impact": (
                    "A binary positive-versus-negative interpretation would "
                    "collapse undeclared outcomes."
                ),
            }
        )
    if missing_expected_classes:
        issues.append(
            {
                "Scope": "Target contract",
                "Feature": None,
                "Issue": "Missing expected target classes",
                "Details": ", ".join(
                    repr(v) for v in missing_expected_classes
                ),
                "Potential impact": (
                    "Class-comparison statistics cannot represent the full "
                    "declared target contract."
                ),
            }
        )
    if len(expected_classes) != 2:
        issues.append(
            {
                "Scope": "Target contract",
                "Feature": None,
                "Issue": "Target contract is not binary",
                "Details": f"expected class count={len(expected_classes)}",
                "Potential impact": (
                    "The configured effect and positive-rate metrics require "
                    "exactly two declared target classes."
                ),
            }
        )

    has_duplicate_columns = (
        numerical_source.columns.duplicated().any()
        or categorical_source.columns.duplicated().any()
    )

    can_analyze = (
        indices_aligned
        and not has_duplicate_columns
        and len(expected_classes) == 2
        and not unexpected_classes
        and not missing_expected_classes
        and missing_target_count == 0
    )

    numerical_rows: list[dict[str, object]] = []
    class_statistic_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    categorical_rows: list[dict[str, object]] = []
    rate_rows: list[dict[str, object]] = []

    if can_analyze:
        overall_positive_rate = float(target_source.eq(positive_class).mean())

        for feature in available_numerical:
            relationship, class_rows, feature_bins, feature_issues = (
                _analyze_numerical_feature(
                    numerical_source[feature],
                    target_source,
                    feature=feature,
                    expected_classes=expected_classes,
                    positive_class=positive_class,
                    overall_positive_rate=overall_positive_rate,
                    bin_count=int(numerical_bin_count),
                    minimum_group_count=int(minimum_group_count),
                    effect_threshold=numerical_threshold,
                    rate_threshold=rate_threshold,
                )
            )
            numerical_rows.append(relationship)
            class_statistic_rows.extend(class_rows)
            bin_rows.extend(feature_bins)
            issues.extend(feature_issues)

        for feature in available_categorical:
            relationship, feature_rates, feature_issues = (
                _analyze_categorical_feature(
                    categorical_source[feature],
                    target_source,
                    feature=feature,
                    expected_categories=category_contract.get(feature, ()),
                    positive_class=positive_class,
                    overall_positive_rate=overall_positive_rate,
                    minimum_group_count=int(minimum_group_count),
                    association_threshold=categorical_threshold,
                    rate_threshold=rate_threshold,
                )
            )
            categorical_rows.append(relationship)
            rate_rows.extend(feature_rates)
            issues.extend(feature_issues)

    return FeatureTargetRelationshipReport(
        requested_numerical_features=requested_numerical,
        available_numerical_features=available_numerical,
        missing_numerical_features=missing_numerical,
        requested_categorical_features=requested_categorical,
        available_categorical_features=available_categorical,
        missing_categorical_features=missing_categorical,
        expected_target_classes=expected_classes,
        observed_target_classes=observed_classes,
        unexpected_target_classes=unexpected_classes,
        missing_expected_target_classes=missing_expected_classes,
        positive_class=positive_class,
        row_count=len(target_source),
        missing_target_count=missing_target_count,
        indices_aligned=indices_aligned,
        numerical_effect_review_threshold=numerical_threshold,
        categorical_association_review_threshold=categorical_threshold,
        rate_difference_review_threshold=rate_threshold,
        minimum_group_count=int(minimum_group_count),
        numerical_relationships=pd.DataFrame(
            numerical_rows,
            columns=_NUMERICAL_RELATIONSHIP_COLUMNS,
        ),
        numerical_class_statistics=pd.DataFrame(
            class_statistic_rows,
            columns=_NUMERICAL_CLASS_STATISTICS_COLUMNS,
        ),
        numerical_bins=pd.DataFrame(
            bin_rows,
            columns=_NUMERICAL_BIN_COLUMNS,
        ),
        categorical_relationships=pd.DataFrame(
            categorical_rows,
            columns=_CATEGORICAL_RELATIONSHIP_COLUMNS,
        ),
        categorical_rates=pd.DataFrame(
            rate_rows,
            columns=_CATEGORICAL_RATE_COLUMNS,
        ),
        issues=pd.DataFrame(issues, columns=_ISSUE_COLUMNS),
    )


def _analyze_numerical_feature(
    values: pd.Series,
    target: pd.Series,
    *,
    feature: str,
    expected_classes: tuple[object, ...],
    positive_class: object,
    overall_positive_rate: float,
    bin_count: int,
    minimum_group_count: int,
    effect_threshold: float,
    rate_threshold: float,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    numeric = pd.to_numeric(values, errors="coerce")
    paired = pd.DataFrame(
        {"value": numeric, "target": target},
        index=target.index,
    ).dropna()
    valid_rows = len(paired)
    missing_rows = len(target) - valid_rows
    negative_class = next(
        value for value in expected_classes if value != positive_class
    )
    positive_values = paired.loc[
        paired["target"].eq(positive_class), "value"
    ]
    negative_values = paired.loc[
        paired["target"].eq(negative_class), "value"
    ]

    issues: list[dict[str, object]] = []
    unique_count = int(paired["value"].nunique(dropna=True))
    if unique_count <= 1:
        issues.append(_constant_feature_issue("Numerical", feature))

    class_rows = [
        _numerical_class_statistics(
            feature=feature,
            target_class=target_class,
            all_target_rows=int(target.eq(target_class).sum()),
            values=paired.loc[paired["target"].eq(target_class), "value"],
        )
        for target_class in expected_classes
    ]

    positive_mean = _series_statistic(positive_values, "mean")
    negative_mean = _series_statistic(negative_values, "mean")
    positive_median = _series_statistic(positive_values, "median")
    negative_median = _series_statistic(negative_values, "median")
    mean_difference = _difference(positive_mean, negative_mean)
    median_difference = _difference(positive_median, negative_median)

    point_biserial: float | None = None
    if valid_rows >= 2 and unique_count > 1:
        binary = paired["target"].eq(positive_class).astype(float)
        point_biserial = _finite_or_none(paired["value"].corr(binary))

    cohens_d = _cohens_d(positive_values, negative_values)
    eta_squared = _eta_squared_binary(
        paired["value"],
        paired["target"],
    )
    bins = _numerical_quantile_bins(
        paired,
        feature=feature,
        positive_class=positive_class,
        overall_positive_rate=overall_positive_rate,
        bin_count=bin_count,
        minimum_group_count=minimum_group_count,
    )
    bin_rates = [
        row["Positive-class rate"]
        for row in bins
        if row["Positive-class rate"] is not None
    ]
    rate_spread = (
        max(bin_rates) - min(bin_rates) if bin_rates else None
    )

    absolute_d = None if cohens_d is None else abs(cohens_d)
    review = bool(
        (absolute_d is not None and absolute_d >= effect_threshold)
        or (rate_spread is not None and rate_spread >= rate_threshold)
    )
    if cohens_d is None:
        interpretation = "Insufficient variation for standardized effect"
    elif review:
        direction = "higher" if cohens_d > 0 else "lower"
        interpretation = (
            f"Positive class is associated with {direction} values; "
            "review for modeling"
        )
    elif abs(cohens_d) >= effect_threshold / 2:
        interpretation = "Limited-to-moderate numerical target separation"
    else:
        interpretation = "Limited numerical target separation"

    return (
        {
            "Feature": feature,
            "Valid paired rows": valid_rows,
            "Missing paired rows": missing_rows,
            "Positive-class count": len(positive_values),
            "Negative-class count": len(negative_values),
            "Positive-class mean": positive_mean,
            "Negative-class mean": negative_mean,
            "Mean difference": mean_difference,
            "Positive-class median": positive_median,
            "Negative-class median": negative_median,
            "Median difference": median_difference,
            "Point-biserial correlation": point_biserial,
            "Absolute point-biserial correlation": (
                None if point_biserial is None else abs(point_biserial)
            ),
            "Cohen's d": cohens_d,
            "Absolute Cohen's d": absolute_d,
            "Eta squared": eta_squared,
            "Quantile positive-class rate spread": rate_spread,
            "Review flag": review,
            "Interpretation": interpretation,
        },
        class_rows,
        bins,
        issues,
    )


def _numerical_class_statistics(
    *,
    feature: str,
    target_class: object,
    all_target_rows: int,
    values: pd.Series,
) -> dict[str, object]:
    valid_count = len(values)
    return {
        "Feature": feature,
        "Target class": target_class,
        "Row count": all_target_rows,
        "Valid numeric count": valid_count,
        "Mean": _series_statistic(values, "mean"),
        "Median": _series_statistic(values, "median"),
        "Standard deviation": _series_statistic(values, "std"),
        "Minimum": _series_statistic(values, "min"),
        "Maximum": _series_statistic(values, "max"),
        "IQR": (
            None
            if valid_count == 0
            else float(values.quantile(0.75) - values.quantile(0.25))
        ),
    }


def _numerical_quantile_bins(
    paired: pd.DataFrame,
    *,
    feature: str,
    positive_class: object,
    overall_positive_rate: float,
    bin_count: int,
    minimum_group_count: int,
) -> list[dict[str, object]]:
    unique_count = int(paired["value"].nunique(dropna=True))
    if unique_count <= 1 or paired.empty:
        return []

    requested_bins = min(bin_count, unique_count)
    try:
        quantiles = pd.qcut(
            paired["value"],
            q=requested_bins,
            duplicates="drop",
        )
    except ValueError:
        return []

    working = paired.assign(_bin=quantiles)
    categories = list(working["_bin"].cat.categories)
    rows: list[dict[str, object]] = []
    for order, interval in enumerate(categories, start=1):
        group = working.loc[working["_bin"].eq(interval)]
        row_count = len(group)
        positive_count = int(group["target"].eq(positive_class).sum())
        negative_count = row_count - positive_count
        rate = positive_count / row_count if row_count else None
        rows.append(
            {
                "Feature": feature,
                "Bin": f"Q{order}",
                "Bin order": order,
                "Lower bound": float(interval.left),
                "Upper bound": float(interval.right),
                "Row count": row_count,
                "Positive count": positive_count,
                "Negative count": negative_count,
                "Positive-class rate": rate,
                "Overall positive-class rate": overall_positive_rate,
                "Rate difference": (
                    None if rate is None else rate - overall_positive_rate
                ),
                "Lift": (
                    None
                    if rate is None or overall_positive_rate == 0
                    else rate / overall_positive_rate
                ),
                "Low-support flag": row_count < minimum_group_count,
            }
        )
    return rows


def _analyze_categorical_feature(
    values: pd.Series,
    target: pd.Series,
    *,
    feature: str,
    expected_categories: tuple[object, ...],
    positive_class: object,
    overall_positive_rate: float,
    minimum_group_count: int,
    association_threshold: float,
    rate_threshold: float,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    cleaned = _clean_categorical_series(values)
    paired = pd.DataFrame(
        {"category": cleaned, "target": target},
        index=target.index,
    ).dropna()
    valid_rows = len(paired)
    missing_rows = len(target) - valid_rows
    observed_categories = tuple(pd.unique(paired["category"]))
    ordered_categories = _ordered_categories(
        observed_categories,
        expected_categories,
    )
    cardinality = len(observed_categories)
    issues: list[dict[str, object]] = []
    if cardinality <= 1:
        issues.append(_constant_feature_issue("Categorical", feature))

    cramers_v: float | None = None
    uncertainty: float | None = None
    if valid_rows > 0 and cardinality > 1:
        contingency = pd.crosstab(
            paired["category"],
            paired["target"],
            dropna=True,
        )
        cramers_v = _cramers_v(contingency)
        uncertainty = _uncertainty_coefficient(
            target=paired["target"],
            predictor=paired["category"],
        )

    rate_rows: list[dict[str, object]] = []
    for order, category in enumerate(ordered_categories, start=1):
        category_mask = paired["category"].eq(category)
        category_target = paired.loc[category_mask, "target"]
        row_count = len(category_target)
        positive_count = int(category_target.eq(positive_class).sum())
        negative_count = row_count - positive_count
        rate = positive_count / row_count if row_count else None
        remainder_target = paired.loc[~category_mask, "target"]
        remainder_positive = int(
            remainder_target.eq(positive_class).sum()
        )
        remainder_negative = len(remainder_target) - remainder_positive
        odds_ratio = _odds_ratio(
            positive_count,
            negative_count,
            remainder_positive,
            remainder_negative,
        )
        wilson_lower, wilson_upper = _wilson_interval(
            positive_count,
            row_count,
        )
        rate_rows.append(
            {
                "Feature": feature,
                "Category": category,
                "Category order": order,
                "Row count": row_count,
                "Positive count": positive_count,
                "Negative count": negative_count,
                "Positive-class rate": rate,
                "Overall positive-class rate": overall_positive_rate,
                "Rate difference": (
                    None if rate is None else rate - overall_positive_rate
                ),
                "Lift": (
                    None
                    if rate is None or overall_positive_rate == 0
                    else rate / overall_positive_rate
                ),
                "Odds ratio versus remaining categories": odds_ratio,
                "Wilson interval lower": wilson_lower,
                "Wilson interval upper": wilson_upper,
                "Low-support flag": row_count < minimum_group_count,
                "Expected category": category in expected_categories,
            }
        )

    observed_rate_rows = [
        row for row in rate_rows if row["Row count"] > 0
    ]
    rates = [
        float(row["Positive-class rate"])
        for row in observed_rate_rows
        if row["Positive-class rate"] is not None
    ]
    minimum_rate = min(rates) if rates else None
    maximum_rate = max(rates) if rates else None
    rate_spread = (
        maximum_rate - minimum_rate
        if minimum_rate is not None and maximum_rate is not None
        else None
    )
    weighted_difference = (
        None
        if valid_rows == 0
        else sum(
            row["Row count"]
            * abs(float(row["Positive-class rate"]) - overall_positive_rate)
            for row in observed_rate_rows
            if row["Positive-class rate"] is not None
        )
        / valid_rows
    )
    low_support_count = sum(
        bool(row["Low-support flag"]) for row in observed_rate_rows
    )
    review = bool(
        (cramers_v is not None and cramers_v >= association_threshold)
        or (rate_spread is not None and rate_spread >= rate_threshold)
    )
    if cramers_v is None:
        interpretation = "Insufficient variation for categorical association"
    elif review:
        interpretation = "Category churn rates merit modeling review"
    elif cramers_v >= association_threshold / 2:
        interpretation = "Limited-to-moderate categorical target association"
    else:
        interpretation = "Limited categorical target association"

    return (
        {
            "Feature": feature,
            "Valid paired rows": valid_rows,
            "Missing paired rows": missing_rows,
            "Observed categories": cardinality,
            "Cramer's V": cramers_v,
            "U(Target | Feature)": uncertainty,
            "Minimum positive-class rate": minimum_rate,
            "Maximum positive-class rate": maximum_rate,
            "Positive-class rate spread": rate_spread,
            "Weighted absolute rate difference": weighted_difference,
            "Low-support category count": low_support_count,
            "Review flag": review,
            "Interpretation": interpretation,
        },
        rate_rows,
        issues,
    )


def _cohens_d(
    positive: pd.Series,
    negative: pd.Series,
) -> float | None:
    n_positive = len(positive)
    n_negative = len(negative)
    if n_positive < 2 or n_negative < 2:
        return None
    variance_positive = float(positive.var(ddof=1))
    variance_negative = float(negative.var(ddof=1))
    denominator_df = n_positive + n_negative - 2
    if denominator_df <= 0:
        return None
    pooled_variance = (
        (n_positive - 1) * variance_positive
        + (n_negative - 1) * variance_negative
    ) / denominator_df
    if pooled_variance <= 0:
        return None
    return _finite_or_none(
        (float(positive.mean()) - float(negative.mean()))
        / sqrt(pooled_variance)
    )


def _eta_squared_binary(
    values: pd.Series,
    groups: pd.Series,
) -> float | None:
    if len(values) == 0 or values.nunique(dropna=True) <= 1:
        return None
    grand_mean = float(values.mean())
    total = float(((values - grand_mean) ** 2).sum())
    if total <= 0:
        return None
    between = 0.0
    for _, group in values.groupby(groups, sort=False):
        if group.empty:
            continue
        between += len(group) * (float(group.mean()) - grand_mean) ** 2
    return _finite_or_none(between / total)


def _cramers_v(contingency: pd.DataFrame) -> float | None:
    observed = contingency.astype(float)
    total = float(observed.to_numpy().sum())
    rows, columns = observed.shape
    if total <= 0 or rows <= 1 or columns <= 1:
        return None
    row_totals = observed.sum(axis=1)
    column_totals = observed.sum(axis=0)
    chi_square = 0.0
    for row in observed.index:
        for column in observed.columns:
            expected = row_totals.loc[row] * column_totals.loc[column] / total
            if expected > 0:
                chi_square += (
                    (observed.loc[row, column] - expected) ** 2 / expected
                )
    denominator = total * min(rows - 1, columns - 1)
    if denominator <= 0:
        return None
    return _finite_or_none(sqrt(chi_square / denominator))


def _uncertainty_coefficient(
    *,
    target: pd.Series,
    predictor: pd.Series,
) -> float | None:
    target_entropy = _entropy(target)
    if target_entropy <= 0:
        return None
    conditional_entropy = 0.0
    total = len(target)
    for _, indices in predictor.groupby(predictor, sort=False).groups.items():
        subset = target.loc[indices]
        conditional_entropy += len(subset) / total * _entropy(subset)
    return _finite_or_none(
        max(0.0, min(1.0, (target_entropy - conditional_entropy) / target_entropy))
    )


def _entropy(series: pd.Series) -> float:
    probabilities = series.value_counts(normalize=True, dropna=True)
    return float(
        -sum(probability * log(probability) for probability in probabilities)
    )


def _odds_ratio(
    category_positive: int,
    category_negative: int,
    remainder_positive: int,
    remainder_negative: int,
) -> float | None:
    if (
        category_positive
        + category_negative
        + remainder_positive
        + remainder_negative
        == 0
    ):
        return None
    cells = [
        float(category_positive),
        float(category_negative),
        float(remainder_positive),
        float(remainder_negative),
    ]
    if any(value == 0 for value in cells):
        cells = [value + 0.5 for value in cells]
    numerator = cells[0] * cells[3]
    denominator = cells[1] * cells[2]
    if denominator == 0:
        return None
    return _finite_or_none(numerator / denominator)


def _wilson_interval(
    successes: int,
    total: int,
    *,
    z_score: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    z_squared = z_score**2
    denominator = 1 + z_squared / total
    centre = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        z_score
        * sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total**2)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _ordered_categories(
    observed: tuple[object, ...],
    expected: tuple[object, ...],
) -> tuple[object, ...]:
    ordered = list(expected)
    ordered.extend(value for value in observed if value not in expected)
    return tuple(ordered)


def _clean_categorical_series(series: pd.Series) -> pd.Series:
    cleaned = series.copy(deep=True)
    string_mask = cleaned.map(lambda value: isinstance(value, str))
    if bool(string_mask.any()):
        cleaned.loc[string_mask] = cleaned.loc[string_mask].map(str.strip)
        blank_mask = string_mask & cleaned.eq("")
        cleaned = cleaned.mask(blank_mask)
    return cleaned


def _series_statistic(series: pd.Series, statistic: str) -> float | None:
    if series.empty:
        return None
    value = getattr(series, statistic)()
    return _finite_or_none(value)


def _difference(
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _validate_dataframe(dataframe: pd.DataFrame, *, name: str) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")


def _validate_series(series: pd.Series, *, name: str) -> None:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")


def _normalize_feature_names(
    features: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(features, (str, bytes)):
        raise FeatureTargetAnalysisError(f"{name} must be a sequence of names.")
    normalized: list[str] = []
    for feature in features:
        if not isinstance(feature, str) or not feature.strip():
            raise FeatureTargetAnalysisError(
                f"{name} must contain non-empty strings."
            )
        normalized.append(feature.strip())
    duplicates = _find_duplicates(normalized)
    if duplicates:
        raise FeatureTargetAnalysisError(
            f"{name} contains duplicate names: {', '.join(duplicates)}."
        )
    return tuple(normalized)


def _normalize_values(
    values: Sequence[object],
    *,
    name: str,
) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise FeatureTargetAnalysisError(f"{name} must be a sequence.")
    normalized = tuple(values)
    if not normalized:
        raise FeatureTargetAnalysisError(f"{name} cannot be empty.")
    if any(_is_missing_scalar(value) for value in normalized):
        raise FeatureTargetAnalysisError(f"{name} cannot contain missing values.")
    for index, value in enumerate(normalized):
        if value in normalized[:index]:
            raise FeatureTargetAnalysisError(
                f"{name} contains duplicate value: {value!r}."
            )
    return normalized


def _normalize_category_contract(
    contract: Mapping[str, Sequence[object]],
    *,
    requested_categorical: tuple[str, ...],
) -> dict[str, tuple[object, ...]]:
    if not isinstance(contract, Mapping):
        raise TypeError("expected_category_values must be a mapping.")
    normalized: dict[str, tuple[object, ...]] = {}
    for feature, values in contract.items():
        if feature not in requested_categorical:
            raise FeatureTargetAnalysisError(
                "expected_category_values contains an undeclared feature: "
                f"{feature!r}."
            )
        normalized[feature] = _normalize_values(
            values,
            name=f"expected_category_values[{feature!r}]",
        )
    return normalized


def _validate_nonnegative_threshold(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FeatureTargetAnalysisError(f"{name} must be numeric.")
    normalized = float(value)
    if normalized < 0:
        raise FeatureTargetAnalysisError(f"{name} must be non-negative.")
    return normalized


def _validate_unit_threshold(value: Real, *, name: str) -> float:
    normalized = _validate_nonnegative_threshold(value, name=name)
    if normalized > 1:
        raise FeatureTargetAnalysisError(f"{name} must be at most 1.")
    return normalized


def _duplicate_column_issue(
    scope: str,
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    duplicates = tuple(
        dict.fromkeys(
            str(column)
            for column in dataframe.columns[dataframe.columns.duplicated()]
        )
    )
    return {
        "Scope": f"{scope.title()} projection",
        "Feature": None,
        "Issue": f"Duplicated {scope} column labels",
        "Details": ", ".join(duplicates),
        "Potential impact": "Feature selection would be ambiguous.",
    }


def _missing_feature_issue(scope: str, feature: str) -> dict[str, object]:
    return {
        "Scope": f"{scope} projection",
        "Feature": feature,
        "Issue": f"Missing {scope.casefold()} feature",
        "Details": feature,
        "Potential impact": "The declared feature could not be analyzed.",
    }


def _constant_feature_issue(scope: str, feature: str) -> dict[str, object]:
    return {
        "Scope": f"{scope} relationship",
        "Feature": feature,
        "Issue": f"Constant {scope.casefold()} feature",
        "Details": "At most one observed non-missing value",
        "Potential impact": (
            "Association or standardized separation cannot be estimated."
        ),
    }


def _finite_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    normalized = float(value)
    if normalized in {float("inf"), float("-inf")}:
        return None
    return normalized


def _is_missing_scalar(value: object) -> bool:
    result = pd.isna(value)
    if hasattr(result, "ndim") and getattr(result, "ndim") != 0:
        return False
    try:
        return bool(result)
    except TypeError:
        return False


def _find_duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)
