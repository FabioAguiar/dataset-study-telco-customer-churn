"""Tests for reusable categorical target-distribution analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_target import (
    TargetAnalysisError,
    analyze_target_distribution,
)


def _analyze(
    values: list[object],
    **kwargs: object,
):
    dataframe = pd.DataFrame({"Churn": values})
    return analyze_target_distribution(
        dataframe,
        target="Churn",
        expected_classes=("No", "Yes"),
        positive_class="Yes",
        **kwargs,
    )


def test_balanced_binary_target_reports_tie() -> None:
    report = _analyze(["No", "Yes", "No", "Yes"])

    assert report.row_count == 4
    assert report.non_missing_count == 4
    assert report.missing_count == 0
    assert report.class_count == 2
    assert report.majority_classes == ("No", "Yes")
    assert report.minority_classes == ("No", "Yes")
    assert report.majority_class is None
    assert report.minority_class is None
    assert report.has_majority_tie
    assert report.has_minority_tie
    assert report.imbalance_ratio == pytest.approx(1.0)
    assert report.majority_baseline_accuracy == pytest.approx(0.5)
    assert report.positive_class_share == pytest.approx(0.5)

    distribution = report.distribution_frame()
    assert list(distribution["Class"]) == ["No", "Yes"]
    assert list(distribution["Count"]) == [2, 2]
    assert list(distribution["Role"]) == [
        "Tied / Negative",
        "Tied / Positive",
    ]


def test_imbalanced_binary_target_reports_majority_and_minority() -> None:
    report = _analyze(["No", "No", "No", "Yes"])

    assert report.majority_class == "No"
    assert report.minority_class == "Yes"
    assert report.majority_share == pytest.approx(0.75)
    assert report.minority_share == pytest.approx(0.25)
    assert report.imbalance_ratio == pytest.approx(3.0)
    assert report.positive_class_count == 1
    assert report.positive_class_share == pytest.approx(0.25)
    assert report.majority_baseline_accuracy == pytest.approx(0.75)

    distribution = report.distribution_frame()
    assert list(distribution["Percentage"]) == [0.75, 0.25]
    assert list(distribution["Role"]) == [
        "Majority / Negative",
        "Minority / Positive",
    ]


def test_multiclass_target_supports_intermediate_class() -> None:
    dataframe = pd.DataFrame(
        {"Outcome": ["A", "A", "A", "B", "B", "C"]}
    )

    report = analyze_target_distribution(
        dataframe,
        target="Outcome",
        expected_classes=("A", "B", "C"),
        positive_class="C",
    )

    assert report.class_count == 3
    assert report.majority_class == "A"
    assert report.minority_class == "C"
    assert report.imbalance_ratio == pytest.approx(3.0)
    assert list(report.distribution_frame()["Role"]) == [
        "Majority / Negative",
        "Intermediate / Negative",
        "Minority / Positive",
    ]


def test_expected_class_order_is_preserved() -> None:
    report = _analyze(["Yes", "No", "Yes"])

    assert list(report.distribution_frame()["Class"]) == ["No", "Yes"]


def test_observed_order_is_preserved_without_expected_classes() -> None:
    dataframe = pd.DataFrame({"Outcome": ["B", "A", "C", "A"]})

    report = analyze_target_distribution(
        dataframe,
        target="Outcome",
    )

    assert list(report.distribution_frame()["Class"]) == ["B", "A", "C"]
    assert list(report.distribution_frame()["Count"]) == [1, 2, 1]


def test_missing_target_values_are_reported_and_excluded_from_shares() -> None:
    report = _analyze(["No", None, "Yes", pd.NA])

    assert report.row_count == 4
    assert report.non_missing_count == 2
    assert report.missing_count == 2
    assert report.has_missing_values
    assert report.positive_class_share == pytest.approx(0.5)

    issues = report.issues_frame()
    assert list(issues["Issue"]) == ["Missing target values"]
    assert issues.iloc[0]["Count"] == 2

    with pytest.raises(
        TargetAnalysisError,
        match="missing_target_values:2",
    ):
        report.raise_if_invalid()


def test_missing_expected_class_is_reported_with_zero_count() -> None:
    report = _analyze(["No", "No"])

    assert report.has_missing_expected_classes
    assert report.missing_expected_classes == ("Yes",)
    assert report.positive_class_count == 0
    assert report.positive_class_share == pytest.approx(0.0)

    distribution = report.distribution_frame()
    yes_row = distribution.loc[distribution["Class"] == "Yes"].iloc[0]
    assert yes_row["Count"] == 0
    assert yes_row["Role"] == "Absent expected / Positive"

    with pytest.raises(
        TargetAnalysisError,
        match="missing_expected_classes:'Yes'",
    ):
        report.raise_if_invalid()


def test_unexpected_class_is_appended_and_reported() -> None:
    report = _analyze(["No", "Yes", "Unknown"])

    assert report.has_unexpected_classes
    assert report.unexpected_classes == ("Unknown",)
    assert list(report.distribution_frame()["Class"]) == [
        "No",
        "Yes",
        "Unknown",
    ]

    issues = report.issues_frame()
    assert issues.iloc[0]["Issue"] == "Unexpected classes"
    assert issues.iloc[0]["Count"] == 1

    with pytest.raises(
        TargetAnalysisError,
        match="unexpected_classes:'Unknown'",
    ):
        report.raise_if_invalid()


def test_validation_flags_can_allow_observed_issues() -> None:
    report = _analyze(["No", None, "Unknown"])

    report.raise_if_invalid(
        require_no_missing_target=False,
        require_expected_classes_present=False,
        require_no_unexpected_classes=False,
    )


def test_empty_dataframe_produces_an_empty_observed_distribution() -> None:
    dataframe = pd.DataFrame({"Churn": pd.Series(dtype="object")})

    report = analyze_target_distribution(
        dataframe,
        target="Churn",
        expected_classes=("No", "Yes"),
        positive_class="Yes",
    )

    assert report.row_count == 0
    assert report.non_missing_count == 0
    assert report.class_count == 0
    assert report.majority_class is None
    assert report.minority_class is None
    assert report.majority_share is None
    assert report.minority_share is None
    assert report.imbalance_ratio is None
    assert report.positive_class_share is None
    assert list(report.distribution_frame()["Count"]) == [0, 0]
    assert report.missing_expected_classes == ("No", "Yes")


def test_target_column_must_exist() -> None:
    dataframe = pd.DataFrame({"Outcome": ["No"]})

    with pytest.raises(KeyError, match="Target column not found"):
        analyze_target_distribution(
            dataframe,
            target="Churn",
        )


def test_duplicated_column_labels_are_rejected() -> None:
    dataframe = pd.DataFrame(
        [["No", "Yes"]],
        columns=["Churn", "Churn"],
    )

    with pytest.raises(
        TargetAnalysisError,
        match="duplicated column labels",
    ):
        analyze_target_distribution(
            dataframe,
            target="Churn",
        )


def test_expected_classes_must_be_unique_and_non_missing() -> None:
    dataframe = pd.DataFrame({"Churn": ["No", "Yes"]})

    with pytest.raises(
        TargetAnalysisError,
        match="duplicate value",
    ):
        analyze_target_distribution(
            dataframe,
            target="Churn",
            expected_classes=("No", "No"),
        )

    with pytest.raises(
        TargetAnalysisError,
        match="missing values",
    ):
        analyze_target_distribution(
            dataframe,
            target="Churn",
            expected_classes=("No", None),
        )


def test_positive_class_must_belong_to_expected_classes() -> None:
    dataframe = pd.DataFrame({"Churn": ["No", "Yes"]})

    with pytest.raises(
        TargetAnalysisError,
        match="positive_class must be included",
    ):
        analyze_target_distribution(
            dataframe,
            target="Churn",
            expected_classes=("No", "Yes"),
            positive_class="Maybe",
        )


def test_summary_and_distribution_frames_are_defensive_copies() -> None:
    report = _analyze(["No", "No", "Yes"])

    first = report.distribution_frame()
    first.loc[0, "Count"] = 999

    assert report.distribution_frame().loc[0, "Count"] == 2
    assert list(report.summary_frame().columns) == [
        "Metric",
        "Value",
        "Interpretation",
    ]


def test_analysis_does_not_modify_the_source_dataframe() -> None:
    dataframe = pd.DataFrame(
        {"Churn": ["No", "Yes", "No"]},
        index=[10, 20, 30],
    )
    before = dataframe.copy(deep=True)

    analyze_target_distribution(
        dataframe,
        target="Churn",
        expected_classes=("No", "Yes"),
        positive_class="Yes",
    )

    pd.testing.assert_frame_equal(dataframe, before)
