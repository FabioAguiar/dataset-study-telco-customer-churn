"""Tests for reusable feature-relationship exploration."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_relationships import (
    FeatureRelationshipAnalysisError,
    analyze_feature_relationships,
)


def _report(
    numerical: pd.DataFrame,
    categorical: pd.DataFrame | None = None,
    *,
    numerical_features: tuple[str, ...] | None = None,
    categorical_features: tuple[str, ...] | None = None,
    interaction_candidates: tuple[dict[str, object], ...] = (),
):
    categorical = (
        pd.DataFrame(index=numerical.index)
        if categorical is None
        else categorical
    )
    return analyze_feature_relationships(
        numerical,
        categorical,
        numerical_features=(
            tuple(numerical.columns)
            if numerical_features is None
            else numerical_features
        ),
        categorical_features=(
            tuple(categorical.columns)
            if categorical_features is None
            else categorical_features
        ),
        interaction_candidates=interaction_candidates,
    )


def test_perfect_positive_pearson_and_rank_correlation() -> None:
    numerical = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})

    report = _report(numerical)
    row = report.numerical_relationships_frame().iloc[0]

    assert row["Pearson correlation"] == pytest.approx(1.0)
    assert row["Rank correlation"] == pytest.approx(1.0)
    assert row["Strong association"]
    assert row["Potential redundancy"]


def test_perfect_negative_correlation() -> None:
    numerical = pd.DataFrame({"a": [1, 2, 3], "b": [6, 4, 2]})

    row = _report(numerical).numerical_relationships_frame().iloc[0]

    assert row["Pearson correlation"] == pytest.approx(-1.0)
    assert row["Rank correlation"] == pytest.approx(-1.0)


def test_rank_correlation_detects_monotonic_nonlinear_relation() -> None:
    numerical = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [1, 4, 9, 16]}
    )

    row = _report(numerical).numerical_relationships_frame().iloc[0]

    assert row["Rank correlation"] == pytest.approx(1.0)
    assert row["Pearson correlation"] < 1.0


def test_missing_numerical_pairs_are_excluded() -> None:
    numerical = pd.DataFrame(
        {"a": [1.0, None, 3.0], "b": [1.0, 2.0, None]}
    )

    row = _report(numerical).numerical_relationships_frame().iloc[0]

    assert row["Valid paired rows"] == 1
    assert row["Missing paired rows"] == 2
    assert row["Pearson correlation"] is None


def test_constant_numerical_feature_is_reported() -> None:
    numerical = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})

    report = _report(numerical)

    assert report.has_constant_features
    assert "Constant numerical feature" in set(
        report.issues_frame()["Issue"]
    )
    with pytest.raises(
        FeatureRelationshipAnalysisError,
        match="constant_features_detected",
    ):
        report.raise_if_invalid(require_sufficient_variation=True)


def test_numerical_matrix_is_symmetric_with_unit_diagonal() -> None:
    numerical = pd.DataFrame(
        {"a": [1, 2, 3], "b": [2, 4, 6], "c": [3, 1, 2]}
    )

    matrix = _report(numerical).numerical_correlation_matrix()

    pd.testing.assert_frame_equal(matrix, matrix.T)
    assert list(matrix.columns) == ["a", "b", "c"]
    assert list(matrix.index) == ["a", "b", "c"]
    assert all(matrix.loc[name, name] == 1.0 for name in matrix.columns)


def test_rank_matrix_aliases_are_supported() -> None:
    numerical = pd.DataFrame({"a": [1, 2, 3], "b": [1, 4, 9]})
    report = _report(numerical)

    pd.testing.assert_frame_equal(
        report.numerical_correlation_matrix(method="rank"),
        report.numerical_correlation_matrix(method="spearman"),
    )

    with pytest.raises(FeatureRelationshipAnalysisError, match="method"):
        report.numerical_correlation_matrix(method="kendall")


def test_cramers_v_is_one_for_perfect_association() -> None:
    numerical = pd.DataFrame(index=range(4))
    categorical = pd.DataFrame(
        {"a": ["x", "x", "y", "y"], "b": ["m", "m", "n", "n"]}
    )

    row = _report(numerical, categorical).categorical_relationships_frame().iloc[0]

    assert row["Cramer's V"] == pytest.approx(1.0)
    assert row["U(A | B)"] == pytest.approx(1.0)
    assert row["U(B | A)"] == pytest.approx(1.0)
    assert row["Potential redundancy"]


def test_cramers_v_is_zero_for_balanced_independence() -> None:
    numerical = pd.DataFrame(index=range(4))
    categorical = pd.DataFrame(
        {"a": ["x", "x", "y", "y"], "b": ["m", "n", "m", "n"]}
    )

    row = _report(numerical, categorical).categorical_relationships_frame().iloc[0]

    assert row["Cramer's V"] == pytest.approx(0.0)
    assert row["U(A | B)"] == pytest.approx(0.0)
    assert row["U(B | A)"] == pytest.approx(0.0)


def test_directional_dependency_is_not_mistaken_for_redundancy() -> None:
    numerical = pd.DataFrame(index=range(6))
    categorical = pd.DataFrame(
        {
            "base": ["No", "No", "Yes", "Yes", "Yes", "Yes"],
            "detail": [
                "No service",
                "No service",
                "A",
                "B",
                "A",
                "B",
            ],
        }
    )

    row = _report(numerical, categorical).categorical_relationships_frame().iloc[0]

    assert row["U(A | B)"] == pytest.approx(1.0)
    assert row["U(B | A)"] < 1.0
    assert row["Structural dependency"]
    assert not row["Potential redundancy"]


def test_constant_categorical_feature_is_reported() -> None:
    numerical = pd.DataFrame(index=range(3))
    categorical = pd.DataFrame({"a": ["x"] * 3, "b": ["m", "n", "m"]})

    report = _report(numerical, categorical)

    assert report.has_constant_features
    assert report.categorical_relationships_frame().iloc[0]["Cramer's V"] is None


def test_blank_and_missing_categories_are_excluded() -> None:
    numerical = pd.DataFrame(index=range(5))
    categorical = pd.DataFrame(
        {"a": ["x", " ", None, "y", "y"], "b": ["m", "m", "n", "n", "n"]}
    )

    row = _report(numerical, categorical).categorical_relationships_frame().iloc[0]

    assert row["Valid paired rows"] == 3
    assert row["Missing paired rows"] == 2


def test_pandas_string_dtype_is_supported_without_mutation() -> None:
    numerical = pd.DataFrame(index=range(4))
    categorical = pd.DataFrame(
        {
            "a": pd.Series(["x", "x", "y", "y"], dtype="string"),
            "b": pd.Series(["m", "m", "n", "n"], dtype="string"),
        }
    )
    original = categorical.copy(deep=True)

    report = _report(numerical, categorical)

    assert report.categorical_relationships_frame().iloc[0]["Cramer's V"] == pytest.approx(1.0)
    pd.testing.assert_frame_equal(categorical, original)


def test_categorical_matrix_is_symmetric_with_unit_diagonal() -> None:
    numerical = pd.DataFrame(index=range(4))
    categorical = pd.DataFrame(
        {
            "a": ["x", "x", "y", "y"],
            "b": ["m", "m", "n", "n"],
            "c": ["q", "r", "q", "r"],
        }
    )

    matrix = _report(numerical, categorical).categorical_association_matrix()

    pd.testing.assert_frame_equal(matrix, matrix.T)
    assert all(matrix.loc[name, name] == 1.0 for name in matrix.columns)


def test_eta_squared_is_one_for_separated_groups() -> None:
    numerical = pd.DataFrame({"value": [0.0, 0.0, 10.0, 10.0]})
    categorical = pd.DataFrame({"group": ["a", "a", "b", "b"]})

    row = _report(numerical, categorical).mixed_relationships_frame().iloc[0]

    assert row["Eta squared"] == pytest.approx(1.0)
    assert row["Strong association"]


def test_eta_squared_is_zero_when_group_means_are_equal() -> None:
    numerical = pd.DataFrame({"value": [0.0, 10.0, 0.0, 10.0]})
    categorical = pd.DataFrame({"group": ["a", "a", "b", "b"]})

    row = _report(numerical, categorical).mixed_relationships_frame().iloc[0]

    assert row["Eta squared"] == pytest.approx(0.0)


def test_single_category_mixed_relation_is_undefined() -> None:
    numerical = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    categorical = pd.DataFrame({"group": ["a", "a", "a"]})

    report = _report(numerical, categorical)
    row = report.mixed_relationships_frame().iloc[0]

    assert row["Eta squared"] is None
    assert report.has_constant_features


def test_mixed_matrix_preserves_declared_order() -> None:
    numerical = pd.DataFrame({"n2": [1, 2, 3, 4], "n1": [4, 3, 2, 1]})
    categorical = pd.DataFrame({"c2": ["a", "a", "b", "b"], "c1": ["x", "y", "x", "y"]})

    report = analyze_feature_relationships(
        numerical,
        categorical,
        numerical_features=("n2", "n1"),
        categorical_features=("c2", "c1"),
    )
    matrix = report.mixed_association_matrix()

    assert list(matrix.index) == ["c2", "c1"]
    assert list(matrix.columns) == ["n2", "n1"]


def test_product_interaction_is_analyzed() -> None:
    numerical = pd.DataFrame(
        {
            "tenure": [1.0, 2.0, 3.0],
            "monthly": [10.0, 20.0, 30.0],
            "total": [10.0, 40.0, 90.0],
        }
    )
    candidate = (
        {
            "name": "tenure_monthly_product",
            "left": "tenure",
            "right": "monthly",
            "operation": "product",
            "compare_to": "total",
        },
    )

    row = _report(
        numerical,
        interaction_candidates=candidate,
    ).interactions_frame().iloc[0]

    assert row["Pearson correlation"] == pytest.approx(1.0)
    assert row["Mean absolute difference"] == pytest.approx(0.0)
    assert row["Median absolute difference"] == pytest.approx(0.0)
    assert row["Mean relative difference"] == pytest.approx(0.0)
    assert row["Potential redundancy"]


def test_interaction_ignores_zero_denominator_for_relative_difference() -> None:
    numerical = pd.DataFrame(
        {"a": [0.0, 2.0], "b": [1.0, 2.0], "c": [0.0, 5.0]}
    )
    candidate = (
        {
            "name": "ab",
            "left": "a",
            "right": "b",
            "operation": "product",
            "compare_to": "c",
        },
    )

    row = _report(numerical, interaction_candidates=candidate).interactions_frame().iloc[0]

    assert row["Mean relative difference"] == pytest.approx(0.2)


def test_missing_interaction_feature_is_reported() -> None:
    numerical = pd.DataFrame({"a": [1, 2], "b": [2, 3]})
    candidate = (
        {
            "name": "missing",
            "left": "a",
            "right": "b",
            "operation": "product",
            "compare_to": "c",
        },
    )

    report = _report(numerical, interaction_candidates=candidate)

    assert report.interactions_frame().empty
    assert "Interaction feature missing" in set(report.issues_frame()["Issue"])


def test_indices_must_align_for_mixed_relationships() -> None:
    numerical = pd.DataFrame({"value": [1, 2]}, index=[0, 1])
    categorical = pd.DataFrame({"group": ["a", "b"]}, index=[1, 2])

    report = _report(numerical, categorical)

    assert report.has_alignment_issues
    assert report.mixed_relationships_frame().empty
    with pytest.raises(
        FeatureRelationshipAnalysisError,
        match="projection_indices_not_aligned",
    ):
        report.raise_if_invalid()


def test_missing_features_are_reported() -> None:
    numerical = pd.DataFrame({"a": [1, 2]})
    categorical = pd.DataFrame({"c": ["x", "y"]})

    report = analyze_feature_relationships(
        numerical,
        categorical,
        numerical_features=("a", "missing_n"),
        categorical_features=("c", "missing_c"),
    )

    assert report.has_missing_features
    assert report.missing_numerical_features == ("missing_n",)
    assert report.missing_categorical_features == ("missing_c",)
    with pytest.raises(FeatureRelationshipAnalysisError, match="missing_numerical_features"):
        report.raise_if_invalid()


def test_duplicate_feature_names_are_rejected() -> None:
    frame = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(FeatureRelationshipAnalysisError, match="duplicate"):
        analyze_feature_relationships(
            frame,
            pd.DataFrame(index=frame.index),
            numerical_features=("a", "a"),
            categorical_features=(),
        )


def test_duplicated_dataframe_columns_are_rejected() -> None:
    numerical = pd.DataFrame([[1, 2]], columns=["a", "a"])

    with pytest.raises(FeatureRelationshipAnalysisError, match="duplicated column labels"):
        analyze_feature_relationships(
            numerical,
            pd.DataFrame(index=numerical.index),
            numerical_features=("a",),
            categorical_features=(),
        )


def test_invalid_threshold_is_rejected() -> None:
    frame = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(FeatureRelationshipAnalysisError, match="between 0 and 1"):
        analyze_feature_relationships(
            frame,
            pd.DataFrame(index=frame.index),
            numerical_features=("a",),
            categorical_features=(),
            strong_numerical_threshold=1.1,
        )


def test_invalid_interaction_operation_is_rejected() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]})

    with pytest.raises(FeatureRelationshipAnalysisError, match="Unsupported interaction operation"):
        _report(
            frame,
            interaction_candidates=(
                {
                    "name": "invalid",
                    "left": "a",
                    "right": "b",
                    "operation": "sum",
                    "compare_to": "c",
                },
            ),
        )


def test_duplicate_interaction_names_are_rejected() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]})
    candidate = {
        "name": "same",
        "left": "a",
        "right": "b",
        "operation": "product",
        "compare_to": "c",
    }

    with pytest.raises(FeatureRelationshipAnalysisError, match="names must be unique"):
        _report(
            frame,
            interaction_candidates=(candidate, candidate),
        )


def test_returned_frames_are_defensive_copies() -> None:
    numerical = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    report = _report(numerical)

    relationships = report.numerical_relationships_frame()
    matrix = report.numerical_correlation_matrix()
    relationships.loc[0, "Pearson correlation"] = 0.0
    matrix.loc["a", "b"] = 0.0

    assert report.numerical_relationships_frame().loc[0, "Pearson correlation"] == pytest.approx(1.0)
    assert report.numerical_correlation_matrix().loc["a", "b"] == pytest.approx(1.0)


def test_input_frames_are_not_mutated() -> None:
    numerical = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    categorical = pd.DataFrame({"c": ["x", "y", "x"], "d": ["m", "n", "m"]})
    numerical_original = numerical.copy(deep=True)
    categorical_original = categorical.copy(deep=True)

    _report(numerical, categorical)

    pd.testing.assert_frame_equal(numerical, numerical_original)
    pd.testing.assert_frame_equal(categorical, categorical_original)


def test_summary_reports_expected_pair_counts() -> None:
    numerical = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5]})
    categorical = pd.DataFrame({"x": ["a", "b", "a"], "y": ["m", "n", "m"]})

    summary = _report(numerical, categorical).summary_frame().set_index("Metric")

    assert summary.loc["Numerical relationships", "Value"] == 3
    assert summary.loc["Categorical relationships", "Value"] == 1
    assert summary.loc["Categorical-numerical relationships", "Value"] == 6
