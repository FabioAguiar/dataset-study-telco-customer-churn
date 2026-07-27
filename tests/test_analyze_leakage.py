"""Tests for reusable potential-data-leakage auditing."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from scripts.analyze_leakage import (
    DataLeakageAnalysisError,
    analyze_data_leakage,
)


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D"],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "category": ["x", "y", "x", "y"],
            "Churn": ["No", "No", "Yes", "Yes"],
        }
    )


def _field_declarations() -> dict[str, dict[str, object]]:
    return {
        "customerID": {
            "role": "identifier",
            "uses_target": False,
            "available_at_inference": True,
            "available_before_target": True,
            "model_disposition": "Exclude",
            "reason": "Unique account identifier.",
        },
        "feature": {
            "role": "candidate feature",
            "uses_target": False,
            "available_at_inference": True,
            "available_before_target": True,
            "model_disposition": "Safe",
            "reason": "Observed before prediction.",
        },
        "category": {
            "role": "candidate feature",
            "uses_target": False,
            "available_at_inference": True,
            "available_before_target": True,
            "model_disposition": "Safe",
            "reason": "Observed before prediction.",
        },
        "Churn": {
            "role": "target",
            "uses_target": True,
            "available_at_inference": False,
            "available_before_target": False,
            "model_disposition": "Prohibited",
            "reason": "Direct prediction target.",
        },
    }


def _context(*, complete: bool = True) -> dict[str, object]:
    return {
        "prediction_unit": "customer account",
        "scoring_population": "active accounts" if complete else None,
        "observation_time_definition": "snapshot date" if complete else None,
        "target_event_definition": "future churn" if complete else None,
        "prediction_horizon": "next 30 days" if complete else None,
        "feature_snapshot_cutoff": "snapshot date" if complete else None,
        "source_snapshot_precedes_target": True if complete else None,
        "status": "Resolved" if complete else "Unresolved",
    }


def _pipeline_rules(*, compliant: bool = True) -> dict[str, bool]:
    return {
        "split_before_learning_transformations": compliant,
        "fit_imputation_on_training_only": compliant,
        "fit_scaling_on_training_only": compliant,
        "fit_encoding_on_training_only": compliant,
        "fit_rare_category_rules_on_training_only": compliant,
        "fit_outlier_rules_on_training_only": compliant,
        "fit_feature_selection_on_training_only": compliant,
        "apply_resampling_to_training_only": compliant,
        "keep_test_set_untouched_until_final_evaluation": compliant,
    }


def _analyze(
    dataframe: pd.DataFrame | None = None,
    *,
    field_declarations: dict[str, dict[str, object]] | None = None,
    transformations: dict[str, dict[str, object]] | None = None,
    context: dict[str, object] | None = None,
    pipeline_rules: dict[str, bool] | None = None,
    identifiers: tuple[str, ...] = ("customerID",),
    candidate_features: tuple[str, ...] = ("feature", "category"),
):
    return analyze_data_leakage(
        _dataframe() if dataframe is None else dataframe,
        target="Churn",
        identifiers=identifiers,
        candidate_features=candidate_features,
        field_declarations=(
            _field_declarations()
            if field_declarations is None
            else field_declarations
        ),
        transformation_declarations=(
            {}
            if transformations is None
            else transformations
        ),
        inference_context=(
            _context()
            if context is None
            else context
        ),
        pipeline_rules=(
            _pipeline_rules()
            if pipeline_rules is None
            else pipeline_rules
        ),
    )


def test_valid_audit_produces_all_report_tables() -> None:
    report = _analyze()

    assert report.is_structurally_valid
    assert report.is_modeling_ready
    assert len(report.field_audit_frame()) == 4
    assert len(report.context_frame()) == 7
    assert len(report.pipeline_policy_frame()) == 9
    assert report.transformation_audit_frame().empty
    assert report.target_proxy_candidates_frame().empty
    assert not report.risk_register_frame().empty


def test_target_is_prohibited_but_not_counted_as_candidate_leakage() -> None:
    report = _analyze()
    target_row = report.field_audit_frame().set_index("Field").loc["Churn"]

    assert bool(target_row["Direct target leakage"])
    assert target_row["Model disposition"] == "Prohibited"
    assert not report.has_direct_target_leakage


def test_identifier_is_reported_as_memorization_risk() -> None:
    report = _analyze()
    row = report.field_audit_frame().set_index("Field").loc["customerID"]

    assert bool(row["Identifier risk"])
    assert row["Risk level"] == "High"
    assert report.has_identifier_risks


def test_missing_target_is_structural_error() -> None:
    dataframe = _dataframe().drop(columns="Churn")
    report = _analyze(dataframe)

    assert not report.is_structurally_valid
    with pytest.raises(DataLeakageAnalysisError, match="target field is missing"):
        report.raise_if_invalid()


def test_missing_identifier_is_structural_error() -> None:
    dataframe = _dataframe().drop(columns="customerID")
    report = _analyze(dataframe)

    assert report.missing_identifiers == ("customerID",)
    with pytest.raises(DataLeakageAnalysisError, match="identifier fields"):
        report.raise_if_invalid()


def test_missing_candidate_feature_is_structural_error() -> None:
    dataframe = _dataframe().drop(columns="feature")
    report = _analyze(dataframe)

    assert report.missing_candidate_features == ("feature",)
    with pytest.raises(DataLeakageAnalysisError, match="candidate features"):
        report.raise_if_invalid()


def test_undeclared_field_is_reported() -> None:
    declarations = _field_declarations()
    declarations.pop("category")
    report = _analyze(field_declarations=declarations)

    assert report.undeclared_fields == ("category",)
    with pytest.raises(DataLeakageAnalysisError, match="no leakage declaration"):
        report.raise_if_invalid()


def test_duplicate_candidate_declaration_is_reported() -> None:
    report = _analyze(candidate_features=("feature", "feature", "category"))

    assert "Duplicate candidate declaration" in set(
        report.issues_frame()["Issue"]
    )
    with pytest.raises(DataLeakageAnalysisError, match="not unique"):
        report.raise_if_invalid()


def test_identifier_candidate_overlap_is_reported() -> None:
    report = _analyze(candidate_features=("customerID", "feature", "category"))

    assert "Overlapping field roles" in set(report.issues_frame()["Issue"])


def test_invalid_disposition_is_reported() -> None:
    declarations = _field_declarations()
    declarations["feature"]["model_disposition"] = "Maybe"
    report = _analyze(field_declarations=declarations)

    with pytest.raises(DataLeakageAnalysisError, match="dispositions are invalid"):
        report.raise_if_invalid()


def test_unknown_transformation_source_is_reported() -> None:
    report = _analyze(
        transformations={
            "derived": {
                "sources": ("missing",),
                "uses_target": False,
                "learned_from_data": False,
                "fit_scope": "none",
                "inference_reproducible": True,
                "aggregation_cutoff_respected": True,
                "model_disposition": "Safe deterministic",
                "reason": "Synthetic test.",
            }
        }
    )

    with pytest.raises(DataLeakageAnalysisError, match="sources are invalid"):
        report.raise_if_invalid()


def test_exact_target_copy_is_detected() -> None:
    dataframe = _dataframe().assign(proxy=lambda frame: frame["Churn"])
    declarations = _field_declarations()
    declarations["proxy"] = {
        "role": "candidate feature",
        "uses_target": False,
        "available_at_inference": True,
        "available_before_target": True,
        "model_disposition": "Conditionally safe",
        "reason": "Unknown provenance.",
    }
    report = _analyze(
        dataframe,
        field_declarations=declarations,
        candidate_features=("feature", "category", "proxy"),
    )

    proxy = report.target_proxy_candidates_frame().iloc[0]
    assert proxy["Field"] == "proxy"
    assert proxy["Detection method"] == "Exact target copy"
    assert proxy["Exact match rate"] == pytest.approx(1.0)
    assert report.has_target_proxy_candidates
    assert not report.is_modeling_ready


def test_normalized_text_target_copy_is_detected() -> None:
    dataframe = _dataframe().assign(proxy=[" no ", "NO", " yes", "YES "])
    declarations = _field_declarations()
    declarations["proxy"] = {
        "role": "candidate feature",
        "uses_target": False,
        "available_at_inference": True,
        "available_before_target": True,
        "model_disposition": "Conditionally safe",
        "reason": "Unknown provenance.",
    }
    report = _analyze(
        dataframe,
        field_declarations=declarations,
        candidate_features=("feature", "category", "proxy"),
    )

    row = report.target_proxy_candidates_frame().iloc[0]
    assert row["Detection method"] == "Normalized target copy"
    assert row["Normalized match rate"] == pytest.approx(1.0)


def test_binary_target_encoding_is_detected() -> None:
    dataframe = _dataframe().assign(proxy=[0, 0, 1, 1])
    declarations = _field_declarations()
    declarations["proxy"] = {
        "role": "candidate feature",
        "uses_target": False,
        "available_at_inference": True,
        "available_before_target": True,
        "model_disposition": "Conditionally safe",
        "reason": "Unknown provenance.",
    }
    report = _analyze(
        dataframe,
        field_declarations=declarations,
        candidate_features=("feature", "category", "proxy"),
    )

    row = report.target_proxy_candidates_frame().iloc[0]
    assert row["Detection method"] == "Binary target encoding"
    assert bool(row["Bidirectional mapping"])


def test_deterministic_multicategory_target_mapping_is_detected() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": list("ABCDEFGH"),
            "feature": range(8),
            "category": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "Churn": ["No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes"],
        }
    )
    report = _analyze(dataframe)

    candidates = report.target_proxy_candidates_frame()
    assert "category" in set(candidates["Field"])
    assert "Deterministic target mapping" in set(candidates["Detection method"])


def test_unique_identifier_is_not_misclassified_as_deterministic_proxy() -> None:
    report = _analyze()

    assert "customerID" not in set(
        report.target_proxy_candidates_frame().get("Field", pd.Series(dtype=str))
    )


def test_strong_but_non_deterministic_association_is_not_called_leakage() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(8)],
            "feature": range(8),
            "category": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "Churn": ["No", "No", "No", "Yes", "Yes", "Yes", "Yes", "No"],
        }
    )
    report = _analyze(dataframe)

    assert report.target_proxy_candidates_frame().empty


def test_unresolved_inference_context_blocks_modeling() -> None:
    report = _analyze(context=_context(complete=False))

    assert report.has_unresolved_temporal_contract
    assert not report.is_modeling_ready
    with pytest.raises(DataLeakageAnalysisError, match="temporal contract"):
        report.raise_if_modeling_unsafe()


def test_candidate_unavailable_at_inference_blocks_modeling() -> None:
    declarations = _field_declarations()
    declarations["feature"]["available_at_inference"] = False
    report = _analyze(field_declarations=declarations)

    assert report.has_inference_availability_issues
    assert not report.is_modeling_ready


def test_candidate_with_unknown_temporal_order_is_flagged() -> None:
    declarations = _field_declarations()
    declarations["feature"]["available_before_target"] = None
    report = _analyze(field_declarations=declarations)

    row = report.field_audit_frame().set_index("Field").loc["feature"]
    assert bool(row["Temporal risk"])
    assert row["Risk level"] == "Medium"


def test_explicit_post_event_field_is_high_risk() -> None:
    declarations = _field_declarations()
    declarations["feature"]["available_before_target"] = False
    report = _analyze(field_declarations=declarations)

    row = report.field_audit_frame().set_index("Field").loc["feature"]
    assert row["Risk level"] == "High"
    assert report.has_temporal_risks


def test_safe_deterministic_transformation_is_audited() -> None:
    report = _analyze(
        transformations={
            "feature_squared": {
                "sources": ("feature",),
                "uses_target": False,
                "learned_from_data": False,
                "fit_scope": "none",
                "inference_reproducible": True,
                "aggregation_cutoff_respected": True,
                "model_disposition": "Safe deterministic",
                "reason": "Deterministic arithmetic.",
            }
        }
    )

    row = report.transformation_audit_frame().iloc[0]
    assert not bool(row["Target-derived"])
    assert not bool(row["Global-fit risk"])
    assert row["Risk level"] == "Low"
    assert report.is_modeling_ready


def test_target_derived_exploratory_transformation_is_controlled_but_not_input() -> None:
    report = _analyze(
        transformations={
            "category_churn_rate": {
                "sources": ("category", "Churn"),
                "uses_target": True,
                "learned_from_data": True,
                "fit_scope": "exploratory",
                "inference_reproducible": False,
                "aggregation_cutoff_respected": True,
                "model_disposition": "Exploratory only",
                "reason": "Documentation metric.",
            }
        }
    )

    row = report.transformation_audit_frame().iloc[0]
    assert bool(row["Target-derived"])
    assert row["Risk level"] == "Critical"
    assert report.has_target_derived_transformations


def test_global_fit_transformation_blocks_modeling() -> None:
    report = _analyze(
        transformations={
            "global_scaler": {
                "sources": ("feature",),
                "uses_target": False,
                "learned_from_data": True,
                "fit_scope": "global",
                "inference_reproducible": True,
                "aggregation_cutoff_respected": True,
                "model_disposition": "Train-only fit",
                "reason": "Must be fitted on training data.",
            }
        }
    )

    assert report.has_global_fit_risks
    assert not report.is_modeling_ready


def test_train_only_transformation_does_not_create_global_fit_risk() -> None:
    report = _analyze(
        transformations={
            "training_scaler": {
                "sources": ("feature",),
                "uses_target": False,
                "learned_from_data": True,
                "fit_scope": "training_only",
                "inference_reproducible": True,
                "aggregation_cutoff_respected": True,
                "model_disposition": "Train-only fit",
                "reason": "Fitted only on training data.",
            }
        }
    )

    assert not report.has_global_fit_risks
    assert report.is_modeling_ready


def test_unresolved_aggregate_cutoff_is_temporal_risk() -> None:
    report = _analyze(
        transformations={
            "cumulative_total": {
                "sources": ("feature",),
                "uses_target": False,
                "learned_from_data": False,
                "fit_scope": "none",
                "inference_reproducible": True,
                "aggregation_cutoff_respected": None,
                "model_disposition": "Conditionally safe",
                "reason": "Cutoff requires confirmation.",
            }
        }
    )

    row = report.transformation_audit_frame().iloc[0]
    assert bool(row["Temporal risk"])
    assert report.has_temporal_risks


def test_noncompliant_pipeline_rule_blocks_modeling() -> None:
    rules = _pipeline_rules()
    rules["fit_encoding_on_training_only"] = False
    report = _analyze(pipeline_rules=rules)

    assert not report.is_modeling_ready
    with pytest.raises(DataLeakageAnalysisError, match="train-only"):
        report.raise_if_modeling_unsafe()


def test_report_frames_are_defensive_copies() -> None:
    report = _analyze()
    field_frame = report.field_audit_frame()
    field_frame.loc[0, "Field"] = "changed"

    assert report.field_audit_frame().loc[0, "Field"] != "changed"


def test_analysis_does_not_mutate_inputs() -> None:
    dataframe = _dataframe()
    declarations = _field_declarations()
    transformations = {
        "feature_squared": {
            "sources": ("feature",),
            "uses_target": False,
            "learned_from_data": False,
            "fit_scope": "none",
            "inference_reproducible": True,
            "aggregation_cutoff_respected": True,
            "model_disposition": "Safe deterministic",
            "reason": "Synthetic test.",
        }
    }
    context = _context()
    rules = _pipeline_rules()

    dataframe_before = dataframe.copy(deep=True)
    declarations_before = copy.deepcopy(declarations)
    transformations_before = copy.deepcopy(transformations)
    context_before = copy.deepcopy(context)
    rules_before = copy.deepcopy(rules)

    analyze_data_leakage(
        dataframe,
        target="Churn",
        identifiers=("customerID",),
        candidate_features=("feature", "category"),
        field_declarations=declarations,
        transformation_declarations=transformations,
        inference_context=context,
        pipeline_rules=rules,
    )

    pd.testing.assert_frame_equal(dataframe, dataframe_before)
    assert declarations == declarations_before
    assert transformations == transformations_before
    assert context == context_before
    assert rules == rules_before


def test_pandas_string_dtype_is_supported() -> None:
    dataframe = _dataframe()
    dataframe["category"] = pd.Series(
        dataframe["category"],
        dtype="string",
    )
    dataframe["Churn"] = pd.Series(
        dataframe["Churn"],
        dtype="string",
    )

    report = _analyze(dataframe)

    assert report.is_structurally_valid


def test_results_are_deterministic() -> None:
    first = _analyze()
    second = _analyze()

    pd.testing.assert_frame_equal(
        first.field_audit_frame(),
        second.field_audit_frame(),
    )
    pd.testing.assert_frame_equal(
        first.risk_register_frame(),
        second.risk_register_frame(),
    )
