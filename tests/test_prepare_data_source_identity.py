"""Tests for the portable preparation-manifest source-identity object."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_data import (
    ConditionalNumericRule,
    build_preparation_manifest,
    build_source_identity,
    prepare_tabular_dataset,
    validate_raw_dataset,
)


COLUMN_ORDER = ("customerID", "tenure", "TotalCharges", "Churn")
IDENTIFIER_COLUMNS = ("customerID",)
FEATURE_COLUMNS = ("tenure", "TotalCharges")
TARGET_COLUMN = "Churn"
TARGET_CLASSES = ("No", "Yes")
RULE = ConditionalNumericRule(
    column="TotalCharges",
    condition_column="tenure",
    condition_value=0,
    blank_replacement=0.0,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D"],
            "tenure": [0, 1, 2, 3],
            "TotalCharges": ["   ", "10.5", "20.5", "30.5"],
            "Churn": ["No", "No", "Yes", "Yes"],
        }
    )


def _raw_report(frame: pd.DataFrame):
    return validate_raw_dataset(
        frame,
        column_order=COLUMN_ORDER,
        identifier_columns=IDENTIFIER_COLUMNS,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        target_classes=TARGET_CLASSES,
        categorical_expected_values={},
        expected_types={
            "customerID": "string",
            "tenure": "integer",
            "TotalCharges": "numeric",
            "Churn": "string",
        },
        numeric_text_columns=("TotalCharges",),
    )


def test_source_identity_contains_all_seven_minimum_fields() -> None:
    frame = _frame()
    identity = build_source_identity(
        dataset_slug="telco-customer-churn",
        source_path="data/raw/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        source_sha256="1" * 64,
        prepared_sha256="2" * 64,
        raw_report=_raw_report(frame),
        source_producer_revision=None,
        source_producer_revision_unavailable_reason=(
            "Kaggle dataset versioning not exposed by the download API in use."
        ),
    )

    assert identity["dataset_logical_identity"] == "telco-customer-churn"
    assert identity["source_filename_or_logical_source"] == (
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    assert identity["row_column_identity"]["row_count"] == 4
    assert identity["row_column_identity"]["column_count"] == 4
    assert identity["row_column_identity"]["column_order"] == list(COLUMN_ORDER)
    assert identity["content_hash"]["source_sha256"] == "1" * 64
    assert identity["content_hash"]["prepared_sha256"] == "2" * 64
    assert identity["source_revision_or_producer_revision"] is None
    assert identity["source_revision_unavailable_reason"]
    assert identity["artifact_type_version"]
    assert identity["artifact_sha256"] == "1" * 64


def test_source_identity_does_not_contain_absolute_or_directory_path() -> None:
    frame = _frame()
    identity = build_source_identity(
        dataset_slug="telco-customer-churn",
        source_path="data/raw/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        source_sha256="1" * 64,
        prepared_sha256="2" * 64,
        raw_report=_raw_report(frame),
        source_producer_revision_unavailable_reason="Not exposed by the source API.",
    )

    rendered_filename = identity["source_filename_or_logical_source"]
    assert not Path(rendered_filename).is_absolute()
    assert "/" not in rendered_filename
    assert "\\" not in rendered_filename


def test_source_identity_requires_a_reason_when_revision_is_absent() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="unavailable_reason"):
        build_source_identity(
            dataset_slug="telco-customer-churn",
            source_path="data/raw/telco-customer-churn/source.csv",
            source_sha256="1" * 64,
            prepared_sha256="2" * 64,
            raw_report=_raw_report(frame),
        )


def test_source_identity_preserves_a_real_revision_when_supplied() -> None:
    frame = _frame()
    identity = build_source_identity(
        dataset_slug="telco-customer-churn",
        source_path="data/raw/telco-customer-churn/source.csv",
        source_sha256="1" * 64,
        prepared_sha256="2" * 64,
        raw_report=_raw_report(frame),
        source_producer_revision="v3",
    )

    assert identity["source_revision_or_producer_revision"] == "v3"
    assert identity["source_revision_unavailable_reason"] is None


def test_preparation_manifest_embeds_source_identity_when_supplied() -> None:
    frame = _frame()
    raw_validation = _raw_report(frame)
    result = prepare_tabular_dataset(frame, conditional_numeric_rules=(RULE,))
    prepared_validation = _raw_report(result.dataframe)
    identity = build_source_identity(
        dataset_slug="telco-customer-churn",
        source_path="data/raw/telco-customer-churn/source.csv",
        source_sha256="1" * 64,
        prepared_sha256="2" * 64,
        raw_report=raw_validation,
        source_producer_revision_unavailable_reason="Not exposed by the source API.",
    )

    manifest = build_preparation_manifest(
        dataset_slug="telco-customer-churn",
        source_path="data/raw/telco-customer-churn/source.csv",
        source_sha256="1" * 64,
        prepared_path="data/processed/telco-customer-churn/prepared.csv",
        prepared_sha256="2" * 64,
        raw_report=raw_validation,
        prepared_report=prepared_validation,
        preparation=result,
        raw_fingerprint_before="fp",
        raw_fingerprint_after="fp",
        source_sha256_after="1" * 64,
        deterministic_rules=[RULE.as_dict()],
        readiness={},
        source_identity=identity,
    )

    assert manifest["source_identity"]["dataset_logical_identity"] == (
        "telco-customer-churn"
    )
    assert manifest["source_identity"] is not identity


def test_preparation_manifest_omits_source_identity_when_not_supplied() -> None:
    frame = _frame()
    raw_validation = _raw_report(frame)
    result = prepare_tabular_dataset(frame, conditional_numeric_rules=(RULE,))
    prepared_validation = _raw_report(result.dataframe)

    manifest = build_preparation_manifest(
        dataset_slug="telco-customer-churn",
        source_path="data/raw/telco-customer-churn/source.csv",
        source_sha256="1" * 64,
        prepared_path="data/processed/telco-customer-churn/prepared.csv",
        prepared_sha256="2" * 64,
        raw_report=raw_validation,
        prepared_report=prepared_validation,
        preparation=result,
        raw_fingerprint_before="fp",
        raw_fingerprint_after="fp",
        source_sha256_after="1" * 64,
        deterministic_rules=[RULE.as_dict()],
        readiness={},
    )

    assert "source_identity" not in manifest
