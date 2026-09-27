"""Tests for the holdout established before target-aware exploration."""

from __future__ import annotations

import copy

import pytest

from scripts.prepare_data import (
    PartitionValidationError,
    build_holdout_membership,
    build_split_manifest,
    split_classification_dataset,
    validate_dataset_partitions,
    verify_partitions_match_holdout,
)
from tests.test_prepare_data import (
    IDENTIFIER_COLUMNS,
    TARGET_CLASSES,
    make_telco_frame,
    prepared_result,
    split_policy,
)

SOURCE_SHA = "a" * 64


def _raw_partitions(frame=None, **policy_changes):
    frame = make_telco_frame(120) if frame is None else frame
    policy = split_policy(**policy_changes)
    partitions = split_classification_dataset(
        frame,
        policy=policy,
        identifier_columns=IDENTIFIER_COLUMNS,
        target_classes=TARGET_CLASSES,
    )
    return frame, policy, partitions


def _holdout(policy, partitions):
    return build_holdout_membership(
        dataset_slug="telco-customer-churn",
        policy=policy,
        partitions=partitions,
        identifier_columns=IDENTIFIER_COLUMNS,
        target_column="Churn",
        source_sha256=SOURCE_SHA,
        established_by="notebooks/01#8A",
        exploration_partition="train",
        established_before=["target_distribution"],
    )


def test_prepared_split_reproduces_the_holdout_established_on_raw_rows() -> None:
    raw, policy, raw_partitions = _raw_partitions()
    holdout = _holdout(policy, raw_partitions)

    prepared = prepared_result(raw).dataframe
    prepared_partitions = split_classification_dataset(
        prepared,
        policy=policy,
        identifier_columns=IDENTIFIER_COLUMNS,
        target_classes=TARGET_CLASSES,
    )

    digests = verify_partitions_match_holdout(
        prepared_partitions,
        holdout,
        identifier_columns=IDENTIFIER_COLUMNS,
        policy=policy,
        source_sha256=SOURCE_SHA,
    )
    assert set(digests) == {"train", "validation", "test"}
    assert holdout["partitions"]["test"]["row_count"] == len(raw_partitions.test)


def test_a_different_split_is_rejected() -> None:
    _, policy, partitions = _raw_partitions()
    holdout = _holdout(policy, partitions)
    _, _, other = _raw_partitions(random_seed=7)

    with pytest.raises(PartitionValidationError, match="does not reproduce"):
        verify_partitions_match_holdout(
            other, holdout, identifier_columns=IDENTIFIER_COLUMNS
        )


def test_a_tampered_identifier_list_is_rejected() -> None:
    _, policy, partitions = _raw_partitions()
    holdout = copy.deepcopy(_holdout(policy, partitions))
    moved = holdout["partitions"]["test"]["identifiers"].pop()
    holdout["partitions"]["train"]["identifiers"].append(moved)

    with pytest.raises(PartitionValidationError):
        verify_partitions_match_holdout(
            partitions, holdout, identifier_columns=IDENTIFIER_COLUMNS
        )


def test_policy_and_source_drift_are_rejected() -> None:
    _, policy, partitions = _raw_partitions()
    holdout = _holdout(policy, partitions)

    with pytest.raises(PartitionValidationError, match="split policy"):
        verify_partitions_match_holdout(
            partitions,
            holdout,
            identifier_columns=IDENTIFIER_COLUMNS,
            policy=split_policy(random_seed=7),
        )
    with pytest.raises(PartitionValidationError, match="source file"):
        verify_partitions_match_holdout(
            partitions,
            holdout,
            identifier_columns=IDENTIFIER_COLUMNS,
            source_sha256="b" * 64,
        )


def test_test_partition_cannot_be_the_exploration_partition() -> None:
    _, policy, partitions = _raw_partitions()
    with pytest.raises(ValueError, match="test partition"):
        build_holdout_membership(
            dataset_slug="telco-customer-churn",
            policy=policy,
            partitions=partitions,
            identifier_columns=IDENTIFIER_COLUMNS,
            target_column="Churn",
            source_sha256=SOURCE_SHA,
            established_by="x",
            exploration_partition="test",
            established_before=[],
        )


def test_split_manifest_records_the_holdout_reference() -> None:
    raw, policy, partitions = _raw_partitions()
    validation = validate_dataset_partitions(
        raw,
        partitions,
        identifier_columns=IDENTIFIER_COLUMNS,
        target_column="Churn",
        target_classes=TARGET_CLASSES,
        prevalence_tolerance=0.05,
    )
    reference = {"path": "artifacts/x.json", "sha256": "c" * 64}
    manifest = build_split_manifest(
        dataset_slug="telco-customer-churn",
        policy=policy,
        partitions=partitions,
        validation=validation,
        partition_paths={name: f"data/{name}.csv" for name in ("train", "validation", "test")},
        partition_sha256={name: "d" * 64 for name in ("train", "validation", "test")},
        holdout_reference=reference,
    )
    assert manifest["holdout_establishment"] == reference
