"""Deterministic and reusable tabular data-preparation utilities.

The module contains dataset-agnostic validation, conditional numeric
materialization, classification splitting, manifest construction, atomic
artifact persistence, and preparation-handoff validation. Dataset-specific
contracts are supplied by callers.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import platform
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import pandas as pd
from pandas.api import types as pandas_types
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split


CONTRACT_VERSION = "tabular-preparation.v1"
CSV_ENCODING = "utf-8"
CSV_LINE_TERMINATOR = "\n"
_VOLATILE_SEMANTIC_KEYS = frozenset(
    {
        "created_at",
        "created_at_utc",
        "generated_at",
        "generated_at_utc",
        "timestamp",
        "updated_at",
        "updated_at_utc",
    }
)


class PreparationError(RuntimeError):
    """Base exception for preparation failures."""


class DatasetValidationError(PreparationError):
    """Raised when a raw or prepared dataset violates its contract."""


class ConditionalMaterializationError(PreparationError):
    """Raised when a conditional numeric rule cannot be applied safely."""


class SplitPolicyError(PreparationError):
    """Raised when a classification split policy is invalid."""


class PartitionValidationError(PreparationError):
    """Raised when generated partitions violate isolation or coverage rules."""


class ArtifactConflictError(PreparationError):
    """Raised when an existing artifact is semantically divergent."""


class HandoffValidationError(PreparationError):
    """Raised when a persisted preparation handoff is invalid."""


def _copy_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.copy(deep=True)


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(value))


def _tuple_mapping(mapping: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((str(key), copy.deepcopy(value)) for key, value in mapping.items())


def _mapping_from_tuple(items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in items}


@dataclass(frozen=True, slots=True)
class ConditionalNumericRule:
    """Declare one deterministic conditional numeric materialization rule."""

    column: str
    condition_column: str
    condition_value: Any
    blank_replacement: float | int
    strip_strings: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible representation."""
        return {
            "column": self.column,
            "condition_column": self.condition_column,
            "condition_value": copy.deepcopy(self.condition_value),
            "blank_replacement": self.blank_replacement,
            "strip_strings": self.strip_strings,
        }


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    """Immutable summary of a dataset-contract validation."""

    stage: str
    row_count: int
    column_count: int
    column_order: tuple[str, ...]
    dtypes: tuple[tuple[str, str], ...]
    target_counts: tuple[tuple[str, int], ...]
    observed_categories: tuple[tuple[str, tuple[Any, ...]], ...]
    checks: tuple[tuple[str, bool], ...]

    @property
    def is_valid(self) -> bool:
        """Return whether every recorded check passed."""
        return all(value for _, value in self.checks)

    def as_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible representation."""
        return {
            "stage": self.stage,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_order": list(self.column_order),
            "dtypes": dict(self.dtypes),
            "target_counts": dict(self.target_counts),
            "observed_categories": {
                column: list(values)
                for column, values in self.observed_categories
            },
            "checks": dict(self.checks),
            "valid": self.is_valid,
        }


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Prepared DataFrame plus deterministic materialization evidence."""

    _dataframe: pd.DataFrame
    materialized_counts: tuple[tuple[str, int], ...]
    invalid_conversion_counts: tuple[tuple[str, int], ...]
    rules: tuple[ConditionalNumericRule, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_dataframe", _copy_frame(self._dataframe))

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a defensive copy of the prepared DataFrame."""
        return _copy_frame(self._dataframe)

    def materialization_count(self, column: str) -> int:
        """Return the number of materialized values for one column."""
        return dict(self.materialized_counts).get(column, 0)

    def as_dict(self) -> dict[str, Any]:
        """Return materialization evidence without exposing mutable frames."""
        return {
            "materialized_counts": dict(self.materialized_counts),
            "invalid_conversion_counts": dict(self.invalid_conversion_counts),
            "rules": [rule.as_dict() for rule in self.rules],
        }


@dataclass(frozen=True, slots=True)
class DatasetRoles:
    """Defensive lineage, predictor, and target projections."""

    _lineage: pd.DataFrame
    _features: pd.DataFrame
    _target: pd.Series

    def __post_init__(self) -> None:
        object.__setattr__(self, "_lineage", _copy_frame(self._lineage))
        object.__setattr__(self, "_features", _copy_frame(self._features))
        object.__setattr__(self, "_target", self._target.copy(deep=True))

    @property
    def lineage(self) -> pd.DataFrame:
        return _copy_frame(self._lineage)

    @property
    def features(self) -> pd.DataFrame:
        return _copy_frame(self._features)

    @property
    def target(self) -> pd.Series:
        return self._target.copy(deep=True)


@dataclass(frozen=True, slots=True)
class ClassificationSplitPolicy:
    """Declarative split policy for an educational classification snapshot."""

    evaluation_mode: str
    purpose: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    stratify_by: str
    random_seed: int
    shuffle: bool
    educational_justification: str
    operational_validity: str = "unconfirmed"
    temporal_contract_status: str = "unresolved"
    feature_inference_availability: str = "unconfirmed"

    @property
    def second_stage_seed(self) -> int:
        """Return a deterministic distinct seed for the temporary split."""
        return self.random_seed + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_mode": self.evaluation_mode,
            "purpose": self.purpose,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "stratify_by": self.stratify_by,
            "random_seed": self.random_seed,
            "shuffle": self.shuffle,
            "educational_justification": self.educational_justification,
            "operational_validity": self.operational_validity,
            "temporal_contract_status": self.temporal_contract_status,
            "feature_inference_availability": self.feature_inference_availability,
            "stage_seeds": {
                "train_vs_temporary": self.random_seed,
                "validation_vs_test": self.second_stage_seed,
            },
        }


@dataclass(frozen=True, slots=True)
class DatasetPartitions:
    """Defensive train, validation, and test DataFrame partitions."""

    _train: pd.DataFrame
    _validation: pd.DataFrame
    _test: pd.DataFrame
    split_method: str
    rounding_method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_train", _copy_frame(self._train))
        object.__setattr__(self, "_validation", _copy_frame(self._validation))
        object.__setattr__(self, "_test", _copy_frame(self._test))

    @property
    def train(self) -> pd.DataFrame:
        return _copy_frame(self._train)

    @property
    def validation(self) -> pd.DataFrame:
        return _copy_frame(self._validation)

    @property
    def test(self) -> pd.DataFrame:
        return _copy_frame(self._test)

    def as_mapping(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True, slots=True)
class PartitionValidationReport:
    """Immutable partition-integrity evidence."""

    row_counts: tuple[tuple[str, int], ...]
    class_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    class_prevalence: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    membership: tuple[tuple[str, tuple[str, ...]], ...]
    checks: tuple[tuple[str, bool], ...]
    prevalence_tolerance: float

    @property
    def is_valid(self) -> bool:
        return all(value for _, value in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_counts": dict(self.row_counts),
            "class_counts": {
                name: dict(values) for name, values in self.class_counts
            },
            "class_prevalence": {
                name: dict(values) for name, values in self.class_prevalence
            },
            "membership": {
                name: list(values) for name, values in self.membership
            },
            "checks": dict(self.checks),
            "prevalence_tolerance": self.prevalence_tolerance,
            "valid": self.is_valid,
        }


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    """Result of one atomic preparation-artifact transaction."""

    statuses: tuple[tuple[str, str], ...]
    sha256: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "statuses": dict(self.statuses),
            "sha256": dict(self.sha256),
        }


@dataclass(frozen=True, slots=True)
class PreparationHandoff:
    """Validated persisted data and manifests for the next notebook."""

    _prepared: pd.DataFrame
    _train: pd.DataFrame
    _validation: pd.DataFrame
    _test: pd.DataFrame
    _manifests: tuple[tuple[str, Mapping[str, Any]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_prepared", _copy_frame(self._prepared))
        object.__setattr__(self, "_train", _copy_frame(self._train))
        object.__setattr__(self, "_validation", _copy_frame(self._validation))
        object.__setattr__(self, "_test", _copy_frame(self._test))
        object.__setattr__(
            self,
            "_manifests",
            tuple((name, _copy_mapping(value)) for name, value in self._manifests),
        )

    @property
    def prepared(self) -> pd.DataFrame:
        return _copy_frame(self._prepared)

    @property
    def train(self) -> pd.DataFrame:
        return _copy_frame(self._train)

    @property
    def validation(self) -> pd.DataFrame:
        return _copy_frame(self._validation)

    @property
    def test(self) -> pd.DataFrame:
        return _copy_frame(self._test)

    @property
    def manifests(self) -> dict[str, dict[str, Any]]:
        return {name: _copy_mapping(value) for name, value in self._manifests}


def fingerprint_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of the bytes stored in ``path``."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path.name}")
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_scalar(value: Any) -> Any:
    if value is pd.NA or value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def fingerprint_dataframe(dataframe: pd.DataFrame) -> str:
    """Return a stable logical SHA-256 fingerprint for a DataFrame.

    The canonical payload includes column order, dtype strings, index names,
    index values, and row values. It is independent of object identity.
    """
    frame = _copy_frame(dataframe)
    payload = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_names": [
            None if name is None else str(name) for name in frame.index.names
        ],
        "index": [
            [_normalize_scalar(item) for item in value]
            if isinstance(value, tuple)
            else _normalize_scalar(value)
            for value in frame.index.tolist()
        ],
        "records": [
            [_normalize_scalar(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataframe_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to deterministic UTF-8 CSV bytes."""
    buffer = io.StringIO(newline="")
    dataframe.to_csv(
        buffer,
        index=False,
        encoding=CSV_ENCODING,
        lineterminator=CSV_LINE_TERMINATOR,
        float_format="%.15g",
    )
    return buffer.getvalue().encode(CSV_ENCODING)


def fingerprint_dataframe_csv(dataframe: pd.DataFrame) -> str:
    """Return the SHA-256 digest of deterministic CSV bytes."""
    return hashlib.sha256(dataframe_csv_bytes(dataframe)).hexdigest()


def _is_string_series(series: pd.Series) -> bool:
    if pandas_types.is_string_dtype(series.dtype):
        return True
    non_missing = series.dropna()
    return non_missing.map(lambda value: isinstance(value, str)).all()


def _matches_expected_type(
    series: pd.Series,
    expectation: str,
    *,
    allow_numeric_text: bool,
) -> bool:
    normalized = expectation.strip().lower()
    if normalized == "string":
        return _is_string_series(series)
    if normalized == "integer":
        return pandas_types.is_integer_dtype(series.dtype)
    if normalized == "numeric":
        if pandas_types.is_numeric_dtype(series.dtype):
            return True
        if not allow_numeric_text:
            return False
        normalized_values = series.map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        non_blank = normalized_values.loc[
            ~normalized_values.map(
                lambda value: isinstance(value, str) and value == ""
            )
        ]
        converted = pd.to_numeric(non_blank, errors="coerce")
        return converted.notna().all()
    if normalized == "boolean":
        return pandas_types.is_bool_dtype(series.dtype)
    raise ValueError(f"Unsupported expected type: {expectation!r}")


def _validate_contract_configuration(
    *,
    column_order: Sequence[str],
    identifier_columns: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
) -> None:
    columns = tuple(column_order)
    identifiers = tuple(identifier_columns)
    features = tuple(feature_columns)
    if not columns:
        raise ValueError("column_order cannot be empty.")
    if not identifiers:
        raise ValueError("identifier_columns cannot be empty.")
    if not target_column:
        raise ValueError("target_column cannot be empty.")
    if len(set(columns)) != len(columns):
        raise ValueError("column_order contains duplicate names.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("identifier_columns contains duplicate names.")
    if len(set(features)) != len(features):
        raise ValueError("feature_columns contains duplicate names.")
    overlap = (set(identifiers) & set(features)) | ({target_column} & set(features))
    if overlap:
        raise ValueError(f"Dataset roles overlap: {sorted(overlap)}")
    declared = set(identifiers) | set(features) | {target_column}
    if declared != set(columns):
        missing = sorted(set(columns) - declared)
        extra = sorted(declared - set(columns))
        raise ValueError(
            "Role declarations must cover column_order exactly; "
            f"unassigned={missing}, outside_order={extra}."
        )


def validate_raw_dataset(
    dataframe: pd.DataFrame,
    *,
    column_order: Sequence[str],
    identifier_columns: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
    target_classes: Sequence[Any],
    categorical_expected_values: Mapping[str, Sequence[Any]],
    expected_types: Mapping[str, str],
    numeric_text_columns: Sequence[str] = (),
    allow_unexpected_columns: bool = False,
    require_all_expected_categories: bool = True,
) -> DatasetValidationReport:
    """Validate a raw dataset without mutating it."""
    _validate_contract_configuration(
        column_order=column_order,
        identifier_columns=identifier_columns,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    frame = _copy_frame(dataframe)
    expected_order = tuple(column_order)
    observed_order = tuple(str(column) for column in frame.columns)
    missing = [column for column in expected_order if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in expected_order]
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}")
    if unexpected and not allow_unexpected_columns:
        raise DatasetValidationError(f"Unexpected columns: {unexpected}")
    if not allow_unexpected_columns and observed_order != expected_order:
        raise DatasetValidationError(
            "Column order does not match the declared raw schema."
        )

    undeclared_types = [column for column in expected_order if column not in expected_types]
    missing_type_columns = [column for column in expected_types if column not in frame.columns]
    if undeclared_types or missing_type_columns:
        raise DatasetValidationError(
            "Expected type declarations must match the schema exactly; "
            f"undeclared={undeclared_types}, missing_columns={missing_type_columns}."
        )

    for column in identifier_columns:
        series = frame[column]
        if series.isna().any():
            raise DatasetValidationError(
                f"Identifier column '{column}' contains missing values."
            )
        blank_mask = series.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        if blank_mask.any():
            raise DatasetValidationError(
                f"Identifier column '{column}' contains blank values."
            )
        if series.duplicated(keep=False).any():
            raise DatasetValidationError(
                f"Identifier column '{column}' contains duplicate values."
            )

    expected_target = tuple(target_classes)
    if not expected_target:
        raise ValueError("target_classes cannot be empty.")
    target = frame[target_column]
    if target.isna().any():
        raise DatasetValidationError(
            f"Target column '{target_column}' contains missing values."
        )
    observed_target = tuple(pd.unique(target).tolist())
    unexpected_target = [value for value in observed_target if value not in expected_target]
    absent_target = [value for value in expected_target if value not in observed_target]
    if unexpected_target:
        raise DatasetValidationError(
            f"Target column '{target_column}' contains unexpected classes: "
            f"{unexpected_target}."
        )
    if absent_target:
        raise DatasetValidationError(
            f"Target column '{target_column}' is missing expected classes: "
            f"{absent_target}."
        )

    observed_categories: list[tuple[str, tuple[Any, ...]]] = []
    for column, expected_values in categorical_expected_values.items():
        if column not in frame.columns:
            raise DatasetValidationError(
                f"Categorical contract references missing column '{column}'."
            )
        series = frame[column]
        if series.isna().any():
            raise DatasetValidationError(
                f"Categorical column '{column}' contains missing values."
            )
        expected = tuple(expected_values)
        observed = tuple(pd.unique(series).tolist())
        unexpected_values = [value for value in observed if value not in expected]
        absent_values = [value for value in expected if value not in observed]
        if unexpected_values:
            raise DatasetValidationError(
                f"Categorical column '{column}' contains unexpected values: "
                f"{unexpected_values}."
            )
        if require_all_expected_categories and absent_values:
            raise DatasetValidationError(
                f"Categorical column '{column}' is missing expected values: "
                f"{absent_values}."
            )
        observed_categories.append((column, observed))

    numeric_text = set(numeric_text_columns)
    for column, expectation in expected_types.items():
        if not _matches_expected_type(
            frame[column],
            expectation,
            allow_numeric_text=column in numeric_text,
        ):
            raise DatasetValidationError(
                f"Column '{column}' does not match expected type "
                f"'{expectation}'; observed dtype is '{frame[column].dtype}'."
            )

    target_counts = tuple(
        (str(value), int((target == value).sum())) for value in expected_target
    )
    checks = (
        ("required_columns_present", True),
        ("column_order_valid", observed_order == expected_order or allow_unexpected_columns),
        ("identifiers_complete", True),
        ("identifiers_unique", True),
        ("target_valid", True),
        ("categories_valid", True),
        ("types_valid", True),
    )
    return DatasetValidationReport(
        stage="raw",
        row_count=len(frame),
        column_count=len(frame.columns),
        column_order=observed_order,
        dtypes=tuple((str(column), str(frame[column].dtype)) for column in frame.columns),
        target_counts=target_counts,
        observed_categories=tuple(observed_categories),
        checks=checks,
    )


def materialize_conditional_numeric_values(
    dataframe: pd.DataFrame,
    rule: ConditionalNumericRule,
) -> tuple[pd.DataFrame, int, int]:
    """Apply one conditional blank-to-number rule to a defensive copy."""
    frame = _copy_frame(dataframe)
    for column in (rule.column, rule.condition_column):
        if column not in frame.columns:
            raise ConditionalMaterializationError(
                f"Conditional materialization references missing column '{column}'."
            )

    source = frame[rule.column]
    if source.isna().any():
        raise ConditionalMaterializationError(
            f"Column '{rule.column}' contains null values; only explicit blank "
            "strings may be materialized by this rule."
        )

    normalized = source.map(
        lambda value: value.strip()
        if rule.strip_strings and isinstance(value, str)
        else value
    )
    blank_mask = normalized.map(
        lambda value: isinstance(value, str) and value == ""
    )
    condition_mask = frame[rule.condition_column].eq(rule.condition_value)
    invalid_blank_mask = blank_mask & ~condition_mask
    if invalid_blank_mask.any():
        count = int(invalid_blank_mask.sum())
        raise ConditionalMaterializationError(
            f"Column '{rule.column}' contains {count} blank value(s) where "
            f"'{rule.condition_column}' != {rule.condition_value!r}."
        )

    materialization_mask = blank_mask & condition_mask

    # Convert textual values before assigning the numeric replacement. Pandas 3
    # infers text columns as a strict string dtype, which rejects direct
    # assignment of numbers into the string-backed Series. Replacing authorized
    # blanks with a missing sentinel preserves the textual validation boundary;
    # the numeric replacement is applied only after conversion.
    numeric_candidate = normalized.mask(materialization_mask, pd.NA)
    converted = pd.to_numeric(numeric_candidate, errors="coerce")

    invalid_mask = converted.isna() & ~materialization_mask
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        samples = tuple(
            str(value)
            for value in normalized.loc[invalid_mask].head(5).tolist()
        )
        raise ConditionalMaterializationError(
            f"Column '{rule.column}' contains {invalid_count} non-convertible "
            f"value(s); samples={samples}."
        )

    converted = converted.astype("float64")
    converted.loc[materialization_mask] = float(rule.blank_replacement)
    frame[rule.column] = converted
    return frame, int(materialization_mask.sum()), invalid_count


def prepare_tabular_dataset(
    dataframe: pd.DataFrame,
    *,
    conditional_numeric_rules: Sequence[ConditionalNumericRule],
) -> PreparedDataset:
    """Create a defensive prepared projection using only explicit rules."""
    prepared = _copy_frame(dataframe)
    original_columns = tuple(prepared.columns)
    original_index = prepared.index.copy(deep=True)
    materialized_counts: list[tuple[str, int]] = []
    invalid_counts: list[tuple[str, int]] = []
    rules = tuple(copy.deepcopy(tuple(conditional_numeric_rules)))

    for rule in rules:
        prepared, materialized_count, invalid_count = (
            materialize_conditional_numeric_values(prepared, rule)
        )
        materialized_counts.append((rule.column, materialized_count))
        invalid_counts.append((rule.column, invalid_count))

    if tuple(prepared.columns) != original_columns:
        raise ConditionalMaterializationError(
            "Preparation unexpectedly changed the column order."
        )
    if not prepared.index.equals(original_index):
        raise ConditionalMaterializationError(
            "Preparation unexpectedly changed the DataFrame index."
        )
    if len(prepared) != len(dataframe):
        raise ConditionalMaterializationError(
            "Preparation unexpectedly changed the row count."
        )

    return PreparedDataset(
        _dataframe=prepared,
        materialized_counts=tuple(materialized_counts),
        invalid_conversion_counts=tuple(invalid_counts),
        rules=rules,
    )


def validate_prepared_dataset(
    raw_dataframe: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
    *,
    column_order: Sequence[str],
    identifier_columns: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
    target_classes: Sequence[Any],
    categorical_expected_values: Mapping[str, Sequence[Any]],
    expected_types: Mapping[str, str],
    authorized_changed_columns: Sequence[str],
    expected_row_count: int | None = None,
    expected_materialized_counts: Mapping[str, int] | None = None,
    observed_materialized_counts: Mapping[str, int] | None = None,
) -> DatasetValidationReport:
    """Validate prepared data and prove preservation of the raw projection."""
    raw = _copy_frame(raw_dataframe)
    prepared = _copy_frame(prepared_dataframe)
    if expected_row_count is not None and len(prepared) != expected_row_count:
        raise DatasetValidationError(
            f"Prepared row count is {len(prepared)}, expected {expected_row_count}."
        )
    if raw.shape != prepared.shape:
        raise DatasetValidationError(
            f"Prepared shape {prepared.shape} differs from raw shape {raw.shape}."
        )
    if tuple(raw.columns) != tuple(prepared.columns):
        raise DatasetValidationError("Prepared column order differs from raw data.")
    if not raw.index.equals(prepared.index):
        raise DatasetValidationError("Prepared index differs from raw data.")

    authorized = set(authorized_changed_columns)
    for column in raw.columns:
        if column in authorized:
            continue
        try:
            pd.testing.assert_series_equal(
                raw[column], prepared[column], check_dtype=True, check_names=True
            )
        except AssertionError as exc:
            raise DatasetValidationError(
                f"Unauthorized values changed in column '{column}'."
            ) from exc

    if expected_materialized_counts is not None:
        observed = dict(observed_materialized_counts or {})
        for column, expected_count in expected_materialized_counts.items():
            if observed.get(column) != expected_count:
                raise DatasetValidationError(
                    f"Materialized count for '{column}' is {observed.get(column)}, "
                    f"expected {expected_count}."
                )

    report = validate_raw_dataset(
        prepared,
        column_order=column_order,
        identifier_columns=identifier_columns,
        feature_columns=feature_columns,
        target_column=target_column,
        target_classes=target_classes,
        categorical_expected_values=categorical_expected_values,
        expected_types=expected_types,
        numeric_text_columns=(),
        allow_unexpected_columns=False,
        require_all_expected_categories=True,
    )
    return DatasetValidationReport(
        stage="prepared",
        row_count=report.row_count,
        column_count=report.column_count,
        column_order=report.column_order,
        dtypes=report.dtypes,
        target_counts=report.target_counts,
        observed_categories=report.observed_categories,
        checks=report.checks
        + (
            ("raw_shape_preserved", True),
            ("raw_index_preserved", True),
            ("unauthorized_columns_unchanged", True),
            ("materialization_counts_valid", True),
        ),
    )


def separate_dataset_roles(
    dataframe: pd.DataFrame,
    *,
    identifier_columns: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
) -> DatasetRoles:
    """Return defensive lineage, X, and y projections in declared order."""
    frame = _copy_frame(dataframe)
    required = tuple(identifier_columns) + tuple(feature_columns) + (target_column,)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetValidationError(f"Role separation is missing columns: {missing}")
    if set(identifier_columns) & set(feature_columns):
        raise DatasetValidationError("Identifier columns cannot be predictors.")
    if target_column in feature_columns:
        raise DatasetValidationError("Target column cannot be a predictor.")
    return DatasetRoles(
        _lineage=frame.loc[:, list(identifier_columns)],
        _features=frame.loc[:, list(feature_columns)],
        _target=frame.loc[:, target_column],
    )


def validate_split_policy(
    policy: ClassificationSplitPolicy,
    *,
    known_columns: Iterable[str],
) -> dict[str, bool]:
    """Validate an educational stratified-random snapshot policy."""
    if policy.evaluation_mode != "stratified_random_snapshot":
        raise SplitPolicyError(
            "evaluation_mode must be 'stratified_random_snapshot'."
        )
    if policy.purpose != "educational_benchmark":
        raise SplitPolicyError("purpose must be 'educational_benchmark'.")
    if not policy.educational_justification.strip():
        raise SplitPolicyError("An educational justification is required.")
    fractions = (
        policy.train_fraction,
        policy.validation_fraction,
        policy.test_fraction,
    )
    if any(not isinstance(value, (int, float)) for value in fractions):
        raise SplitPolicyError("Split fractions must be numeric.")
    if any(value <= 0 for value in fractions):
        raise SplitPolicyError("Split fractions must be greater than zero.")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SplitPolicyError("Split fractions must sum to 1.0.")
    if policy.stratify_by not in set(known_columns):
        raise SplitPolicyError(
            f"Unknown stratification target: {policy.stratify_by!r}."
        )
    if isinstance(policy.random_seed, bool) or not isinstance(policy.random_seed, int):
        raise SplitPolicyError("random_seed must be an integer.")
    if not policy.shuffle:
        raise SplitPolicyError(
            "shuffle must be true for a stratified random snapshot."
        )
    if policy.operational_validity != "unconfirmed":
        raise SplitPolicyError(
            "operational_validity must remain 'unconfirmed' for this benchmark."
        )
    if policy.temporal_contract_status != "unresolved":
        raise SplitPolicyError(
            "temporal_contract_status must remain 'unresolved'."
        )
    if policy.feature_inference_availability != "unconfirmed":
        raise SplitPolicyError(
            "feature_inference_availability must remain 'unconfirmed'."
        )
    return {
        "mode_valid": True,
        "purpose_valid": True,
        "fractions_valid": True,
        "stratification_target_known": True,
        "seed_valid": True,
        "shuffle_valid": True,
        "educational_boundary_valid": True,
        "operational_validity_unconfirmed": True,
        "temporal_contract_unresolved": True,
        "feature_inference_availability_unconfirmed": True,
    }


def _stable_membership_key(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if columns:
        key_frame = frame.loc[:, list(columns)].copy(deep=True)
        return key_frame.apply(
            lambda row: json.dumps(
                [_normalize_scalar(value) for value in row.tolist()],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            axis=1,
        )
    return pd.Series(
        [
            json.dumps(_normalize_scalar(value), ensure_ascii=False, allow_nan=False)
            for value in frame.index.tolist()
        ],
        index=frame.index,
    )


def split_classification_dataset(
    dataframe: pd.DataFrame,
    *,
    policy: ClassificationSplitPolicy,
    identifier_columns: Sequence[str],
    target_classes: Sequence[Any],
) -> DatasetPartitions:
    """Create deterministic stratified partitions with stable membership.

    Rows are first ordered by the declared identifier key so membership does
    not depend on incidental input ordering. Returned rows are then restored to
    their original source positions within each partition.
    """
    frame = _copy_frame(dataframe)
    validate_split_policy(policy, known_columns=frame.columns)
    for column in identifier_columns:
        if column not in frame.columns:
            raise SplitPolicyError(f"Identifier column not found: {column}")
    if frame.empty:
        raise SplitPolicyError("Cannot split an empty dataset.")
    target = frame[policy.stratify_by]
    absent = [value for value in target_classes if value not in set(target)]
    if absent:
        raise SplitPolicyError(
            f"Cannot stratify because expected target classes are absent: {absent}."
        )

    working = frame.copy(deep=True)
    working["__source_position__"] = range(len(working))
    working["__membership_key__"] = _stable_membership_key(
        working, identifier_columns
    ).to_numpy()
    if working["__membership_key__"].duplicated().any():
        raise SplitPolicyError("Membership keys must be unique before splitting.")
    canonical = working.sort_values(
        "__membership_key__", kind="stable"
    ).reset_index(drop=True)
    canonical_positions = list(range(len(canonical)))
    temporary_fraction = policy.validation_fraction + policy.test_fraction

    train_positions, temporary_positions = train_test_split(
        canonical_positions,
        test_size=temporary_fraction,
        random_state=policy.random_seed,
        shuffle=policy.shuffle,
        stratify=canonical[policy.stratify_by],
    )
    temporary = canonical.iloc[temporary_positions]
    relative_test_fraction = policy.test_fraction / temporary_fraction
    validation_positions, test_positions = train_test_split(
        temporary_positions,
        test_size=relative_test_fraction,
        random_state=policy.second_stage_seed,
        shuffle=policy.shuffle,
        stratify=temporary[policy.stratify_by],
    )

    def project(positions: Sequence[int]) -> pd.DataFrame:
        source_positions = sorted(
            int(value)
            for value in canonical.iloc[list(positions)]["__source_position__"].tolist()
        )
        return frame.iloc[source_positions].copy(deep=True)

    return DatasetPartitions(
        _train=project(train_positions),
        _validation=project(validation_positions),
        _test=project(test_positions),
        split_method=(
            "two_stage_sklearn_train_test_split_with_stratification_and_"
            "identifier_sorted_membership"
        ),
        rounding_method=(
            "scikit-learn float test_size semantics: each held-out size is "
            "rounded up with ceil; the remainder is assigned to the first set"
        ),
    )


def _membership_values(
    frame: pd.DataFrame, identifier_columns: Sequence[str]
) -> tuple[str, ...]:
    return tuple(_stable_membership_key(frame, identifier_columns).tolist())


def validate_dataset_partitions(
    source_dataframe: pd.DataFrame,
    partitions: DatasetPartitions,
    *,
    identifier_columns: Sequence[str],
    target_column: str,
    target_classes: Sequence[Any],
    prevalence_tolerance: float = 0.02,
) -> PartitionValidationReport:
    """Validate class preservation, isolation, coverage, and stable order."""
    if prevalence_tolerance < 0:
        raise ValueError("prevalence_tolerance cannot be negative.")
    source = _copy_frame(source_dataframe)
    partition_map = partitions.as_mapping()
    source_membership = _membership_values(source, identifier_columns)
    source_membership_set = set(source_membership)
    if len(source_membership_set) != len(source):
        raise PartitionValidationError("Source membership identifiers are not unique.")

    source_counts = source[target_column].value_counts(dropna=False)
    source_prevalence = source[target_column].value_counts(normalize=True, dropna=False)
    row_counts: list[tuple[str, int]] = []
    class_counts: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    class_prevalence: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    membership: list[tuple[str, tuple[str, ...]]] = []
    membership_sets: dict[str, set[str]] = {}
    source_position = {value: position for position, value in enumerate(source_membership)}

    for name in ("train", "validation", "test"):
        frame = partition_map[name]
        if frame.empty:
            raise PartitionValidationError(f"Partition '{name}' is empty.")
        if tuple(frame.columns) != tuple(source.columns):
            raise PartitionValidationError(
                f"Partition '{name}' does not preserve source columns."
            )
        observed = tuple(pd.unique(frame[target_column]).tolist())
        absent = [value for value in target_classes if value not in observed]
        if absent:
            raise PartitionValidationError(
                f"Partition '{name}' is missing expected classes: {absent}."
            )
        values = _membership_values(frame, identifier_columns)
        values_set = set(values)
        if len(values_set) != len(frame):
            raise PartitionValidationError(
                f"Partition '{name}' contains duplicate membership identifiers."
            )
        positions = [source_position[value] for value in values]
        if positions != sorted(positions):
            raise PartitionValidationError(
                f"Partition '{name}' is not in stable source-row order."
            )
        counts = tuple(
            (str(value), int((frame[target_column] == value).sum()))
            for value in target_classes
        )
        prevalences = tuple(
            (
                str(value),
                float((frame[target_column] == value).mean()),
            )
            for value in target_classes
        )
        for value in target_classes:
            observed_prevalence = float((frame[target_column] == value).mean())
            expected_prevalence = float(source_prevalence.get(value, 0.0))
            if abs(observed_prevalence - expected_prevalence) > prevalence_tolerance:
                raise PartitionValidationError(
                    f"Partition '{name}' prevalence for {value!r} differs from "
                    f"source by more than {prevalence_tolerance:.4f}."
                )
        row_counts.append((name, len(frame)))
        class_counts.append((name, counts))
        class_prevalence.append((name, prevalences))
        membership.append((name, values))
        membership_sets[name] = values_set

    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlaps = {
        f"{left}_{right}": membership_sets[left] & membership_sets[right]
        for left, right in pairs
    }
    if any(overlaps.values()):
        raise PartitionValidationError(
            "Partition membership overlaps were detected."
        )
    union = set().union(*membership_sets.values())
    if union != source_membership_set:
        missing = len(source_membership_set - union)
        extra = len(union - source_membership_set)
        raise PartitionValidationError(
            f"Partition coverage mismatch: missing={missing}, extra={extra}."
        )
    if sum(dict(row_counts).values()) != len(source):
        raise PartitionValidationError("Partition row counts do not cover the source.")

    checks = (
        ("all_partitions_present", True),
        ("all_classes_present", True),
        ("class_prevalence_within_tolerance", True),
        ("indices_disjoint_by_membership", True),
        ("customer_identifiers_disjoint", True),
        ("no_shared_rows", True),
        ("full_coverage", True),
        ("row_count_preserved", True),
        ("stable_source_order", True),
        ("test_holdout_isolated", True),
    )
    return PartitionValidationReport(
        row_counts=tuple(row_counts),
        class_counts=tuple(class_counts),
        class_prevalence=tuple(class_prevalence),
        membership=tuple(membership),
        checks=checks,
        prevalence_tolerance=prevalence_tolerance,
    )


def _relative_posix(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError(f"Artifact path must be project-relative: {candidate.name}")
    normalized = Path(os.path.normpath(str(candidate)))
    if normalized == Path(".") or ".." in normalized.parts:
        raise ValueError(f"Invalid project-relative artifact path: {candidate}")
    return normalized.as_posix()


def runtime_versions() -> dict[str, str]:
    """Return versions relevant to deterministic preparation."""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn_version,
    }


_SOURCE_IDENTITY_CONTRACT_VERSION: Final[str] = "source-identity.v1"


def build_source_identity(
    *,
    dataset_slug: str,
    source_path: str | Path,
    source_sha256: str,
    prepared_sha256: str,
    raw_report: DatasetValidationReport,
    source_producer_revision: str | None = None,
    source_producer_revision_unavailable_reason: str | None = None,
    contract_version: str = _SOURCE_IDENTITY_CONTRACT_VERSION,
) -> dict[str, Any]:
    """Build a portable, path-free source-identity object.

    The identity is correlatable via logical dataset id, content hash, and
    row/column identity alone, without any repository-relative or absolute
    path. ``source_producer_revision`` is populated only when a real,
    provable revision is available; otherwise it stays ``None`` and the
    reason field explains why no revision claim is made.
    """
    if source_producer_revision is None and not source_producer_revision_unavailable_reason:
        raise ValueError(
            "source_producer_revision_unavailable_reason is required when "
            "source_producer_revision is not supplied."
        )
    return {
        "contract_version": contract_version,
        "dataset_logical_identity": dataset_slug,
        "source_filename_or_logical_source": Path(source_path).name,
        "row_column_identity": {
            "row_count": raw_report.row_count,
            "column_count": raw_report.column_count,
            "column_order": list(raw_report.column_order),
        },
        "content_hash": {
            "source_sha256": source_sha256,
            "prepared_sha256": prepared_sha256,
        },
        "source_revision_or_producer_revision": source_producer_revision,
        "source_revision_unavailable_reason": (
            None
            if source_producer_revision is not None
            else source_producer_revision_unavailable_reason
        ),
        "artifact_type_version": CONTRACT_VERSION,
        "artifact_sha256": source_sha256,
    }


def build_preparation_manifest(
    *,
    dataset_slug: str,
    source_path: str | Path,
    source_sha256: str,
    prepared_path: str | Path,
    prepared_sha256: str,
    raw_report: DatasetValidationReport,
    prepared_report: DatasetValidationReport,
    preparation: PreparedDataset,
    raw_fingerprint_before: str,
    raw_fingerprint_after: str,
    source_sha256_after: str,
    deterministic_rules: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
    contract_version: str = CONTRACT_VERSION,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned preparation manifest."""
    manifest = {
        "schema_version": "preparation-manifest.v1",
        "artifact_type": "preparation_manifest",
        "dataset_slug": dataset_slug,
        "contract_version": contract_version,
        "source_path": _relative_posix(source_path),
        "source_filename": Path(source_path).name,
        "source_sha256": source_sha256,
        "source_sha256_after": source_sha256_after,
        "prepared_path": _relative_posix(prepared_path),
        "prepared_sha256": prepared_sha256,
        "source_row_count": raw_report.row_count,
        "prepared_row_count": prepared_report.row_count,
        "source_column_count": raw_report.column_count,
        "prepared_column_count": prepared_report.column_count,
        "column_order": list(prepared_report.column_order),
        "deterministic_rules": copy.deepcopy(list(deterministic_rules)),
        "materialized_value_counts": dict(preparation.materialized_counts),
        "invalid_conversion_counts": dict(preparation.invalid_conversion_counts),
        "row_preservation_checks": {
            "row_count_preserved": raw_report.row_count == prepared_report.row_count,
            "column_count_preserved": raw_report.column_count == prepared_report.column_count,
            "column_order_preserved": raw_report.column_order == prepared_report.column_order,
        },
        "raw_immutability_checks": {
            "logical_fingerprint_before": raw_fingerprint_before,
            "logical_fingerprint_after": raw_fingerprint_after,
            "logical_fingerprint_preserved": raw_fingerprint_before == raw_fingerprint_after,
            "source_sha256_preserved": source_sha256 == source_sha256_after,
        },
        "runtime_versions": runtime_versions(),
        "readiness": _copy_mapping(readiness),
    }
    if source_identity is not None:
        manifest["source_identity"] = _copy_mapping(source_identity)
    return manifest


def build_feature_manifest(
    *,
    dataset_slug: str,
    identifier_columns: Sequence[str],
    feature_columns: Sequence[str],
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    categorical_expected_values: Mapping[str, Sequence[Any]],
    target_column: str,
    target_classes: Sequence[Any],
    positive_target_class: Any,
    target_encoding: Mapping[Any, int],
    expected_dtypes: Mapping[str, Any],
    preprocessing_contract: Mapping[str, Any],
    prohibited_predictors: Sequence[str],
) -> dict[str, Any]:
    """Build an ordered feature and future-preprocessing contract."""
    return {
        "schema_version": "feature-manifest.v1",
        "artifact_type": "feature_manifest",
        "dataset_slug": dataset_slug,
        "identifier_columns": list(identifier_columns),
        "feature_columns": list(feature_columns),
        "numerical_features": list(numerical_features),
        "categorical_features": list(categorical_features),
        "categorical_expected_values": {
            column: list(values)
            for column, values in categorical_expected_values.items()
        },
        "target_column": target_column,
        "target_classes": list(target_classes),
        "positive_target_class": positive_target_class,
        "target_encoding_contract": {
            str(key): value for key, value in target_encoding.items()
        },
        "expected_dtypes": _copy_mapping(expected_dtypes),
        "preprocessing_contract": _copy_mapping(preprocessing_contract),
        "prohibited_predictors": list(prohibited_predictors),
    }


def build_split_manifest(
    *,
    dataset_slug: str,
    policy: ClassificationSplitPolicy,
    partitions: DatasetPartitions,
    validation: PartitionValidationReport,
    partition_paths: Mapping[str, str | Path],
    partition_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build a versioned split manifest with explicit membership."""
    paths = {name: _relative_posix(path) for name, path in partition_paths.items()}
    required = {"train", "validation", "test"}
    if set(paths) != required or set(partition_sha256) != required:
        raise ValueError("Partition paths and fingerprints must cover train/validation/test.")
    report = validation.as_dict()
    return {
        "schema_version": "split-manifest.v1",
        "artifact_type": "split_manifest",
        "dataset_slug": dataset_slug,
        **policy.as_dict(),
        "split_method": partitions.split_method,
        "rounding_method": partitions.rounding_method,
        "partition_paths": paths,
        "partition_sha256": dict(partition_sha256),
        "row_counts": report["row_counts"],
        "class_counts": report["class_counts"],
        "class_prevalence": report["class_prevalence"],
        "membership": report["membership"],
        "isolation_checks": report["checks"],
        "prevalence_tolerance": report["prevalence_tolerance"],
        "test_holdout_policy": (
            "The test partition is isolated and must not be used for feature, "
            "preprocessing, model, hyperparameter, threshold, or metric selection."
        ),
        "operational_modeling_ready": False,
        "educational_model_selection_ready": True,
    }


def build_quality_evidence(
    *,
    dataset_slug: str,
    raw_report: DatasetValidationReport,
    prepared_report: DatasetValidationReport,
    partition_report: PartitionValidationReport,
    preparation: PreparedDataset,
    fingerprints: Mapping[str, Any],
    readiness: Mapping[str, Any],
    preservation_checks: Mapping[str, Any],
) -> dict[str, Any]:
    """Build consolidated preparation quality evidence."""
    return {
        "schema_version": "quality-evidence.v1",
        "artifact_type": "preparation_quality_evidence",
        "dataset_slug": dataset_slug,
        "raw_schema_validation": raw_report.as_dict(),
        "prepared_dataset_validation": prepared_report.as_dict(),
        "partition_validation": partition_report.as_dict(),
        "materializations": preparation.as_dict(),
        "fingerprint_checks": _copy_mapping(fingerprints),
        "preservation_checks": _copy_mapping(preservation_checks),
        "readiness": _copy_mapping(readiness),
        "operational_block": {
            "operational_modeling_ready": False,
            "operational_validity": "unconfirmed",
            "temporal_contract_status": "unresolved",
            "feature_inference_availability": "unconfirmed",
        },
    }


def _semantic_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_SEMANTIC_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_normalize(item) for item in value]
    return _normalize_scalar(value)


def semantically_equivalent(left: Any, right: Any) -> bool:
    """Compare values while ignoring explicitly volatile timestamp fields."""
    return _semantic_normalize(left) == _semantic_normalize(right)


def _resolve_project_artifact_path(root: Path, relative_path: str | Path) -> Path:
    relative = Path(_relative_posix(relative_path))
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes project root: {relative.as_posix()}") from exc
    return candidate


def _validate_staged_csv(path: Path, expected: pd.DataFrame) -> None:
    reloaded = pd.read_csv(path)
    if tuple(reloaded.columns) != tuple(expected.columns):
        raise PreparationError(
            f"Staged CSV column validation failed for {path.name}."
        )
    if len(reloaded) != len(expected):
        raise PreparationError(
            f"Staged CSV row validation failed for {path.name}."
        )


def write_preparation_artifacts(
    *,
    project_root: str | Path,
    csv_artifacts: Mapping[str | Path, pd.DataFrame],
    json_artifacts: Mapping[str | Path, Mapping[str, Any]],
    overwrite: bool = False,
) -> ArtifactWriteResult:
    """Persist a complete artifact set with staging, conflict checks, and rollback.

    Existing byte-equivalent CSVs and semantically equivalent JSON documents
    are accepted as idempotent. Divergent files fail before promotion unless
    ``overwrite=True`` was supplied deliberately.
    """
    root = Path(project_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not csv_artifacts and not json_artifacts:
        raise ValueError("At least one artifact must be supplied.")

    csv_inputs = {
        _relative_posix(path): _copy_frame(frame)
        for path, frame in csv_artifacts.items()
    }
    json_inputs = {
        _relative_posix(path): _copy_mapping(payload)
        for path, payload in json_artifacts.items()
    }
    overlap = set(csv_inputs) & set(json_inputs)
    if overlap:
        raise ValueError(f"Artifact paths are duplicated across formats: {sorted(overlap)}")

    transaction_id = uuid.uuid4().hex
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".preparation-staging-{transaction_id}-", dir=root)
    )
    backup_root = Path(
        tempfile.mkdtemp(prefix=f".preparation-backup-{transaction_id}-", dir=root)
    )
    all_inputs: dict[str, tuple[str, Any]] = {
        **{path: ("csv", value) for path, value in csv_inputs.items()},
        **{path: ("json", value) for path, value in json_inputs.items()},
    }
    staged_paths: dict[str, Path] = {}
    statuses: dict[str, str] = {}
    digests: dict[str, str] = {}
    promoted: list[str] = []
    backed_up: list[str] = []

    try:
        for relative, (kind, payload) in all_inputs.items():
            staged = staging_root / Path(relative)
            staged.parent.mkdir(parents=True, exist_ok=True)
            if kind == "csv":
                staged.write_bytes(dataframe_csv_bytes(payload))
                _validate_staged_csv(staged, payload)
            else:
                staged.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=False,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                json.loads(staged.read_text(encoding="utf-8"))
            staged_paths[relative] = staged
            digests[relative] = fingerprint_file(staged)

        # Detect every conflict before mutating any destination.
        for relative, (kind, payload) in all_inputs.items():
            destination = _resolve_project_artifact_path(root, relative)
            if not destination.exists():
                statuses[relative] = "created"
                continue
            if not destination.is_file():
                raise ArtifactConflictError(
                    f"Artifact destination is not a file: {relative}"
                )
            if kind == "csv":
                equivalent = destination.read_bytes() == staged_paths[relative].read_bytes()
            else:
                try:
                    existing_payload = json.loads(destination.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ArtifactConflictError(
                        f"Existing JSON artifact is unreadable: {relative}"
                    ) from exc
                equivalent = semantically_equivalent(existing_payload, payload)
            if equivalent:
                statuses[relative] = "reused_equivalent"
            elif overwrite:
                statuses[relative] = "overwritten"
            else:
                raise ArtifactConflictError(
                    f"Existing artifact is semantically divergent: {relative}. "
                    "No overwrite was performed."
                )

        for relative in sorted(all_inputs):
            status = statuses[relative]
            if status == "reused_equivalent":
                digests[relative] = fingerprint_file(
                    _resolve_project_artifact_path(root, relative)
                )
                continue
            destination = _resolve_project_artifact_path(root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = backup_root / Path(relative)
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backed_up.append(relative)
            os.replace(staged_paths[relative], destination)
            promoted.append(relative)
            digests[relative] = fingerprint_file(destination)

    except Exception:
        for relative in reversed(promoted):
            destination = _resolve_project_artifact_path(root, relative)
            destination.unlink(missing_ok=True)
        for relative in reversed(backed_up):
            backup = backup_root / Path(relative)
            destination = _resolve_project_artifact_path(root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)

    return ArtifactWriteResult(
        statuses=tuple(sorted(statuses.items())),
        sha256=tuple(sorted(digests.items())),
    )


def _load_json_artifact(root: Path, relative_path: str | Path) -> dict[str, Any]:
    path = _resolve_project_artifact_path(root, relative_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HandoffValidationError(
            f"Required handoff artifact is missing: {_relative_posix(relative_path)}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise HandoffValidationError(
            f"Invalid JSON handoff artifact: {_relative_posix(relative_path)}"
        ) from exc
    if not isinstance(payload, dict):
        raise HandoffValidationError(
            f"Handoff artifact must contain a JSON object: {_relative_posix(relative_path)}"
        )
    return payload


def load_and_validate_preparation_handoff(
    *,
    project_root: str | Path,
    preparation_manifest_path: str | Path,
    feature_manifest_path: str | Path,
    split_manifest_path: str | Path,
    quality_evidence_path: str | Path,
) -> PreparationHandoff:
    """Load persisted artifacts, verify fingerprints, and return the handoff."""
    root = Path(project_root).expanduser().resolve()
    preparation_manifest = _load_json_artifact(root, preparation_manifest_path)
    feature_manifest = _load_json_artifact(root, feature_manifest_path)
    split_manifest = _load_json_artifact(root, split_manifest_path)
    quality_evidence = _load_json_artifact(root, quality_evidence_path)

    expected_schemas = {
        "preparation": (preparation_manifest, "preparation-manifest.v1"),
        "feature": (feature_manifest, "feature-manifest.v1"),
        "split": (split_manifest, "split-manifest.v1"),
        "quality": (quality_evidence, "quality-evidence.v1"),
    }
    for name, (payload, expected) in expected_schemas.items():
        if payload.get("schema_version") != expected:
            raise HandoffValidationError(
                f"Unexpected {name} schema_version: {payload.get('schema_version')!r}."
            )
    slugs = {
        payload.get("dataset_slug")
        for payload, _ in expected_schemas.values()
    }
    if len(slugs) != 1 or None in slugs:
        raise HandoffValidationError("Handoff dataset_slug values are inconsistent.")
    if split_manifest.get("operational_validity") != "unconfirmed":
        raise HandoffValidationError(
            "Handoff operational_validity must remain unconfirmed."
        )
    if split_manifest.get("operational_modeling_ready") is not False:
        raise HandoffValidationError(
            "Handoff operational_modeling_ready must remain false."
        )

    prepared_relative = preparation_manifest.get("prepared_path")
    partition_paths = split_manifest.get("partition_paths")
    if not isinstance(prepared_relative, str) or not isinstance(partition_paths, dict):
        raise HandoffValidationError("Handoff data paths are missing or invalid.")

    prepared_path = _resolve_project_artifact_path(root, prepared_relative)
    expected_prepared_sha = preparation_manifest.get("prepared_sha256")
    if fingerprint_file(prepared_path) != expected_prepared_sha:
        raise HandoffValidationError("Prepared CSV fingerprint mismatch.")
    prepared = pd.read_csv(prepared_path)
    expected_columns = feature_manifest.get("identifier_columns", []) + feature_manifest.get(
        "feature_columns", []
    ) + [feature_manifest.get("target_column")]
    # Feature order is not necessarily contiguous with identifiers/target in the
    # source schema, so the preparation manifest remains authoritative.
    authoritative_columns = preparation_manifest.get("column_order")
    if list(prepared.columns) != authoritative_columns:
        raise HandoffValidationError("Prepared CSV column order mismatch.")
    if len(prepared) != preparation_manifest.get("prepared_row_count"):
        raise HandoffValidationError("Prepared CSV row count mismatch.")
    if set(expected_columns) != set(authoritative_columns):
        raise HandoffValidationError("Feature roles do not cover prepared columns.")

    loaded_partitions: dict[str, pd.DataFrame] = {}
    partition_sha = split_manifest.get("partition_sha256", {})
    for name in ("train", "validation", "test"):
        relative = partition_paths.get(name)
        if not isinstance(relative, str):
            raise HandoffValidationError(f"Missing path for partition '{name}'.")
        path = _resolve_project_artifact_path(root, relative)
        if fingerprint_file(path) != partition_sha.get(name):
            raise HandoffValidationError(
                f"Partition CSV fingerprint mismatch: {name}."
            )
        frame = pd.read_csv(path)
        if list(frame.columns) != authoritative_columns:
            raise HandoffValidationError(
                f"Partition '{name}' column order mismatch."
            )
        if len(frame) != split_manifest.get("row_counts", {}).get(name):
            raise HandoffValidationError(
                f"Partition '{name}' row count mismatch."
            )
        loaded_partitions[name] = frame

    partition_set = DatasetPartitions(
        _train=loaded_partitions["train"],
        _validation=loaded_partitions["validation"],
        _test=loaded_partitions["test"],
        split_method=str(split_manifest.get("split_method")),
        rounding_method=str(split_manifest.get("rounding_method")),
    )
    validate_dataset_partitions(
        prepared,
        partition_set,
        identifier_columns=feature_manifest["identifier_columns"],
        target_column=feature_manifest["target_column"],
        target_classes=feature_manifest["target_classes"],
        prevalence_tolerance=float(split_manifest.get("prevalence_tolerance", 0.02)),
    )

    manifests = (
        ("preparation_manifest", preparation_manifest),
        ("feature_manifest", feature_manifest),
        ("split_manifest", split_manifest),
        ("quality_evidence", quality_evidence),
    )
    return PreparationHandoff(
        _prepared=prepared,
        _train=loaded_partitions["train"],
        _validation=loaded_partitions["validation"],
        _test=loaded_partitions["test"],
        _manifests=manifests,
    )
