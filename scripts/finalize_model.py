"""Generic, deterministic final-model training and bundle materialization.

The module is dataset-agnostic. Dataset roles, paths, selected estimator,
hyperparameters, thresholds, and upstream contracts are supplied by callers.
It enforces a train+validation final fit, a frozen decision contract before test
access, one probability evaluation of test, complete-pipeline serialization,
and atomic/idempotent artifact persistence.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import joblib
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from scripts.prepare_data import load_and_validate_preparation_handoff
from scripts.select_models import (
    build_candidate_pipeline,
    load_and_validate_model_selection_handoff,
)


FINAL_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "final-pipeline.joblib",
    "final-model-manifest.json",
    "final-test-evidence.json",
    "inference-bundle.json",
    "final-model-handoff.json",
)

SCHEMAS: Mapping[str, tuple[str, str]] = {
    "final-model-manifest.json": (
        "final-model-manifest.v1",
        "final_model_manifest",
    ),
    "final-test-evidence.json": (
        "final-test-evidence.v1",
        "final_test_evidence",
    ),
    "inference-bundle.json": ("inference-bundle.v1", "inference_bundle"),
    "final-model-handoff.json": (
        "final-model-handoff.v1",
        "final_model_handoff",
    ),
}

_VOLATILE_KEYS = frozenset(
    {
        "created_at",
        "generated_at",
        "timestamp",
        "fit_duration_seconds",
        "duration_seconds",
        "byte_sha256",
    }
)


class FinalizationError(RuntimeError):
    """Base error for final-model operations."""


class FinalizationContractError(FinalizationError, ValueError):
    """Raised when a supplied finalization contract is invalid."""


class UpstreamHandoffError(FinalizationError):
    """Raised when an upstream handoff is invalid or inconsistent."""


class TestAccessError(FinalizationError):
    """Raised when test is accessed before the frozen/fitted gate."""


class DuplicateTestEvaluationError(FinalizationError):
    """Raised when test probability evaluation is attempted twice."""


class SerializationValidationError(FinalizationError):
    """Raised when a serialized pipeline fails integrity validation."""


class ArtifactConflictError(FinalizationError):
    """Raised for partial or semantically divergent final artifact sets."""


class UntrustedArtifactError(FinalizationError):
    """Raised when binary artifact trust/integrity cannot be established."""


@dataclass(frozen=True, slots=True)
class FrozenFinalizationContract:
    """Immutable model, feature, target, partition, and threshold decisions."""

    dataset_slug: str
    model_id: str
    model_family: str
    hyperparameters: tuple[tuple[str, Any], ...]
    random_state: int | None
    feature_columns: tuple[str, ...]
    numerical_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    target_column: str
    target_classes: tuple[Any, ...]
    target_encoding: tuple[tuple[Any, int], ...]
    positive_class: Any
    educational_threshold: float
    threshold_scenario_id: str
    threshold_selection_partition: str
    preprocessing_contract: tuple[tuple[str, Any], ...]
    training_partitions: tuple[str, ...] = ("train", "validation")
    evaluation_partition: str = "test"

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_slug": self.dataset_slug,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "hyperparameters": dict(self.hyperparameters),
            "random_state": self.random_state,
            "feature_columns": list(self.feature_columns),
            "numerical_features": list(self.numerical_features),
            "categorical_features": list(self.categorical_features),
            "identifier_columns": list(self.identifier_columns),
            "target_column": self.target_column,
            "target_classes": list(self.target_classes),
            "target_encoding": dict(self.target_encoding),
            "positive_class": self.positive_class,
            "positive_encoded_label": dict(self.target_encoding)[self.positive_class],
            "educational_threshold": self.educational_threshold,
            "threshold_scenario_id": self.threshold_scenario_id,
            "threshold_selection_partition": self.threshold_selection_partition,
            "preprocessing_contract": dict(self.preprocessing_contract),
            "training_partitions": list(self.training_partitions),
            "evaluation_partition": self.evaluation_partition,
            "operational_threshold": "unresolved",
            "operational_validity": "unconfirmed",
        }


@dataclass(frozen=True, slots=True)
class FinalTrainingData:
    """Defensive final train+validation features and encoded target."""

    _features: pd.DataFrame
    _target: pd.Series
    class_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_features", self._features.copy(deep=True))
        object.__setattr__(self, "_target", self._target.copy(deep=True))

    @property
    def features(self) -> pd.DataFrame:
        return self._features.copy(deep=True)

    @property
    def target(self) -> pd.Series:
        return self._target.copy(deep=True)

    @property
    def row_count(self) -> int:
        return int(len(self._features))


@dataclass(frozen=True, slots=True)
class TestPartitionData:
    """Defensive test features and encoded target loaded after the access gate."""

    _features: pd.DataFrame
    _target: pd.Series
    row_count: int
    class_counts: tuple[tuple[str, int], ...]
    partition_path: str
    partition_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_features", self._features.copy(deep=True))
        object.__setattr__(self, "_target", self._target.copy(deep=True))

    @property
    def features(self) -> pd.DataFrame:
        return self._features.copy(deep=True)

    @property
    def target(self) -> pd.Series:
        return self._target.copy(deep=True)


@dataclass(slots=True)
class EvaluationGuard:
    """Per-execution guard preventing a second test probability evaluation."""

    evaluated: bool = False
    probability_call_count: int = 0


@dataclass(frozen=True, slots=True)
class FinalEvaluation:
    """Aggregate-only final test results derived from one probability vector."""

    probability_metrics: Mapping[str, float]
    threshold_default: Mapping[str, Any]
    threshold_educational: Mapping[str, Any]
    precision_recall_curve: Mapping[str, list[float]]
    roc_curve: Mapping[str, list[float]]
    calibration_curve: Mapping[str, list[float]]
    unknown_categories_report: Mapping[str, list[Any]]
    generalization_deltas: Mapping[str, float]
    probability_sha256: str
    test_probability_evaluation_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return _deepcopy(
            {
                "probability_metrics": self.probability_metrics,
                "threshold_default": self.threshold_default,
                "threshold_educational": self.threshold_educational,
                "precision_recall_curve": self.precision_recall_curve,
                "roc_curve": self.roc_curve,
                "calibration_curve": self.calibration_curve,
                "unknown_categories_report": self.unknown_categories_report,
                "generalization_deltas": self.generalization_deltas,
                "probability_sha256": self.probability_sha256,
                "test_probability_evaluation_count": self.test_probability_evaluation_count,
            }
        )


@dataclass(frozen=True, slots=True)
class SerializedPipeline:
    """Validated staging serialization metadata."""

    path: Path
    byte_sha256: str
    state_fingerprint: str
    descriptor: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    """Outcome of an atomic final-artifact transaction."""

    output_directory: Path
    created: tuple[str, ...]
    replaced: tuple[str, ...]
    idempotent: bool
    byte_sha256: Mapping[str, str]
    semantic_sha256: Mapping[str, str]


# ---------------------------------------------------------------------------
# Deterministic primitives
# ---------------------------------------------------------------------------


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise FinalizationContractError("NaN is not allowed in canonical artifacts.")
        if math.isinf(value):
            return "+Infinity" if value > 0 else "-Infinity"
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_volatile(item)
            for key, item in value.items()
            if str(key) not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return _jsonable(value)


def semantic_fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(_strip_volatile(value)))


def _require_relative_path(value: str | Path, *, field: str) -> str:
    rendered = Path(value).as_posix() if isinstance(value, Path) else str(value)
    pure = PurePosixPath(rendered)
    if pure.is_absolute() or ".." in pure.parts:
        raise FinalizationContractError(f"{field} must be project-relative: {rendered}")
    if len(rendered) >= 3 and rendered[1:3] in {":/", ":\\"}:
        raise FinalizationContractError(f"{field} must not be absolute: {rendered}")
    return pure.as_posix()


def _validate_paths_recursively(value: Any, *, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            if (str(key) == "path" or str(key).endswith("_path")) and item is not None:
                _require_relative_path(item, field=field)
            elif str(key).endswith("_paths") and isinstance(item, Sequence) and not isinstance(item, str):
                for index, path in enumerate(item):
                    _require_relative_path(path, field=f"{field}[{index}]")
            else:
                _validate_paths_recursively(item, prefix=field)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_paths_recursively(item, prefix=f"{prefix}[{index}]")


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


# ---------------------------------------------------------------------------
# Upstream gates and frozen contract
# ---------------------------------------------------------------------------


def validate_upstream_handoff_contracts(
    *,
    project_root: str | Path,
    preparation_paths: Mapping[str, str | Path],
    model_selection_handoff_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Load and fully validate the preparation and model-selection handoffs."""

    required = {
        "preparation_manifest_path",
        "feature_manifest_path",
        "split_manifest_path",
        "quality_evidence_path",
    }
    if set(preparation_paths) != required:
        raise UpstreamHandoffError(
            f"Preparation path keys mismatch: expected={sorted(required)}"
        )
    preparation = load_and_validate_preparation_handoff(
        project_root=project_root,
        **{key: preparation_paths[key] for key in sorted(required)},
    )
    selection = load_and_validate_model_selection_handoff(
        project_root=project_root,
        handoff_path=model_selection_handoff_path,
    )
    validate_finalization_contract(selection)
    manifests = preparation.manifests
    feature_manifest = manifests["feature_manifest"]
    if list(selection["feature_columns"]) != list(feature_manifest["feature_columns"]):
        raise UpstreamHandoffError("Feature order differs between upstream handoffs.")
    if dict(selection["target_encoding"]) != dict(feature_manifest["target_encoding_contract"]):
        raise UpstreamHandoffError("Target encoding differs between upstream handoffs.")
    if selection["positive_class"] != feature_manifest["positive_target_class"]:
        raise UpstreamHandoffError("Positive class differs between upstream handoffs.")
    return preparation, _deepcopy(selection)


def validate_finalization_contract(model_selection_handoff: Mapping[str, Any]) -> None:
    """Validate readiness and absolute non-operational constraints from notebook 03."""

    if model_selection_handoff.get("schema_version") != "model-selection-handoff.v1":
        raise FinalizationContractError("Invalid model-selection handoff schema.")
    if model_selection_handoff.get("artifact_type") != "model_selection_handoff":
        raise FinalizationContractError("Invalid model-selection handoff artifact type.")
    if model_selection_handoff.get("test_partition_sealed") is not True:
        raise FinalizationContractError("Test must be sealed at finalization input.")
    if model_selection_handoff.get("test_partition_evaluated") is not False:
        raise FinalizationContractError("Test must not have been evaluated upstream.")
    readiness = model_selection_handoff.get("readiness", {})
    if readiness.get("final_model_training_ready") is not True:
        raise FinalizationContractError("final_model_training_ready must be true.")
    if model_selection_handoff.get("final_model_trained") is not False:
        raise FinalizationContractError("Final model must not be trained upstream.")
    if model_selection_handoff.get("model_artifact") is not None:
        raise FinalizationContractError("Upstream model artifact must be absent.")
    if model_selection_handoff.get("model_artifact_materialized") is not False:
        raise FinalizationContractError("Upstream model artifact must be unmaterialized.")
    if model_selection_handoff.get("model_bundle_materialized") is not False:
        raise FinalizationContractError("Upstream bundle must be unmaterialized.")
    if model_selection_handoff.get("operational_modeling_ready") is not False:
        raise FinalizationContractError("Operational modeling readiness must remain false.")
    if model_selection_handoff.get("operational_validity") != "unconfirmed":
        raise FinalizationContractError("Operational validity must remain unconfirmed.")
    if model_selection_handoff.get("operational_threshold") != "unresolved":
        raise FinalizationContractError("Operational threshold must remain unresolved.")
    _validate_paths_recursively(model_selection_handoff)


def validate_frozen_model_contract(
    contract: FrozenFinalizationContract,
    *,
    expected_model_id: str | None = None,
) -> None:
    """Validate an immutable finalization decision set."""

    if not contract.dataset_slug or not contract.model_id or not contract.model_family:
        raise FinalizationContractError("Dataset and model identity are required.")
    if expected_model_id is not None and contract.model_id != expected_model_id:
        raise FinalizationContractError("Frozen model differs from selected model.")
    features = list(contract.feature_columns)
    if not features or len(features) != len(set(features)):
        raise FinalizationContractError("Feature columns must be unique and non-empty.")
    if set(contract.numerical_features) & set(contract.categorical_features):
        raise FinalizationContractError("Numerical and categorical roles overlap.")
    if set(contract.numerical_features) | set(contract.categorical_features) != set(features):
        raise FinalizationContractError("Feature roles do not cover the frozen feature set.")
    prohibited = set(contract.identifier_columns) | {contract.target_column}
    if prohibited & set(features):
        raise FinalizationContractError("Identifiers/target cannot enter feature columns.")
    encoding = dict(contract.target_encoding)
    if set(encoding) != set(contract.target_classes) or sorted(encoding.values()) != [0, 1]:
        raise FinalizationContractError("Target encoding must map exactly two classes to 0/1.")
    if contract.positive_class not in encoding or encoding[contract.positive_class] != 1:
        raise FinalizationContractError("Positive class must be encoded as 1.")
    if not 0.0 <= float(contract.educational_threshold) <= 1.0:
        raise FinalizationContractError("Educational threshold must be in [0, 1].")
    if contract.threshold_selection_partition != "validation":
        raise FinalizationContractError("Educational threshold origin must be validation.")


def freeze_finalization_decisions(
    *,
    dataset_slug: str,
    model_selection_handoff: Mapping[str, Any],
    identifier_columns: Sequence[str],
    target_column: str,
    target_classes: Sequence[Any],
    estimator_random_state: int | None,
) -> FrozenFinalizationContract:
    """Freeze all decisions before fit and any evaluative test access."""

    validate_finalization_contract(model_selection_handoff)
    if model_selection_handoff.get("dataset_slug") != dataset_slug:
        raise FinalizationContractError("Dataset slug differs from model-selection handoff.")
    threshold = model_selection_handoff["selected_educational_threshold"]
    contract = FrozenFinalizationContract(
        dataset_slug=str(dataset_slug),
        model_id=str(model_selection_handoff["selected_model_id"]),
        model_family=str(model_selection_handoff["selected_model_family"]),
        hyperparameters=tuple(
            sorted(_deepcopy(model_selection_handoff["selected_hyperparameters"]).items())
        ),
        random_state=estimator_random_state,
        feature_columns=tuple(model_selection_handoff["feature_columns"]),
        numerical_features=tuple(model_selection_handoff["numerical_features"]),
        categorical_features=tuple(model_selection_handoff["categorical_features"]),
        identifier_columns=tuple(identifier_columns),
        target_column=str(target_column),
        target_classes=tuple(target_classes),
        target_encoding=tuple(
            (key, int(value))
            for key, value in model_selection_handoff["target_encoding"].items()
        ),
        positive_class=model_selection_handoff["positive_class"],
        educational_threshold=float(threshold["threshold"]),
        threshold_scenario_id=str(threshold["scenario_id"]),
        threshold_selection_partition="validation",
        preprocessing_contract=tuple(
            sorted(_deepcopy(model_selection_handoff["selected_preprocessing_contract"]).items())
        ),
    )
    validate_frozen_model_contract(contract, expected_model_id=model_selection_handoff["selected_model_id"])
    return contract


# ---------------------------------------------------------------------------
# Partition roles and pipeline
# ---------------------------------------------------------------------------


def validate_final_partition_roles(
    frame: pd.DataFrame,
    *,
    contract: FrozenFinalizationContract,
    partition_name: str,
) -> None:
    """Validate exact columns, target classes, and feature order for one partition."""

    expected = [*contract.identifier_columns, *contract.feature_columns, contract.target_column]
    if list(frame.columns) != expected:
        raise FinalizationContractError(
            f"{partition_name} column order mismatch. expected={expected}, observed={list(frame.columns)}"
        )
    observed = set(frame[contract.target_column].dropna().unique().tolist())
    if observed != set(contract.target_classes):
        raise FinalizationContractError(
            f"{partition_name} target classes mismatch: {sorted(map(str, observed))}"
        )


def assemble_final_training_data(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    contract: FrozenFinalizationContract,
) -> FinalTrainingData:
    """Concatenate train+validation defensively; test is intentionally not accepted."""

    train_copy = train.copy(deep=True)
    validation_copy = validation.copy(deep=True)
    validate_final_partition_roles(train_copy, contract=contract, partition_name="train")
    validate_final_partition_roles(validation_copy, contract=contract, partition_name="validation")
    combined = pd.concat([train_copy, validation_copy], axis=0, ignore_index=True)
    features = combined.loc[:, list(contract.feature_columns)].copy(deep=True)
    encoded = combined[contract.target_column].map(dict(contract.target_encoding))
    if encoded.isna().any():
        raise FinalizationContractError("Target encoding produced missing values.")
    target = encoded.astype("int64")
    counts = tuple(
        (str(key), int(value))
        for key, value in combined[contract.target_column].value_counts().sort_index().items()
    )
    if set(target.unique().tolist()) != {0, 1}:
        raise FinalizationContractError("Final training target must contain encoded classes 0 and 1.")
    return FinalTrainingData(features, target, counts)


def reconstruct_selected_pipeline(
    *,
    estimator: BaseEstimator,
    contract: FrozenFinalizationContract,
) -> Pipeline:
    """Reconstruct the selected unfitted pipeline from a supplied estimator instance."""

    model = clone(estimator)
    if model.__class__.__name__ != contract.model_family:
        raise FinalizationContractError(
            f"Estimator family differs from frozen contract: expected={contract.model_family}, observed={model.__class__.__name__}"
        )
    params = dict(contract.hyperparameters)
    accepted = model.get_params(deep=True)
    model_params: dict[str, Any] = {}
    for key, value in params.items():
        if not key.startswith("model__"):
            raise FinalizationContractError(f"Selected hyperparameter lacks model__ prefix: {key}")
        model_key = key.removeprefix("model__")
        if model_key not in accepted:
            raise FinalizationContractError(f"Estimator does not accept selected parameter: {model_key}")
        model_params[model_key] = value
    if contract.random_state is not None and "random_state" in accepted:
        model_params["random_state"] = contract.random_state
    model.set_params(**model_params)
    pipeline = build_candidate_pipeline(
        estimator=model,
        numerical_features=contract.numerical_features,
        categorical_features=contract.categorical_features,
        scale_numerical=False,
    )
    verify_pipeline_contract(pipeline, contract=contract, require_fitted=False)
    return pipeline


def _is_fitted(pipeline: Pipeline) -> bool:
    try:
        check_is_fitted(pipeline)
        check_is_fitted(pipeline.named_steps["preprocess"])
        check_is_fitted(pipeline.named_steps["model"])
        return True
    except Exception:
        return False


def verify_pipeline_contract(
    pipeline: Pipeline,
    *,
    contract: FrozenFinalizationContract,
    require_fitted: bool,
) -> None:
    """Verify pipeline structure, preprocessing semantics, parameters, and fitted state."""

    if not isinstance(pipeline, Pipeline):
        raise FinalizationContractError("Final model artifact must be an sklearn Pipeline.")
    if list(pipeline.named_steps) != ["preprocess", "model"]:
        raise FinalizationContractError("Pipeline steps must be preprocess then model.")
    preprocess = pipeline.named_steps["preprocess"]
    if not isinstance(preprocess, ColumnTransformer):
        raise FinalizationContractError("Preprocess step must be a ColumnTransformer.")
    if preprocess.remainder != "drop" or float(preprocess.sparse_threshold) != 0.0:
        raise FinalizationContractError("ColumnTransformer must drop remainder and force dense output.")
    transformers = {name: (transformer, list(columns)) for name, transformer, columns in preprocess.transformers}
    if transformers.get("numerical", (None, []))[0] != "passthrough":
        raise FinalizationContractError("Selected family must use numerical passthrough.")
    if transformers.get("numerical", (None, []))[1] != list(contract.numerical_features):
        raise FinalizationContractError("Numerical feature order differs from frozen contract.")
    categorical_transformer, categorical_columns = transformers.get("categorical", (None, []))
    if categorical_columns != list(contract.categorical_features):
        raise FinalizationContractError("Categorical feature order differs from frozen contract.")
    if categorical_transformer is None or categorical_transformer.__class__.__name__ != "OneHotEncoder":
        raise FinalizationContractError("Categorical transformer must be OneHotEncoder.")
    if categorical_transformer.handle_unknown != "ignore" or categorical_transformer.drop is not None:
        raise FinalizationContractError("OneHotEncoder policy differs from frozen contract.")
    if hasattr(categorical_transformer, "sparse_output") and categorical_transformer.sparse_output is not False:
        raise FinalizationContractError("OneHotEncoder output must be dense.")
    if hasattr(categorical_transformer, "sparse") and not hasattr(categorical_transformer, "sparse_output") and categorical_transformer.sparse is not False:
        raise FinalizationContractError("OneHotEncoder output must be dense.")
    model_params = pipeline.named_steps["model"].get_params(deep=False)
    for key, expected in contract.hyperparameters:
        name = key.removeprefix("model__")
        if model_params.get(name) != expected:
            raise FinalizationContractError(f"Model parameter mismatch: {name}")
    if contract.random_state is not None and "random_state" in model_params:
        if model_params["random_state"] != contract.random_state:
            raise FinalizationContractError("Estimator random_state mismatch.")
    fitted = _is_fitted(pipeline)
    if require_fitted and not fitted:
        raise FinalizationContractError("Pipeline must be fitted.")
    if not require_fitted and fitted:
        raise FinalizationContractError("Pipeline must be unfitted before final fit.")


def validate_test_access_gate(
    *,
    contract: FrozenFinalizationContract,
    fitted_pipeline: Pipeline,
    test_path: str | Path,
    expected_sha256: str,
    project_root: str | Path,
) -> Path:
    """Authorize test loading only after decisions are frozen and final fit completed."""

    validate_frozen_model_contract(contract)
    verify_pipeline_contract(fitted_pipeline, contract=contract, require_fitted=True)
    relative = _require_relative_path(test_path, field="test_path")
    root = Path(project_root).resolve()
    absolute = (root / relative).resolve()
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise TestAccessError("Test path escapes project root.") from exc
    if not absolute.is_file():
        raise FileNotFoundError(f"Test partition is missing: {relative}")
    observed = sha256_file(absolute)
    if observed != expected_sha256:
        raise TestAccessError(
            f"Test SHA-256 mismatch: expected={expected_sha256}, observed={observed}"
        )
    return absolute


def load_test_partition_after_freeze(
    *,
    project_root: str | Path,
    test_path: str | Path,
    expected_sha256: str,
    fitted_pipeline: Pipeline,
    contract: FrozenFinalizationContract,
) -> TestPartitionData:
    """Load test directly from its validated path after the final fit gate."""

    absolute = validate_test_access_gate(
        contract=contract,
        fitted_pipeline=fitted_pipeline,
        test_path=test_path,
        expected_sha256=expected_sha256,
        project_root=project_root,
    )
    frame = pd.read_csv(absolute)
    validate_final_partition_roles(frame, contract=contract, partition_name="test")
    features = frame.loc[:, list(contract.feature_columns)].copy(deep=True)
    encoded = frame[contract.target_column].map(dict(contract.target_encoding))
    if encoded.isna().any():
        raise TestAccessError("Test target encoding produced missing values.")
    counts = tuple(
        (str(key), int(value))
        for key, value in frame[contract.target_column].value_counts().sort_index().items()
    )
    return TestPartitionData(
        features,
        encoded.astype("int64"),
        int(len(frame)),
        counts,
        _require_relative_path(test_path, field="test_path"),
        expected_sha256,
    )


# ---------------------------------------------------------------------------
# One-time evaluation and aggregate diagnostics
# ---------------------------------------------------------------------------


def compute_fbeta(y_true: Sequence[int], y_pred: Sequence[int], *, beta: float) -> float:
    return float(fbeta_score(y_true, y_pred, beta=beta, pos_label=1, zero_division=0))


def evaluate_fixed_threshold(
    *, y_true: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, Any]:
    """Evaluate one fixed threshold without selecting or modifying it."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise FinalizationContractError("Threshold must be in [0, 1].")
    probability_series = pd.Series(probabilities, dtype="float64").copy(deep=True)
    labels = (probability_series.to_numpy(copy=True) >= float(threshold)).astype(int)
    true = pd.Series(y_true, dtype="int64").to_numpy(copy=True)
    tn, fp, fn, tp = confusion_matrix(true, labels, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(true, labels, pos_label=1, zero_division=0)),
        "recall": float(recall_score(true, labels, pos_label=1, zero_division=0)),
        "f1": float(f1_score(true, labels, pos_label=1, zero_division=0)),
        "f2": compute_fbeta(true, labels, beta=2.0),
        "balanced_accuracy": float(balanced_accuracy_score(true, labels)),
        "accuracy_contextual": float(accuracy_score(true, labels)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "predicted_positive_count": int(labels.sum()),
        "predicted_positive_rate": float(labels.mean()),
    }


def compute_probability_metrics(
    *, y_true: Sequence[int], probabilities: Sequence[float]
) -> dict[str, float]:
    true = pd.Series(y_true, dtype="int64").to_numpy(copy=True)
    prob = pd.Series(probabilities, dtype="float64").to_numpy(copy=True)
    return {
        "average_precision": float(average_precision_score(true, prob)),
        "roc_auc": float(roc_auc_score(true, prob)),
        "log_loss": float(log_loss(true, prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(true, prob)),
    }


def compute_generalization_deltas(
    *,
    validation_metrics: Mapping[str, Any],
    test_probability_metrics: Mapping[str, float],
    test_default_threshold: Mapping[str, Any],
    test_educational_threshold: Mapping[str, Any],
    validation_educational_threshold: Mapping[str, Any],
) -> dict[str, float]:
    """Compute descriptive test-minus-validation deltas; never drive decisions."""

    return {
        "average_precision_test_minus_validation": float(test_probability_metrics["average_precision"] - validation_metrics["average_precision"]),
        "roc_auc_test_minus_validation": float(test_probability_metrics["roc_auc"] - validation_metrics["roc_auc"]),
        "brier_score_test_minus_validation": float(test_probability_metrics["brier_score"] - validation_metrics["brier_score"]),
        "log_loss_test_minus_validation": float(test_probability_metrics["log_loss"] - validation_metrics["log_loss"]),
        "default_precision_test_minus_validation": float(test_default_threshold["precision"] - validation_metrics["precision"]),
        "default_recall_test_minus_validation": float(test_default_threshold["recall"] - validation_metrics["recall"]),
        "educational_precision_test_minus_validation": float(test_educational_threshold["precision"] - validation_educational_threshold["precision"]),
        "educational_recall_test_minus_validation": float(test_educational_threshold["recall"] - validation_educational_threshold["recall"]),
    }


def report_unknown_categories(
    *, fitted_pipeline: Pipeline, features: pd.DataFrame, categorical_features: Sequence[str]
) -> dict[str, list[Any]]:
    """Report categories in an evaluation frame absent from fitted vocabularies."""

    preprocess = fitted_pipeline.named_steps["preprocess"]
    categorical = preprocess.named_transformers_["categorical"]
    report: dict[str, list[Any]] = {}
    for column, fitted_values in zip(categorical_features, categorical.categories_, strict=True):
        known = set(_jsonable(list(fitted_values)))
        observed = set(_jsonable(features[column].dropna().unique().tolist()))
        unknown = sorted(observed - known, key=lambda value: str(value))
        if unknown:
            report[str(column)] = unknown
    return report


def _positive_probability_index(pipeline: Pipeline) -> int:
    classes = list(_jsonable(pipeline.named_steps["model"].classes_))
    if 1 not in classes:
        raise FinalizationContractError("Fitted estimator does not expose encoded positive class 1.")
    return classes.index(1)


def evaluate_final_model_once(
    *,
    fitted_pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    educational_threshold: float,
    educational_recall_target: float,
    validation_metrics: Mapping[str, Any],
    validation_educational_threshold: Mapping[str, Any],
    categorical_features: Sequence[str],
    guard: EvaluationGuard,
) -> FinalEvaluation:
    """Call predict_proba exactly once and derive all final aggregate results."""

    if guard.evaluated:
        raise DuplicateTestEvaluationError("Test probability evaluation already occurred.")
    if not _is_fitted(fitted_pipeline):
        raise TestAccessError("Final pipeline must be fitted before test evaluation.")
    features = x_test.copy(deep=True)
    target = y_test.copy(deep=True)
    matrix = fitted_pipeline.predict_proba(features)  # exactly one test call
    guard.probability_call_count += 1
    if guard.probability_call_count != 1:
        raise DuplicateTestEvaluationError("predict_proba call count exceeded one.")
    probabilities = matrix[:, _positive_probability_index(fitted_pipeline)].copy()
    guard.evaluated = True

    probability_metrics = compute_probability_metrics(y_true=target, probabilities=probabilities)
    default = evaluate_fixed_threshold(y_true=target, probabilities=probabilities, threshold=0.5)
    educational = evaluate_fixed_threshold(
        y_true=target, probabilities=probabilities, threshold=educational_threshold
    )
    educational["educational_recall_target"] = float(educational_recall_target)
    educational["educational_recall_target_satisfied"] = bool(
        educational["recall"] >= educational_recall_target
    )

    precision, recall, pr_thresholds = precision_recall_curve(target, probabilities, pos_label=1)
    fpr, tpr, roc_thresholds = roc_curve(target, probabilities, pos_label=1)
    fraction_positive, mean_predicted = calibration_curve(
        target, probabilities, n_bins=10, strategy="quantile"
    )
    deltas = compute_generalization_deltas(
        validation_metrics=validation_metrics,
        test_probability_metrics=probability_metrics,
        test_default_threshold=default,
        test_educational_threshold=educational,
        validation_educational_threshold=validation_educational_threshold,
    )
    return FinalEvaluation(
        probability_metrics=probability_metrics,
        threshold_default=default,
        threshold_educational=educational,
        precision_recall_curve={
            "precision": _jsonable(precision),
            "recall": _jsonable(recall),
            "thresholds": _jsonable(pr_thresholds),
        },
        roc_curve={
            "false_positive_rate": _jsonable(fpr),
            "true_positive_rate": _jsonable(tpr),
            "thresholds": _jsonable(roc_thresholds),
        },
        calibration_curve={
            "mean_predicted_probability": _jsonable(mean_predicted),
            "fraction_of_positives": _jsonable(fraction_positive),
            "n_bins": 10,
            "strategy": "quantile",
        },
        unknown_categories_report=report_unknown_categories(
            fitted_pipeline=fitted_pipeline,
            features=features,
            categorical_features=categorical_features,
        ),
        generalization_deltas=deltas,
        probability_sha256=sha256_bytes(pd.Series(probabilities).to_csv(index=False, header=False, lineterminator="\n").encode("utf-8")),
    )


# ---------------------------------------------------------------------------
# Fitted-state descriptor and serialization
# ---------------------------------------------------------------------------


def describe_fitted_pipeline(
    *,
    pipeline: Pipeline,
    contract: FrozenFinalizationContract,
    training_data: FinalTrainingData,
    train_sha256: str,
    validation_sha256: str,
    sample_size: int = 32,
) -> dict[str, Any]:
    """Build a canonical JSON descriptor of fitted pipeline state."""

    verify_pipeline_contract(pipeline, contract=contract, require_fitted=True)
    preprocess = pipeline.named_steps["preprocess"]
    categorical = preprocess.named_transformers_["categorical"]
    names = preprocess.get_feature_names_out().tolist()
    sample = training_data.features.iloc[: min(sample_size, training_data.row_count)].copy(deep=True)
    probabilities = pipeline.predict_proba(sample)[:, _positive_probability_index(pipeline)].copy()
    descriptor = {
        "pipeline_class": f"{pipeline.__class__.__module__}.{pipeline.__class__.__name__}",
        "steps": list(pipeline.named_steps),
        "model_class": f"{pipeline.named_steps['model'].__class__.__module__}.{pipeline.named_steps['model'].__class__.__name__}",
        "model_parameters": {
            key: pipeline.named_steps["model"].get_params(deep=False).get(key.removeprefix("model__"))
            for key, _ in contract.hyperparameters
        },
        "random_state": contract.random_state,
        "feature_order": list(contract.feature_columns),
        "categorical_vocabularies": {
            column: _jsonable(values)
            for column, values in zip(contract.categorical_features, categorical.categories_, strict=True)
        },
        "transformed_feature_names": names,
        "estimator_classes": _jsonable(pipeline.named_steps["model"].classes_),
        "transformed_feature_count": len(names),
        "training_partition_sha256": {
            "train": train_sha256,
            "validation": validation_sha256,
        },
        "training_row_count": training_data.row_count,
        "training_target_class_counts": dict(training_data.class_counts),
        "sample_probability_sha256": sha256_bytes(
            pd.Series(probabilities).to_csv(index=False, header=False, lineterminator="\n").encode("utf-8")
        ),
        "sample_size": int(len(sample)),
        "runtime_major_minor": {
            key: ".".join(value.split(".")[:2])
            for key, value in runtime_versions().items()
        },
    }
    return _jsonable(descriptor)


def compute_fitted_model_fingerprint(descriptor: Mapping[str, Any]) -> str:
    return semantic_fingerprint(descriptor)


def serialize_pipeline_to_staging(
    *, pipeline: Pipeline, staging_path: str | Path
) -> str:
    """Serialize a complete fitted Pipeline and return its byte SHA-256."""

    if not isinstance(pipeline, Pipeline) or not _is_fitted(pipeline):
        raise SerializationValidationError("Only a fitted sklearn Pipeline can be serialized.")
    path = Path(staging_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return sha256_file(path)


def validate_serialized_pipeline(
    *,
    staging_path: str | Path,
    expected_sha256: str,
    contract: FrozenFinalizationContract,
    reference_pipeline: Pipeline,
    validation_sample: pd.DataFrame,
) -> Pipeline:
    """Reload and verify a trusted staging joblib without using test data."""

    path = Path(staging_path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise SerializationValidationError("Serialized pipeline SHA-256 mismatch.")
    loaded = joblib.load(path)
    verify_pipeline_contract(loaded, contract=contract, require_fitted=True)
    classes = list(_jsonable(loaded.named_steps["model"].classes_))
    if classes != [0, 1]:
        raise SerializationValidationError(f"Unexpected estimator classes: {classes}")
    sample = validation_sample.copy(deep=True)
    expected = reference_pipeline.predict_proba(sample)
    observed_probabilities = loaded.predict_proba(sample)
    if expected.shape != observed_probabilities.shape or not pd.DataFrame(expected).equals(pd.DataFrame(observed_probabilities)):
        # Exact equality is expected for same-runtime joblib round-trip.
        import numpy as np

        if not np.allclose(expected, observed_probabilities, rtol=0.0, atol=0.0):
            raise SerializationValidationError("Round-trip probabilities differ.")
    return loaded


# ---------------------------------------------------------------------------
# Artifact builders
# ---------------------------------------------------------------------------


def build_final_test_evidence(
    *,
    contract: FrozenFinalizationContract,
    test_partition: TestPartitionData,
    evaluation: FinalEvaluation,
    validation_metrics: Mapping[str, Any],
    validation_educational_threshold: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "final-test-evidence.v1",
        "artifact_type": "final_test_evidence",
        "dataset_slug": contract.dataset_slug,
        "partition": "test",
        "partition_path": test_partition.partition_path,
        "partition_sha256": test_partition.partition_sha256,
        "row_count": test_partition.row_count,
        "class_counts": dict(test_partition.class_counts),
        "test_loaded_only_after_final_fit": True,
        "test_probability_evaluation_count": 1,
        "positive_class": contract.positive_class,
        "encoded_positive_label": 1,
        "probability_metrics": _deepcopy(evaluation.probability_metrics),
        "threshold_0_50": _deepcopy(evaluation.threshold_default),
        "educational_threshold": _deepcopy(evaluation.threshold_educational),
        "precision_recall_curve": _deepcopy(evaluation.precision_recall_curve),
        "roc_curve": _deepcopy(evaluation.roc_curve),
        "calibration_curve": _deepcopy(evaluation.calibration_curve),
        "unknown_categories_report": _deepcopy(evaluation.unknown_categories_report),
        "selected_validation_metrics": _deepcopy(validation_metrics),
        "selected_validation_educational_threshold": _deepcopy(validation_educational_threshold),
        "generalization_deltas": _deepcopy(evaluation.generalization_deltas),
        "frozen_model_contract": {
            "dataset_slug": contract.dataset_slug,
            "model_id": contract.model_id,
            "model_family": contract.model_family,
            "hyperparameters": dict(contract.hyperparameters),
            "feature_columns": list(contract.feature_columns),
            "numerical_features": list(contract.numerical_features),
            "categorical_features": list(contract.categorical_features),
            "target_column_metadata_only": contract.target_column,
            "target_encoding": dict(contract.target_encoding),
            "positive_class": contract.positive_class,
            "operational_validity": "unconfirmed",
            "operational_threshold": "unresolved",
        },
        "frozen_threshold_contract": {
            "threshold": contract.educational_threshold,
            "scenario_id": contract.threshold_scenario_id,
            "origin": "validation",
            "purpose": "educational",
        },
        "probability_vector_sha256_aggregate_only": evaluation.probability_sha256,
        "no_post_test_adjustment": True,
        "test_used_for_model_selection": False,
        "test_used_for_hyperparameter_selection": False,
        "test_used_for_threshold_selection": False,
        "test_used_for_feature_selection": False,
        "test_used_for_preprocessing_selection": False,
        "individual_rows_persisted": False,
        "operational_validity": "unconfirmed",
    }


def _fitted_vocabularies(pipeline: Pipeline, contract: FrozenFinalizationContract) -> dict[str, list[Any]]:
    encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    return {
        column: _jsonable(values)
        for column, values in zip(contract.categorical_features, encoder.categories_, strict=True)
    }


def build_inference_bundle(
    *,
    contract: FrozenFinalizationContract,
    fitted_pipeline: Pipeline,
    model_artifact_path: str,
    model_artifact_sha256: str,
    model_state_fingerprint: str,
    expected_input_dtypes: Mapping[str, str],
    missing_value_policy: Mapping[str, Any],
) -> dict[str, Any]:
    verify_pipeline_contract(fitted_pipeline, contract=contract, require_fitted=True)
    preprocess = fitted_pipeline.named_steps["preprocess"]
    negative_class = next(value for value in contract.target_classes if value != contract.positive_class)
    return {
        "schema_version": "inference-bundle.v1",
        "artifact_type": "inference_bundle",
        "dataset_slug": contract.dataset_slug,
        "bundle_version": "1.0.0",
        "model_artifact_path": _require_relative_path(model_artifact_path, field="model_artifact_path"),
        "model_artifact_format": "joblib",
        "model_artifact_sha256": model_artifact_sha256,
        "model_state_fingerprint": model_state_fingerprint,
        "model_id": contract.model_id,
        "model_family": contract.model_family,
        "selected_hyperparameters": dict(contract.hyperparameters),
        "estimator_random_state": contract.random_state,
        "pipeline_class": f"{fitted_pipeline.__class__.__module__}.{fitted_pipeline.__class__.__name__}",
        "preprocessing_embedded": True,
        "feature_columns": list(contract.feature_columns),
        "numerical_features": list(contract.numerical_features),
        "categorical_features": list(contract.categorical_features),
        "identifier_columns_excluded_from_model": list(contract.identifier_columns),
        "target_column_metadata_only": contract.target_column,
        "target_classes": list(contract.target_classes),
        "target_encoding": dict(contract.target_encoding),
        "positive_class": contract.positive_class,
        "positive_encoded_label": 1,
        "negative_class": negative_class,
        "categorical_strategy": "one_hot",
        "unknown_category_policy": "ignore_and_report",
        "drop_category": None,
        "numerical_scaling": "none",
        "fitted_categorical_vocabularies": _fitted_vocabularies(fitted_pipeline, contract),
        "transformed_feature_names": preprocess.get_feature_names_out().tolist(),
        "expected_input_dtypes": _deepcopy(expected_input_dtypes),
        "required_input_columns": list(contract.feature_columns),
        "prohibited_input_columns": [*contract.identifier_columns, contract.target_column],
        "missing_value_policy": _deepcopy(missing_value_policy),
        "educational_decision_threshold": contract.educational_threshold,
        "threshold_scenario": contract.threshold_scenario_id,
        "threshold_selection_partition": "validation",
        "operational_threshold": "unresolved",
        "output_contract": {
            "positive_class_probability": "float in [0,1]",
            "educational_prediction_encoded": "integer 0 or 1",
            "educational_prediction_label": list(contract.target_classes),
            "educational_threshold": contract.educational_threshold,
            "operational_prediction_available": False,
        },
        "runtime_version_requirements": runtime_versions(),
        "security_note": "Load joblib only from a trusted source after verifying its SHA-256.",
        "limitations": [
            "Educational snapshot evaluation only; operational validity is unconfirmed.",
            "The educational threshold is not a production decision policy.",
            "Temporal contract and feature inference availability remain unresolved.",
        ],
        "readiness": {
            "educational_inference_demo_ready": True,
            "model_artifact_materialized": True,
            "model_bundle_materialized": True,
            "operational_modeling_ready": False,
        },
        "operational_validity": "unconfirmed",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
    }


def build_final_model_manifest(
    *,
    contract: FrozenFinalizationContract,
    upstream_references: Mapping[str, Any],
    training_data: FinalTrainingData,
    test_partition: TestPartitionData,
    fit_duration_seconds: float,
    model_artifact_path: str,
    model_artifact_sha256: str,
    model_state_fingerprint: str,
    fitted_state_descriptor: Mapping[str, Any],
    final_artifact_paths: Mapping[str, str],
    final_artifact_fingerprints: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": "final-model-manifest.v1",
        "artifact_type": "final_model_manifest",
        "dataset_slug": contract.dataset_slug,
        "upstream_references": _deepcopy(upstream_references),
        "selected_model_id": contract.model_id,
        "selected_model_family": contract.model_family,
        "selected_hyperparameters": dict(contract.hyperparameters),
        "estimator_random_state": contract.random_state,
        "preprocessing_contract": dict(contract.preprocessing_contract),
        "feature_columns": list(contract.feature_columns),
        "numerical_features": list(contract.numerical_features),
        "categorical_features": list(contract.categorical_features),
        "identifier_columns": list(contract.identifier_columns),
        "target_column": contract.target_column,
        "target_classes": list(contract.target_classes),
        "target_encoding": dict(contract.target_encoding),
        "positive_class": contract.positive_class,
        "training_partitions": list(contract.training_partitions),
        "final_training_row_count": training_data.row_count,
        "final_training_class_counts": dict(training_data.class_counts),
        "test_row_count": test_partition.row_count,
        "test_class_counts": dict(test_partition.class_counts),
        "educational_threshold": contract.educational_threshold,
        "threshold_scenario_id": contract.threshold_scenario_id,
        "threshold_origin": "validation",
        "threshold_purpose": "educational",
        "test_access_policy": {
            "loaded_after_final_fit": True,
            "test_evaluation_count": 1,
            "used_for_adjustment": False,
        },
        "model_artifact_path": _require_relative_path(model_artifact_path, field="model_artifact_path"),
        "artifact_format": "joblib",
        "model_artifact_byte_sha256": model_artifact_sha256,
        "fitted_state_semantic_fingerprint": model_state_fingerprint,
        "fitted_state_descriptor": _deepcopy(fitted_state_descriptor),
        "fit_duration_seconds": float(fit_duration_seconds),
        "runtime_versions": runtime_versions(),
        "final_artifact_paths": _deepcopy(final_artifact_paths),
        "final_artifact_fingerprints": _deepcopy(final_artifact_fingerprints),
        "readiness": {
            "educational_final_model_completed": True,
            "final_model_trained": True,
            "final_test_evaluation_completed": True,
            "model_artifact_materialized": True,
            "model_bundle_materialized": True,
            "final_model_handoff_ready": True,
            "educational_inference_demo_ready": True,
            "operational_modeling_ready": False,
        },
        "limitations": [
            "Educational snapshot validation only.",
            "Operational validity is unconfirmed.",
            "Operational threshold remains unresolved.",
        ],
        "operational_validity": "unconfirmed",
        "operational_threshold": "unresolved",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
    }


def build_final_model_handoff(
    *,
    contract: FrozenFinalizationContract,
    preparation_handoff_references: Mapping[str, Any],
    model_selection_handoff_references: Mapping[str, Any],
    final_references: Mapping[str, Mapping[str, str]],
    evaluation: FinalEvaluation,
) -> dict[str, Any]:
    return {
        "schema_version": "final-model-handoff.v1",
        "artifact_type": "final_model_handoff",
        "dataset_slug": contract.dataset_slug,
        "preparation_handoff_references": _deepcopy(preparation_handoff_references),
        "model_selection_handoff_references": _deepcopy(model_selection_handoff_references),
        "final_references": _deepcopy(final_references),
        "model_state_fingerprint": final_references["model_artifact"]["semantic_sha256"],
        "selected_model_id": contract.model_id,
        "selected_model_family": contract.model_family,
        "selected_hyperparameters": dict(contract.hyperparameters),
        "preprocessing": dict(contract.preprocessing_contract),
        "feature_order": list(contract.feature_columns),
        "target_encoding": dict(contract.target_encoding),
        "positive_class": contract.positive_class,
        "educational_threshold": contract.educational_threshold,
        "educational_threshold_scenario": contract.threshold_scenario_id,
        "final_training_partitions": list(contract.training_partitions),
        "final_evaluation_partition": "test",
        "final_test_metrics": evaluation.as_dict(),
        "notebook_05_instructions": [
            "Validate this handoff and the inference bundle.",
            "Verify the model artifact SHA-256 before trusted loading.",
            "Load only the complete fitted pipeline; do not refit it.",
            "Do not use the test partition for the demonstration.",
            "Use independent inputs in the declared feature order.",
            "Apply the embedded preprocessing through the pipeline.",
            "Generate positive-class probabilities.",
            "Apply the educational threshold only as a demonstration.",
            "State that the educational threshold is not operational.",
        ],
        "educational_final_model_completed": True,
        "final_model_trained": True,
        "model_artifact_materialized": True,
        "model_bundle_materialized": True,
        "final_test_evaluation_completed": True,
        "final_model_handoff_ready": True,
        "educational_inference_demo_ready": True,
        "test_partition_sealed_at_input": True,
        "test_partition_evaluated": True,
        "test_partition_evaluation_count": 1,
        "test_partition_used_for_adjustment": False,
        "test_partition_used_for_model_selection": False,
        "test_partition_used_for_threshold_selection": False,
        "api_implemented": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "operational_threshold": "unresolved",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
    }


# ---------------------------------------------------------------------------
# Validation, persistence, idempotence, and trusted loading
# ---------------------------------------------------------------------------


def _validate_json_artifact(filename: str, payload: Mapping[str, Any]) -> None:
    expected_schema, expected_type = SCHEMAS[filename]
    if payload.get("schema_version") != expected_schema:
        raise FinalizationContractError(f"Invalid schema for {filename}.")
    if payload.get("artifact_type") != expected_type:
        raise FinalizationContractError(f"Invalid artifact type for {filename}.")
    if payload.get("operational_validity") == "confirmed":
        raise FinalizationContractError("Operational validity cannot be confirmed here.")
    if "operational_threshold" in payload and payload.get("operational_threshold") != "unresolved":
        raise FinalizationContractError("Operational threshold must remain unresolved.")
    _validate_paths_recursively(payload)
    rendered = canonical_json_bytes(payload)
    for prohibited in (b"individual_predictions", b"row_predictions", b"individual_probabilities"):
        if prohibited in rendered and filename == "final-test-evidence.json":
            raise FinalizationContractError("Final test evidence contains row-level prediction data.")


def inspect_final_artifact_set(output_directory: str | Path) -> str:
    output = Path(output_directory)
    present = [(output / filename).is_file() for filename in FINAL_ARTIFACT_FILENAMES]
    if not any(present):
        return "absent"
    if all(present):
        return "complete"
    return "partial"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FinalizationContractError(f"JSON artifact must be an object: {path.name}")
    return payload


def _validate_complete_set(
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate schemas, hashes, cross-references, and model identity for a complete set."""

    state = inspect_final_artifact_set(output)
    if state != "complete":
        raise ArtifactConflictError(f"Final artifact set is {state}, not complete.")
    payloads: dict[str, dict[str, Any]] = {}
    for filename in SCHEMAS:
        payload = _load_json(output / filename)
        _validate_json_artifact(filename, payload)
        payloads[filename] = payload

    handoff = payloads["final-model-handoff.json"]
    bundle = payloads["inference-bundle.json"]
    manifest = payloads["final-model-manifest.json"]
    evidence = payloads["final-test-evidence.json"]
    slugs = {
        handoff.get("dataset_slug"),
        bundle.get("dataset_slug"),
        manifest.get("dataset_slug"),
        evidence.get("dataset_slug"),
    }
    if len(slugs) != 1:
        raise ArtifactConflictError("Dataset slug differs within the final artifact set.")

    model_relative = PurePosixPath(
        _require_relative_path(bundle["model_artifact_path"], field="model_artifact_path")
    )
    expected_model_path = (output / "final-pipeline.joblib").resolve()
    inferred_root = expected_model_path
    for _ in model_relative.parts:
        inferred_root = inferred_root.parent
    if (inferred_root / model_relative).resolve() != expected_model_path:
        raise ArtifactConflictError("Bundle model path does not identify the complete-set joblib.")
    model_hash = sha256_file(expected_model_path)
    if model_hash != bundle.get("model_artifact_sha256"):
        raise ArtifactConflictError("Existing joblib hash differs from the inference bundle.")

    references = handoff.get("final_references", {})
    for name, reference in references.items():
        relative = PurePosixPath(
            _require_relative_path(reference["path"], field=f"final_references.{name}.path")
        )
        absolute = (inferred_root / relative).resolve()
        try:
            absolute.relative_to(inferred_root)
        except ValueError as exc:
            raise ArtifactConflictError("Existing final reference escapes project root.") from exc
        if not absolute.is_file():
            raise ArtifactConflictError(f"Existing final reference is missing: {relative}")
        if sha256_file(absolute) != reference.get("byte_sha256"):
            raise ArtifactConflictError(f"Existing final reference hash mismatch: {relative}")

    manifest_fingerprints = manifest.get("final_artifact_fingerprints", {})
    expected_fingerprint_inputs = {
        "final-pipeline.joblib": (model_hash, bundle.get("model_state_fingerprint")),
        "final-test-evidence.json": (
            sha256_file(output / "final-test-evidence.json"),
            semantic_fingerprint(evidence),
        ),
        "inference-bundle.json": (
            sha256_file(output / "inference-bundle.json"),
            semantic_fingerprint(bundle),
        ),
    }
    for filename, (byte_hash, semantic_hash) in expected_fingerprint_inputs.items():
        declared = manifest_fingerprints.get(filename)
        if not isinstance(declared, Mapping):
            raise ArtifactConflictError(f"Manifest fingerprint missing: {filename}")
        if declared.get("byte_sha256") != byte_hash or declared.get("semantic_sha256") != semantic_hash:
            raise ArtifactConflictError(f"Manifest fingerprint mismatch: {filename}")
    if bundle.get("model_state_fingerprint") != handoff.get("model_state_fingerprint"):
        raise ArtifactConflictError("Existing model-state fingerprints are inconsistent.")
    return handoff, bundle, manifest, evidence

def validate_existing_finalization_equivalence(
    *, output_directory: str | Path, contract: FrozenFinalizationContract
) -> bool:
    """Validate complete existing artifacts against the frozen upstream contract."""

    output = Path(output_directory)
    handoff, bundle, _, _ = _validate_complete_set(output)
    checks = {
        "dataset_slug": contract.dataset_slug,
        "selected_model_id": contract.model_id,
        "selected_model_family": contract.model_family,
        "educational_threshold": contract.educational_threshold,
    }
    for key, expected in checks.items():
        observed = handoff.get(key)
        if observed != expected:
            raise ArtifactConflictError(
                f"Existing final artifact differs at {key}: expected={expected!r}, observed={observed!r}"
            )
    if handoff.get("feature_order") != list(contract.feature_columns):
        raise ArtifactConflictError("Existing final artifact feature order is divergent.")
    if handoff.get("selected_hyperparameters") != dict(contract.hyperparameters):
        raise ArtifactConflictError("Existing final hyperparameters are divergent.")
    if bundle.get("model_state_fingerprint") != handoff.get("model_state_fingerprint"):
        raise ArtifactConflictError("Existing model-state fingerprints are inconsistent.")
    if handoff.get("test_partition_evaluation_count") != 1:
        raise ArtifactConflictError("Existing test evaluation count must remain one.")
    return True


def write_final_model_artifacts(
    *,
    project_root: str | Path,
    output_directory: str | Path,
    pipeline: Pipeline,
    contract: FrozenFinalizationContract,
    training_data: FinalTrainingData,
    train_sha256: str,
    validation_sha256: str,
    test_partition: TestPartitionData,
    evaluation: FinalEvaluation,
    fit_duration_seconds: float,
    upstream_references: Mapping[str, Any],
    preparation_handoff_references: Mapping[str, Any],
    model_selection_handoff_references: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    validation_educational_threshold: Mapping[str, Any],
    expected_input_dtypes: Mapping[str, str],
    missing_value_policy: Mapping[str, Any],
    overwrite: bool = False,
) -> ArtifactWriteResult:
    """Stage, validate, and atomically promote the five final artifacts."""

    root = Path(project_root).resolve()
    relative_output = _require_relative_path(output_directory, field="output_directory")
    output = (root / relative_output).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise FinalizationContractError("Output directory escapes project root.") from exc
    state = inspect_final_artifact_set(output)
    if state == "partial":
        raise ArtifactConflictError("Partial final artifact set detected; refusing repair.")
    if state == "complete" and not overwrite:
        validate_existing_finalization_equivalence(output_directory=output, contract=contract)
        byte_hashes = {name: sha256_file(output / name) for name in FINAL_ARTIFACT_FILENAMES}
        semantic_hashes = {
            name: semantic_fingerprint(_load_json(output / name))
            for name in SCHEMAS
        }
        semantic_hashes["final-pipeline.joblib"] = _load_json(
            output / "inference-bundle.json"
        )["model_state_fingerprint"]
        return ArtifactWriteResult(output, (), (), True, byte_hashes, semantic_hashes)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".final-model-staging-", dir=output.parent))
    backup_root = Path(tempfile.mkdtemp(prefix=".final-model-backup-", dir=output.parent))
    staging = staging_root / output.name
    staging.mkdir(parents=True, exist_ok=True)
    promoted: list[str] = []
    backed_up: list[str] = []
    existing = {name: (output / name).is_file() for name in FINAL_ARTIFACT_FILENAMES}
    try:
        descriptor = describe_fitted_pipeline(
            pipeline=pipeline,
            contract=contract,
            training_data=training_data,
            train_sha256=train_sha256,
            validation_sha256=validation_sha256,
        )
        state_fp = compute_fitted_model_fingerprint(descriptor)
        model_staging = staging / "final-pipeline.joblib"
        model_hash = serialize_pipeline_to_staging(pipeline=pipeline, staging_path=model_staging)
        sample = training_data.features.iloc[: min(32, training_data.row_count)]
        validate_serialized_pipeline(
            staging_path=model_staging,
            expected_sha256=model_hash,
            contract=contract,
            reference_pipeline=pipeline,
            validation_sample=sample,
        )
        model_rel = f"{relative_output}/final-pipeline.joblib"
        evidence = build_final_test_evidence(
            contract=contract,
            test_partition=test_partition,
            evaluation=evaluation,
            validation_metrics=validation_metrics,
            validation_educational_threshold=validation_educational_threshold,
        )
        bundle = build_inference_bundle(
            contract=contract,
            fitted_pipeline=pipeline,
            model_artifact_path=model_rel,
            model_artifact_sha256=model_hash,
            model_state_fingerprint=state_fp,
            expected_input_dtypes=expected_input_dtypes,
            missing_value_policy=missing_value_policy,
        )
        preliminary = {
            "final-test-evidence.json": evidence,
            "inference-bundle.json": bundle,
        }
        fingerprints: dict[str, dict[str, str]] = {
            "final-pipeline.joblib": {
                "byte_sha256": model_hash,
                "semantic_sha256": state_fp,
            }
        }
        for filename, payload in preliminary.items():
            _validate_json_artifact(filename, payload)
            content = canonical_json_text(payload).encode("utf-8")
            (staging / filename).write_bytes(content)
            fingerprints[filename] = {
                "byte_sha256": sha256_bytes(content),
                "semantic_sha256": semantic_fingerprint(payload),
            }
        paths = {name: f"{relative_output}/{name}" for name in FINAL_ARTIFACT_FILENAMES}
        manifest = build_final_model_manifest(
            contract=contract,
            upstream_references=upstream_references,
            training_data=training_data,
            test_partition=test_partition,
            fit_duration_seconds=fit_duration_seconds,
            model_artifact_path=model_rel,
            model_artifact_sha256=model_hash,
            model_state_fingerprint=state_fp,
            fitted_state_descriptor=descriptor,
            final_artifact_paths=paths,
            final_artifact_fingerprints=fingerprints,
        )
        _validate_json_artifact("final-model-manifest.json", manifest)
        manifest_content = canonical_json_text(manifest).encode("utf-8")
        (staging / "final-model-manifest.json").write_bytes(manifest_content)
        fingerprints["final-model-manifest.json"] = {
            "byte_sha256": sha256_bytes(manifest_content),
            "semantic_sha256": semantic_fingerprint(manifest),
        }
        final_refs = {
            "model_artifact": {
                "path": model_rel,
                **fingerprints["final-pipeline.joblib"],
            },
            "final_model_manifest": {
                "path": paths["final-model-manifest.json"],
                **fingerprints["final-model-manifest.json"],
            },
            "final_test_evidence": {
                "path": paths["final-test-evidence.json"],
                **fingerprints["final-test-evidence.json"],
            },
            "inference_bundle": {
                "path": paths["inference-bundle.json"],
                **fingerprints["inference-bundle.json"],
            },
        }
        handoff = build_final_model_handoff(
            contract=contract,
            preparation_handoff_references=preparation_handoff_references,
            model_selection_handoff_references=model_selection_handoff_references,
            final_references=final_refs,
            evaluation=evaluation,
        )
        _validate_json_artifact("final-model-handoff.json", handoff)
        handoff_content = canonical_json_text(handoff).encode("utf-8")
        (staging / "final-model-handoff.json").write_bytes(handoff_content)
        fingerprints["final-model-handoff.json"] = {
            "byte_sha256": sha256_bytes(handoff_content),
            "semantic_sha256": semantic_fingerprint(handoff),
        }
        # Validate every staged artifact and cross-reference before promotion.
        for filename in SCHEMAS:
            _validate_json_artifact(filename, _load_json(staging / filename))
        if sha256_file(model_staging) != bundle["model_artifact_sha256"]:
            raise SerializationValidationError("Bundle/model staging hash mismatch.")

        if state == "complete" and overwrite:
            # Divergence was explicitly authorized; preserve rollback copies.
            pass
        output.mkdir(parents=True, exist_ok=True)
        for filename in FINAL_ARTIFACT_FILENAMES:
            destination = output / filename
            if destination.exists():
                backup = backup_root / filename
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backed_up.append(filename)
            os.replace(staging / filename, destination)
            promoted.append(filename)

        # Complete-set validation after promotion.
        for filename in SCHEMAS:
            _validate_json_artifact(filename, _load_json(output / filename))
        loaded_bundle = load_and_validate_inference_bundle(
            project_root=root,
            bundle_path=paths["inference-bundle.json"],
        )
        load_trusted_pipeline_from_bundle(project_root=root, bundle=loaded_bundle)
        load_and_validate_final_model_handoff(
            project_root=root,
            handoff_path=paths["final-model-handoff.json"],
        )
        byte_hashes = {name: sha256_file(output / name) for name in FINAL_ARTIFACT_FILENAMES}
        semantic_hashes = {**{name: values["semantic_sha256"] for name, values in fingerprints.items()}}
        return ArtifactWriteResult(
            output,
            tuple(name for name in promoted if not existing[name]),
            tuple(name for name in promoted if existing[name]),
            False,
            byte_hashes,
            semantic_hashes,
        )
    except Exception:
        for filename in reversed(promoted):
            destination = output / filename
            if destination.exists():
                destination.unlink()
        for filename in reversed(backed_up):
            backup = backup_root / filename
            destination = output / filename
            if backup.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)


def load_and_validate_inference_bundle(
    *, project_root: str | Path, bundle_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    relative = _require_relative_path(bundle_path, field="bundle_path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FinalizationContractError("Bundle path escapes project root.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Inference bundle not found: {relative}")
    bundle = _load_json(path)
    _validate_json_artifact("inference-bundle.json", bundle)
    model_relative = _require_relative_path(bundle["model_artifact_path"], field="model_artifact_path")
    model_path = (root / model_relative).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_relative}")
    if sha256_file(model_path) != bundle["model_artifact_sha256"]:
        raise SerializationValidationError("Model artifact hash differs from inference bundle.")
    return _deepcopy(bundle)


def load_and_validate_final_model_handoff(
    *, project_root: str | Path, handoff_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    relative = _require_relative_path(handoff_path, field="handoff_path")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Final model handoff not found: {relative}")
    handoff = _load_json(path)
    _validate_json_artifact("final-model-handoff.json", handoff)
    required_true = {
        "educational_final_model_completed",
        "final_model_trained",
        "model_artifact_materialized",
        "model_bundle_materialized",
        "final_test_evaluation_completed",
        "final_model_handoff_ready",
        "educational_inference_demo_ready",
        "test_partition_evaluated",
    }
    if any(handoff.get(key) is not True for key in required_true):
        raise FinalizationContractError("Final handoff readiness is incomplete.")
    if handoff.get("test_partition_evaluation_count") != 1:
        raise FinalizationContractError("Final test evaluation count must equal one.")
    if handoff.get("test_partition_used_for_adjustment") is not False:
        raise FinalizationContractError("Test must not be used for adjustment.")
    for reference in handoff.get("final_references", {}).values():
        ref_path = _require_relative_path(reference["path"], field="final_reference.path")
        absolute = (root / ref_path).resolve()
        if not absolute.is_file():
            raise FileNotFoundError(f"Referenced final artifact missing: {ref_path}")
        if sha256_file(absolute) != reference["byte_sha256"]:
            raise FinalizationContractError(f"Referenced final artifact hash mismatch: {ref_path}")
    return _deepcopy(handoff)


def load_trusted_pipeline_from_bundle(
    *, project_root: str | Path, bundle: Mapping[str, Any]
) -> Pipeline:
    """Load a joblib only after bundle validation and exact SHA-256 verification."""

    _validate_json_artifact("inference-bundle.json", bundle)
    root = Path(project_root).resolve()
    relative = _require_relative_path(bundle["model_artifact_path"], field="model_artifact_path")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {relative}")
    observed = sha256_file(path)
    if observed != bundle["model_artifact_sha256"]:
        raise UntrustedArtifactError("Refusing to load joblib with a divergent SHA-256.")
    loaded = joblib.load(path)
    if not isinstance(loaded, Pipeline) or not _is_fitted(loaded):
        raise SerializationValidationError("Trusted artifact is not a fitted sklearn Pipeline.")
    return loaded


# Explicit aliases retained for a readable notebook API.
validate_finalization_contract = validate_finalization_contract
