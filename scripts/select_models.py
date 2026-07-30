"""Generic, deterministic utilities for educational model selection.

The module deliberately has no knowledge of the Telco dataset. Dataset paths,
feature roles, model families, search spaces, seeds, thresholds, and readiness
contracts are supplied by the caller. The test partition is not accepted by any
selection or threshold-analysis API.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ARTIFACT_FILENAMES: tuple[str, ...] = (
    "model-selection-manifest.json",
    "candidate-results.json",
    "cross-validation-results.csv",
    "validation-evidence.json",
    "threshold-analysis.json",
    "model-selection-handoff.json",
)

_JSON_ARTIFACT_SCHEMAS: Mapping[str, tuple[str, str]] = {
    "model-selection-manifest.json": (
        "model-selection-manifest.v1",
        "model_selection_manifest",
    ),
    "candidate-results.json": ("candidate-results.v1", "candidate_results"),
    "validation-evidence.json": (
        "validation-evidence.v1",
        "validation_evidence",
    ),
    "threshold-analysis.json": (
        "threshold-analysis.v1",
        "threshold_analysis",
    ),
    "model-selection-handoff.json": (
        "model-selection-handoff.v1",
        "model_selection_handoff",
    ),
}

_VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "created_at",
        "generated_at",
        "timestamp",
        "duration_seconds",
        "search_duration_seconds",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "byte_sha256",
    }
)
_VOLATILE_CSV_COLUMNS: frozenset[str] = frozenset(
    {
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "search_duration_seconds",
    }
)


class ModelSelectionError(RuntimeError):
    """Base error for model-selection operations."""


class ModelSelectionContractError(ModelSelectionError, ValueError):
    """Raised when a model-selection contract is invalid."""


class FeatureRoleError(ModelSelectionError, ValueError):
    """Raised when feature, identifier, target, or partition roles conflict."""


class CandidateSpecificationError(ModelSelectionError, ValueError):
    """Raised when candidate model specifications are invalid."""


class NoEligibleCandidateError(ModelSelectionError):
    """Raised when no candidate exceeds the baseline by the required margin."""


class ArtifactConflictError(ModelSelectionError):
    """Raised when an existing artifact set is semantically divergent."""


class ModelSelectionHandoffError(ModelSelectionError):
    """Raised when the persisted model-selection handoff is invalid."""


@dataclass(frozen=True, slots=True)
class PartitionRoles:
    """Defensive train/validation feature and target partitions."""

    _x_train: pd.DataFrame
    _y_train: pd.Series
    _x_validation: pd.DataFrame
    _y_validation: pd.Series

    @property
    def x_train(self) -> pd.DataFrame:
        return self._x_train.copy(deep=True)

    @property
    def y_train(self) -> pd.Series:
        return self._y_train.copy(deep=True)

    @property
    def x_validation(self) -> pd.DataFrame:
        return self._x_validation.copy(deep=True)

    @property
    def y_validation(self) -> pd.Series:
        return self._y_validation.copy(deep=True)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Completed deterministic hyperparameter-search outcome."""

    model_id: str
    family: str
    search_strategy: str
    candidate_count: int
    duration_seconds: float
    search: GridSearchCV | RandomizedSearchCV

    @property
    def best_estimator(self) -> BaseEstimator:
        """Return the fitted best estimator.

        The estimator itself is intentionally not deep-copied because fitted
        forest structures can be large. Selection helpers never mutate it.
        """

        return self.search.best_estimator_

    @property
    def best_parameters(self) -> dict[str, Any]:
        return copy.deepcopy(self.search.best_params_)

    @property
    def cv_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.search.cv_results_).copy(deep=True)


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    """Result of atomic artifact materialization."""

    output_directory: Path
    created: tuple[str, ...]
    replaced: tuple[str, ...]
    idempotent: bool
    byte_sha256: Mapping[str, str]
    semantic_sha256: Mapping[str, str]


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


def _jsonable(value: Any) -> Any:
    """Convert common scientific Python values to deterministic JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
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
    """Serialize JSON canonically for semantic fingerprinting."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    """Serialize readable, deterministic JSON for artifacts."""

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


def semantic_fingerprint_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(_strip_volatile(value)))


def _semantic_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy(deep=True)
    drop = [column for column in cleaned.columns if column in _VOLATILE_CSV_COLUMNS]
    if drop:
        cleaned = cleaned.drop(columns=drop)
    # CSV round-trips can change the final binary digits of floating-point
    # values while preserving the same reported metric. Twelve decimal places
    # are materially stricter than the platform-level tolerance required here.
    for column in cleaned.select_dtypes(include=["floating"]).columns:
        cleaned[column] = cleaned[column].round(12)
    return cleaned.reset_index(drop=True)


def semantic_fingerprint_csv(frame: pd.DataFrame) -> str:
    content = _semantic_csv_frame(frame).to_csv(index=False, lineterminator="\n")
    return sha256_bytes(content.encode("utf-8"))


def _require_relative_path(value: str | Path, *, field: str) -> str:
    rendered = Path(value).as_posix() if isinstance(value, Path) else str(value)
    pure = PurePosixPath(rendered)
    if pure.is_absolute() or ".." in pure.parts:
        raise ModelSelectionContractError(
            f"{field} must be a project-relative path: {rendered}"
        )
    if len(rendered) >= 3 and rendered[1:3] in {":/", ":\\"}:
        raise ModelSelectionContractError(
            f"{field} must not be an absolute Windows path: {rendered}"
        )
    return pure.as_posix()


def _validate_paths_recursively(value: Any, *, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _validate_paths_recursively(item, prefix=child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_paths_recursively(item, prefix=f"{prefix}[{index}]")
    elif isinstance(value, str) and "path" in prefix.lower():
        _require_relative_path(value, field=prefix)


def validate_model_selection_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and defensively copy the educational selection contract."""

    result = _deepcopy(dict(contract))
    required = {
        "evaluation_mode",
        "purpose",
        "primary_metric",
        "refit_metric",
        "cv",
        "dummy_average_precision_margin",
        "practical_tie_tolerance",
        "educational_recall_target",
        "test_partition_sealed",
        "operational_validity",
        "operational_threshold",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise ModelSelectionContractError(
            f"Missing model-selection contract fields: {missing}"
        )
    if result["primary_metric"] != "average_precision":
        raise ModelSelectionContractError("Average Precision must be primary_metric.")
    if result["refit_metric"] != "average_precision":
        raise ModelSelectionContractError("Average Precision must be refit_metric.")
    cv = result["cv"]
    if not isinstance(cv, Mapping):
        raise ModelSelectionContractError("cv must be a mapping.")
    if cv.get("strategy") != "StratifiedKFold":
        raise ModelSelectionContractError("cv.strategy must be StratifiedKFold.")
    if not isinstance(cv.get("n_splits"), int) or cv["n_splits"] < 2:
        raise ModelSelectionContractError("cv.n_splits must be an integer >= 2.")
    if cv.get("shuffle") is not True:
        raise ModelSelectionContractError("cv.shuffle must be true.")
    if not isinstance(cv.get("random_state"), int):
        raise ModelSelectionContractError("cv.random_state must be an integer.")
    if result["test_partition_sealed"] is not True:
        raise ModelSelectionContractError("test_partition_sealed must be true.")
    if result["operational_validity"] != "unconfirmed":
        raise ModelSelectionContractError(
            "operational_validity must remain 'unconfirmed'."
        )
    if result["operational_threshold"] != "unresolved":
        raise ModelSelectionContractError(
            "operational_threshold must remain 'unresolved'."
        )
    recall_target = result["educational_recall_target"]
    if not isinstance(recall_target, (int, float)) or not 0 < recall_target <= 1:
        raise ModelSelectionContractError(
            "educational_recall_target must be in the interval (0, 1]."
        )
    for field in ("dummy_average_precision_margin", "practical_tie_tolerance"):
        if not isinstance(result[field], (int, float)) or result[field] < 0:
            raise ModelSelectionContractError(f"{field} must be non-negative.")
    _validate_paths_recursively(result)
    return result


def validate_feature_partition_roles(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: Sequence[str],
    identifier_columns: Sequence[str],
    target_column: str,
    target_classes: Sequence[Any],
    positive_class: Any,
    target_encoding: Mapping[Any, int],
) -> PartitionRoles:
    """Validate partition roles without accepting or touching test data."""

    features = tuple(feature_columns)
    identifiers = tuple(identifier_columns)
    if not features or len(set(features)) != len(features):
        raise FeatureRoleError("feature_columns must be non-empty and unique.")
    if len(set(identifiers)) != len(identifiers):
        raise FeatureRoleError("identifier_columns must be unique.")
    prohibited = set(identifiers) | {target_column}
    leaked = [column for column in features if column in prohibited]
    if leaked:
        raise FeatureRoleError(f"Predictors contain identifier/target columns: {leaked}")
    if positive_class not in target_classes:
        raise FeatureRoleError("positive_class is not present in target_classes.")
    if set(target_encoding) != set(target_classes):
        raise FeatureRoleError("target_encoding keys must match target_classes.")
    if sorted(target_encoding.values()) != list(range(len(target_classes))):
        raise FeatureRoleError("target_encoding values must be contiguous integers.")
    if target_encoding[positive_class] != 1:
        raise FeatureRoleError("The positive class must be encoded as 1.")

    expected = list(identifiers) + list(features) + [target_column]
    for name, frame in (("train", train), ("validation", validation)):
        if not isinstance(frame, pd.DataFrame):
            raise FeatureRoleError(f"{name} must be a pandas DataFrame.")
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise FeatureRoleError(f"{name} is missing required columns: {missing}")
        observed = list(frame[target_column].dropna().unique())
        unexpected = sorted(set(observed) - set(target_classes), key=str)
        if unexpected:
            raise FeatureRoleError(
                f"{name} contains unexpected target classes: {unexpected}"
            )
        if frame[target_column].isna().any():
            raise FeatureRoleError(f"{name} target contains missing values.")

    x_train = train.loc[:, list(features)].copy(deep=True)
    x_validation = validation.loc[:, list(features)].copy(deep=True)
    if list(x_train.columns) != list(features):
        raise FeatureRoleError("Training feature order does not match the contract.")
    if list(x_validation.columns) != list(features):
        raise FeatureRoleError("Validation feature order does not match the contract.")
    y_train = train[target_column].map(target_encoding)
    y_validation = validation[target_column].map(target_encoding)
    if y_train.isna().any() or y_validation.isna().any():
        raise FeatureRoleError("Target encoding produced missing values.")
    y_train = y_train.astype("int64").copy(deep=True)
    y_validation = y_validation.astype("int64").copy(deep=True)
    return PartitionRoles(x_train, y_train, x_validation, y_validation)


def validate_candidate_model_specs(
    *,
    baseline_spec: Mapping[str, Any],
    candidate_specs: Sequence[Mapping[str, Any]],
    required_family_search_strategies: Mapping[str, str],
    expected_candidate_count: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Validate caller-declared baseline and candidate specifications."""

    baseline = _deepcopy(dict(baseline_spec))
    candidates = tuple(_deepcopy(dict(spec)) for spec in candidate_specs)
    if baseline.get("eligible") is not False:
        raise CandidateSpecificationError("The baseline must be non-eligible.")
    estimator = baseline.get("estimator")
    if not isinstance(estimator, DummyClassifier) or estimator.strategy != "prior":
        raise CandidateSpecificationError(
            "The baseline estimator must be DummyClassifier(strategy='prior')."
        )
    if len(candidates) != expected_candidate_count:
        raise CandidateSpecificationError(
            f"Expected {expected_candidate_count} candidate models, got {len(candidates)}."
        )
    ids = [spec.get("model_id") for spec in candidates]
    if any(not isinstance(model_id, str) or not model_id for model_id in ids):
        raise CandidateSpecificationError("Every candidate requires a stable model_id.")
    if len(set(ids)) != len(ids):
        raise CandidateSpecificationError("Candidate model_id values must be unique.")
    observed_families = {spec.get("family") for spec in candidates}
    if observed_families != set(required_family_search_strategies):
        raise CandidateSpecificationError(
            "Candidate families differ from the declared required families."
        )
    for spec in candidates:
        family = spec["family"]
        strategy = spec.get("search_strategy")
        expected_strategy = required_family_search_strategies[family]
        if strategy != expected_strategy:
            raise CandidateSpecificationError(
                f"{family} requires {expected_strategy}, got {strategy}."
            )
        if not isinstance(spec.get("estimator"), BaseEstimator):
            raise CandidateSpecificationError(f"{family} estimator is invalid.")
        space = spec.get("search_space")
        if not isinstance(space, Mapping) or not space:
            raise CandidateSpecificationError(f"{family} search_space is empty.")
        for key, values in space.items():
            if not str(key).startswith("model__"):
                raise CandidateSpecificationError(
                    f"Search parameter must target the model step: {key}"
                )
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise CandidateSpecificationError(
                    f"Search parameter values must be a sequence: {key}"
                )
            if len(values) == 0:
                raise CandidateSpecificationError(f"Search parameter is empty: {key}")
        available = set(build_candidate_pipeline(
            estimator=spec["estimator"],
            numerical_features=spec.get("numerical_features", ("__placeholder_num__",)),
            categorical_features=spec.get("categorical_features", ("__placeholder_cat__",)),
            scale_numerical=bool(spec.get("scale_numerical", False)),
        ).get_params(deep=True))
        invalid = sorted(set(space) - available)
        if invalid:
            raise CandidateSpecificationError(
                f"{family} search parameters are incompatible: {invalid}"
            )
        total = len(ParameterGrid(space))
        if strategy == "GridSearchCV":
            if spec.get("candidate_count") not in (None, total):
                raise CandidateSpecificationError(
                    f"{family} candidate_count does not match the grid size {total}."
                )
        else:
            n_iter = spec.get("n_iter")
            if not isinstance(n_iter, int) or n_iter <= 0:
                raise CandidateSpecificationError(f"{family} n_iter must be positive.")
            if n_iter > total:
                raise CandidateSpecificationError(
                    f"{family} n_iter exceeds the finite search space ({total})."
                )
        if spec.get("random_state") is not None and not isinstance(
            spec.get("random_state"), int
        ):
            raise CandidateSpecificationError(f"{family} random_state must be integer.")
    return baseline, candidates


def _one_hot_encoder() -> OneHotEncoder:
    """Build a dense, drop-free encoder compatible with sklearn >= 1.2."""

    kwargs = {"handle_unknown": "ignore", "drop": None}
    signature = inspect.signature(OneHotEncoder)
    if "sparse_output" in signature.parameters:
        kwargs["sparse_output"] = False
    else:  # pragma: no cover - retained for older compatible environments
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def build_preprocessing_pipeline(
    *,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    scale_numerical: bool,
) -> ColumnTransformer:
    """Build an unfitted fold-safe preprocessing transformer."""

    numerical = list(numerical_features)
    categorical = list(categorical_features)
    if not numerical and not categorical:
        raise FeatureRoleError("At least one numerical or categorical feature is required.")
    if set(numerical) & set(categorical):
        raise FeatureRoleError("Numerical and categorical features must be disjoint.")
    transformers: list[tuple[str, Any, list[str]]] = []
    if numerical:
        numeric_transformer: Any = StandardScaler() if scale_numerical else "passthrough"
        transformers.append(("numerical", numeric_transformer, numerical))
    if categorical:
        transformers.append(("categorical", _one_hot_encoder(), categorical))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def build_candidate_pipeline(
    *,
    estimator: BaseEstimator,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    scale_numerical: bool,
) -> Pipeline:
    """Build an unfitted preprocessing-plus-estimator pipeline."""

    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessing_pipeline(
                    numerical_features=numerical_features,
                    categorical_features=categorical_features,
                    scale_numerical=scale_numerical,
                ),
            ),
            ("model", clone(estimator)),
        ]
    )


def build_scoring_contract(*, positive_label: int = 1) -> dict[str, Any]:
    """Build explicit probability and positive-class scorers."""

    return {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "precision": make_scorer(
            precision_score, pos_label=positive_label, zero_division=0
        ),
        "recall": make_scorer(
            recall_score, pos_label=positive_label, zero_division=0
        ),
        "f1": make_scorer(f1_score, pos_label=positive_label, zero_division=0),
        "f2": make_scorer(
            fbeta_score,
            beta=2,
            pos_label=positive_label,
            zero_division=0,
        ),
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "neg_log_loss": "neg_log_loss",
        "neg_brier_score": "neg_brier_score",
    }


def build_cross_validation(
    *, n_splits: int, shuffle: bool, random_state: int
) -> StratifiedKFold:
    """Build deterministic stratified cross-validation."""

    if not isinstance(n_splits, int) or n_splits < 2:
        raise ModelSelectionContractError("n_splits must be an integer >= 2.")
    if shuffle is not True:
        raise ModelSelectionContractError("Cross-validation shuffle must be true.")
    if not isinstance(random_state, int):
        raise ModelSelectionContractError("random_state must be an integer.")
    return StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )


def describe_cv_folds(
    *, cv: StratifiedKFold, x: pd.DataFrame, y: pd.Series
) -> list[dict[str, Any]]:
    """Return class distributions for each train/validation fold."""

    rows: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(cv.split(x, y), start=1):
        y_train = y.iloc[train_index]
        y_validation = y.iloc[validation_index]
        rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "train_class_counts": {
                    str(key): int(value)
                    for key, value in y_train.value_counts().sort_index().items()
                },
                "validation_class_counts": {
                    str(key): int(value)
                    for key, value in y_validation.value_counts().sort_index().items()
                },
            }
        )
    return rows


def _candidate_count(strategy: str, space: Mapping[str, Sequence[Any]], n_iter: int | None) -> int:
    total = len(ParameterGrid(space))
    if strategy == "GridSearchCV":
        return total
    if n_iter is None:
        raise CandidateSpecificationError("RandomizedSearchCV requires n_iter.")
    return min(int(n_iter), total)


def run_model_search(
    *,
    model_id: str,
    family: str,
    pipeline: Pipeline,
    search_strategy: str,
    search_space: Mapping[str, Sequence[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: Mapping[str, Any],
    cv: StratifiedKFold,
    refit_metric: str,
    n_jobs: int,
    random_state: int | None = None,
    n_iter: int | None = None,
    error_score: str | float = "raise",
) -> SearchOutcome:
    """Run a declared search exclusively on the supplied training partition."""

    x = x_train.copy(deep=True)
    y = y_train.copy(deep=True)
    if refit_metric != "average_precision":
        raise ModelSelectionContractError("Search refit_metric must be average_precision.")
    if refit_metric not in scoring:
        raise ModelSelectionContractError("refit_metric is absent from scoring.")
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ModelSelectionContractError("n_jobs must be a non-zero integer.")
    common = dict(
        estimator=clone(pipeline),
        scoring=dict(scoring),
        refit=refit_metric,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score=error_score,
    )
    if search_strategy == "GridSearchCV":
        search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
            param_grid=_deepcopy(dict(search_space)), **common
        )
    elif search_strategy == "RandomizedSearchCV":
        if not isinstance(n_iter, int) or n_iter <= 0:
            raise CandidateSpecificationError("RandomizedSearchCV requires positive n_iter.")
        if not isinstance(random_state, int):
            raise CandidateSpecificationError(
                "RandomizedSearchCV requires an integer random_state."
            )
        search = RandomizedSearchCV(
            param_distributions=_deepcopy(dict(search_space)),
            n_iter=n_iter,
            random_state=random_state,
            **common,
        )
    else:
        raise CandidateSpecificationError(
            f"Unsupported search strategy: {search_strategy}"
        )
    started = time.perf_counter()
    search.fit(x, y)
    duration = time.perf_counter() - started
    expected = _candidate_count(search_strategy, search_space, n_iter)
    actual = len(search.cv_results_["params"])
    if actual != expected:
        raise ModelSelectionError(
            f"Search candidate count mismatch for {model_id}: expected {expected}, got {actual}."
        )
    return SearchOutcome(
        model_id=model_id,
        family=family,
        search_strategy=search_strategy,
        candidate_count=actual,
        duration_seconds=float(duration),
        search=search,
    )


def compute_cv_confidence_interval(
    mean: float, std: float, *, n_splits: int, z_value: float = 1.96
) -> tuple[float, float]:
    """Return the requested approximate mean ± z * std / sqrt(k) interval."""

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    half_width = z_value * float(std) / math.sqrt(n_splits)
    return float(mean - half_width), float(mean + half_width)


def summarize_search_results(
    outcome: SearchOutcome, *, n_splits: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize the best search row and normalize the full CV table."""

    frame = outcome.cv_results
    best_index = int(outcome.search.best_index_)
    best = frame.iloc[best_index]
    mean_ap = float(best["mean_test_average_precision"])
    std_ap = float(best["std_test_average_precision"])
    lower, upper = compute_cv_confidence_interval(
        mean_ap, std_ap, n_splits=n_splits
    )
    summary = {
        "model_id": outcome.model_id,
        "family": outcome.family,
        "search_strategy": outcome.search_strategy,
        "candidate_count": int(outcome.candidate_count),
        "best_parameters": _jsonable(outcome.best_parameters),
        "best_index": best_index,
        "best_cv_average_precision": mean_ap,
        "cv_average_precision_mean": mean_ap,
        "cv_average_precision_std": std_ap,
        "cv_average_precision_confidence_lower": lower,
        "cv_average_precision_confidence_upper": upper,
        "cv_roc_auc_mean": float(best["mean_test_roc_auc"]),
        "cv_roc_auc_std": float(best["std_test_roc_auc"]),
        "cv_precision_mean": float(best["mean_test_precision"]),
        "cv_precision_std": float(best["std_test_precision"]),
        "cv_recall_mean": float(best["mean_test_recall"]),
        "cv_recall_std": float(best["std_test_recall"]),
        "cv_f1_mean": float(best["mean_test_f1"]),
        "cv_f2_mean": float(best["mean_test_f2"]),
        "cv_balanced_accuracy_mean": float(best["mean_test_balanced_accuracy"]),
        "cv_accuracy_mean": float(best["mean_test_accuracy"]),
        "cv_log_loss_mean": float(-best["mean_test_neg_log_loss"]),
        "cv_brier_score_mean": float(-best["mean_test_neg_brier_score"]),
        "mean_fit_time": float(best["mean_fit_time"]),
        "std_fit_time": float(best["std_fit_time"]),
        "search_duration_seconds": float(outcome.duration_seconds),
    }

    normalized = pd.DataFrame(
        {
            "model_id": outcome.model_id,
            "family": outcome.family,
            "search_strategy": outcome.search_strategy,
            "candidate_index": list(range(len(frame))),
            "parameters": [
                json.dumps(_jsonable(params), sort_keys=True, separators=(",", ":"))
                for params in frame["params"]
            ],
            "rank_average_precision": frame["rank_test_average_precision"].astype(int),
            "mean_cv_average_precision": frame["mean_test_average_precision"],
            "std_cv_average_precision": frame["std_test_average_precision"],
            "mean_cv_roc_auc": frame["mean_test_roc_auc"],
            "std_cv_roc_auc": frame["std_test_roc_auc"],
            "mean_cv_precision": frame["mean_test_precision"],
            "std_cv_precision": frame["std_test_precision"],
            "mean_cv_recall": frame["mean_test_recall"],
            "std_cv_recall": frame["std_test_recall"],
            "mean_cv_f1": frame["mean_test_f1"],
            "std_cv_f1": frame["std_test_f1"],
            "mean_cv_f2": frame["mean_test_f2"],
            "std_cv_f2": frame["std_test_f2"],
            "mean_cv_balanced_accuracy": frame["mean_test_balanced_accuracy"],
            "std_cv_balanced_accuracy": frame["std_test_balanced_accuracy"],
            "mean_cv_accuracy": frame["mean_test_accuracy"],
            "std_cv_accuracy": frame["std_test_accuracy"],
            "mean_cv_log_loss": -frame["mean_test_neg_log_loss"],
            "std_cv_log_loss": frame["std_test_neg_log_loss"],
            "mean_cv_brier_score": -frame["mean_test_neg_brier_score"],
            "std_cv_brier_score": frame["std_test_neg_brier_score"],
            "mean_fit_time": frame["mean_fit_time"],
            "std_fit_time": frame["std_fit_time"],
            "mean_score_time": frame["mean_score_time"],
            "std_score_time": frame["std_score_time"],
        }
    )
    return summary, normalized.copy(deep=True)


def compute_fbeta(
    y_true: Sequence[int], y_pred: Sequence[int], *, beta: float
) -> float:
    """Compute an explicitly zero-division-safe F-beta score."""

    return float(
        fbeta_score(y_true, y_pred, beta=beta, pos_label=1, zero_division=0)
    )


def _positive_probabilities(estimator: BaseEstimator, x: pd.DataFrame) -> list[float]:
    if not hasattr(estimator, "predict_proba"):
        raise ModelSelectionError("Estimator does not expose predict_proba.")
    probabilities = estimator.predict_proba(x.copy(deep=True))
    classes = list(getattr(estimator, "classes_", []))
    if 1 not in classes:
        raise ModelSelectionError("Fitted estimator does not expose positive class 1.")
    index = classes.index(1)
    return [float(value) for value in probabilities[:, index]]


def evaluate_probability_classifier(
    *,
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y_true: pd.Series | Sequence[int],
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate a fitted probability classifier on an explicitly supplied set."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")
    y = pd.Series(y_true, copy=True).astype(int).reset_index(drop=True)
    probabilities = _positive_probabilities(estimator, x)
    predicted = [int(value >= threshold) for value in probabilities]
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision_curve, recall_curve_values, pr_thresholds = precision_recall_curve(
        y, probabilities, pos_label=1
    )
    if y.nunique() == 2:
        fpr, tpr, roc_thresholds = roc_curve(y, probabilities, pos_label=1)
        roc_auc = float(roc_auc_score(y, probabilities))
    else:
        fpr, tpr, roc_thresholds, roc_auc = [], [], [], None
    probability_true, probability_predicted = calibration_curve(
        y, probabilities, n_bins=calibration_bins, strategy="uniform"
    )
    metrics = {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": roc_auc,
        "precision": float(precision_score(y, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        "f2": compute_fbeta(y, predicted, beta=2),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "accuracy": float(accuracy_score(y, predicted)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "predicted_positive_count": int(sum(predicted)),
        "predicted_positive_rate": float(sum(predicted) / len(predicted)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    return {
        "metrics": metrics,
        "precision_recall_curve": {
            "precision": [float(value) for value in precision_curve],
            "recall": [float(value) for value in recall_curve_values],
            "thresholds": [float(value) for value in pr_thresholds],
        },
        "roc_curve": {
            "false_positive_rate": [float(value) for value in fpr],
            "true_positive_rate": [float(value) for value in tpr],
            "thresholds": [float(value) for value in roc_thresholds],
        },
        "calibration_curve": {
            "mean_predicted_probability": [float(value) for value in probability_predicted],
            "fraction_of_positives": [float(value) for value in probability_true],
            "n_bins": int(calibration_bins),
        },
        "positive_probabilities": list(probabilities),
    }


def evaluate_candidates_on_validation(
    *,
    estimators: Mapping[str, BaseEstimator],
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Evaluate each tuned family once on the external validation partition."""

    results: dict[str, dict[str, Any]] = {}
    for model_id in sorted(estimators):
        results[model_id] = evaluate_probability_classifier(
            estimator=estimators[model_id],
            x=x_validation,
            y_true=y_validation,
            threshold=threshold,
        )
    return _deepcopy(results)


def report_unknown_categories(
    *, estimator: BaseEstimator, x_validation: pd.DataFrame
) -> dict[str, list[Any]]:
    """Report validation categories unseen by the fitted training encoder."""

    if not isinstance(estimator, Pipeline):
        raise ModelSelectionError("Expected a fitted sklearn Pipeline.")
    preprocess = estimator.named_steps.get("preprocess")
    if not isinstance(preprocess, ColumnTransformer):
        raise ModelSelectionError("Pipeline is missing the fitted ColumnTransformer.")
    encoder = preprocess.named_transformers_.get("categorical")
    if not isinstance(encoder, OneHotEncoder):
        return {}
    categorical_columns: list[str] | None = None
    for name, _transformer, columns in preprocess.transformers_:
        if name == "categorical":
            categorical_columns = list(columns)
            break
    if categorical_columns is None:
        return {}
    report: dict[str, list[Any]] = {}
    for column, fitted_categories in zip(categorical_columns, encoder.categories_):
        known = {_jsonable(value) for value in fitted_categories}
        observed = {_jsonable(value) for value in x_validation[column].dropna().unique()}
        unknown = sorted(observed - known, key=lambda item: (str(type(item)), str(item)))
        if unknown:
            report[column] = unknown
    return _deepcopy(report)


def detect_practical_tie(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    tolerance: float,
) -> bool:
    """Apply validation-difference and overlapping-CV-interval tie rules."""

    difference = abs(
        float(first["validation_average_precision"])
        - float(second["validation_average_precision"])
    )
    overlap = max(
        float(first["cv_average_precision_confidence_lower"]),
        float(second["cv_average_precision_confidence_lower"]),
    ) <= min(
        float(first["cv_average_precision_confidence_upper"]),
        float(second["cv_average_precision_confidence_upper"]),
    )
    return difference <= tolerance and overlap


def select_candidate_model(
    *,
    cv_summaries: Mapping[str, Mapping[str, Any]],
    validation_evaluations: Mapping[str, Mapping[str, Any]],
    dummy_validation_metrics: Mapping[str, Any],
    dummy_average_precision_margin: float,
    practical_tie_tolerance: float,
    simplicity_order: Sequence[str],
) -> dict[str, Any]:
    """Select an eligible family with deterministic practical-tie handling."""

    simplicity_rank = {family: index for index, family in enumerate(simplicity_order)}
    dummy_ap = float(dummy_validation_metrics["average_precision"])
    records: list[dict[str, Any]] = []
    for model_id in sorted(cv_summaries):
        cv = cv_summaries[model_id]
        evaluation = validation_evaluations[model_id]
        metrics = evaluation["metrics"]
        margin = float(metrics["average_precision"]) - dummy_ap
        family = str(cv["family"])
        records.append(
            {
                "model_id": model_id,
                "family": family,
                "validation_average_precision": float(metrics["average_precision"]),
                "validation_roc_auc": float(metrics["roc_auc"]),
                "validation_brier_score": float(metrics["brier_score"]),
                "validation_log_loss": float(metrics["log_loss"]),
                "cv_average_precision_std": float(cv["cv_average_precision_std"]),
                "cv_average_precision_confidence_lower": float(
                    cv["cv_average_precision_confidence_lower"]
                ),
                "cv_average_precision_confidence_upper": float(
                    cv["cv_average_precision_confidence_upper"]
                ),
                "margin_over_dummy": margin,
                "eligible": margin > dummy_average_precision_margin,
                "simplicity_rank": simplicity_rank.get(family, len(simplicity_rank)),
            }
        )
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        raise NoEligibleCandidateError(
            "No candidate exceeded the Dummy Average Precision by the required margin."
        )
    eligible.sort(
        key=lambda row: (-row["validation_average_precision"], row["model_id"])
    )
    finalists = eligible[:2]
    practical_tie = len(finalists) == 2 and detect_practical_tie(
        finalists[0], finalists[1], tolerance=practical_tie_tolerance
    )
    criteria: list[dict[str, Any]] = []
    selected = finalists[0]
    rationale: str
    if practical_tie:
        first, second = finalists
        comparisons = (
            ("lower_validation_brier_score", "validation_brier_score", "min"),
            ("lower_validation_log_loss", "validation_log_loss", "min"),
            ("lower_cv_average_precision_std", "cv_average_precision_std", "min"),
            ("higher_validation_roc_auc", "validation_roc_auc", "max"),
            ("higher_interpretability", "simplicity_rank", "min"),
            ("lower_complexity", "simplicity_rank", "min"),
            ("stable_model_id", "model_id", "min"),
        )
        for criterion, field, direction in comparisons:
            a, b = first[field], second[field]
            winner: str | None = None
            if isinstance(a, (float, int)) and isinstance(b, (float, int)):
                if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12):
                    if direction == "min":
                        winner = first["model_id"] if a < b else second["model_id"]
                    else:
                        winner = first["model_id"] if a > b else second["model_id"]
            elif a != b:
                winner = first["model_id"] if str(a) < str(b) else second["model_id"]
            criteria.append(
                {
                    "criterion": criterion,
                    "first_value": _jsonable(a),
                    "second_value": _jsonable(b),
                    "winner": winner,
                }
            )
            if winner is not None:
                selected = first if first["model_id"] == winner else second
                break
        rationale = (
            "The finalists were practically tied on validation Average Precision with "
            "overlapping approximate CV intervals; deterministic calibration, stability, "
            "interpretability, complexity, and stable-ID criteria were applied in order."
        )
    else:
        rationale = (
            "The selected candidate had the highest validation Average Precision among "
            "eligible models without satisfying the practical-tie rule against the runner-up."
        )
    return {
        "dummy_average_precision": dummy_ap,
        "required_margin": float(dummy_average_precision_margin),
        "eligible_model_ids": [record["model_id"] for record in eligible],
        "candidate_eligibility": records,
        "finalists": [record["model_id"] for record in finalists],
        "practical_tie": bool(practical_tie),
        "practical_tie_tolerance": float(practical_tie_tolerance),
        "criteria_applied": criteria,
        "selected_model_id": selected["model_id"],
        "selected_model_family": selected["family"],
        "selection_rationale": rationale,
    }


def _threshold_rows(
    y_true: pd.Series, probabilities: Sequence[float], thresholds: Sequence[float]
) -> list[dict[str, Any]]:
    """Compute confusion-derived threshold metrics with cumulative counts."""

    labels = [int(value) for value in y_true]
    pairs = sorted(zip((float(value) for value in probabilities), labels), key=lambda item: item[0])
    sorted_scores = [score for score, _label in pairs]
    prefix_positives = [0]
    for _score, label in pairs:
        prefix_positives.append(prefix_positives[-1] + int(label == 1))
    total_rows = len(pairs)
    total_positives = prefix_positives[-1]
    total_negatives = total_rows - total_positives
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        index = bisect.bisect_left(sorted_scores, float(threshold))
        predicted_positive_count = total_rows - index
        true_positives = total_positives - prefix_positives[index]
        false_positives = predicted_positive_count - true_positives
        false_negatives = total_positives - true_positives
        true_negatives = total_negatives - false_positives
        precision = (
            true_positives / predicted_positive_count
            if predicted_positive_count
            else 0.0
        )
        recall = true_positives / total_positives if total_positives else 0.0
        f1_denominator = precision + recall
        f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0
        f2_denominator = 4 * precision + recall
        f2 = 5 * precision * recall / f2_denominator if f2_denominator else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "f2": float(f2),
                "true_positives": int(true_positives),
                "false_positives": int(false_positives),
                "true_negatives": int(true_negatives),
                "false_negatives": int(false_negatives),
                "predicted_positive_count": int(predicted_positive_count),
                "predicted_positive_rate": float(predicted_positive_count / total_rows),
            }
        )
    return rows


def _threshold_metrics(
    y_true: pd.Series, probabilities: Sequence[float], threshold: float
) -> dict[str, Any]:
    return _threshold_rows(y_true, probabilities, [threshold])[0]


def _choose_maximum(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    return dict(
        max(
            rows,
            key=lambda row: (
                float(row[metric]),
                float(row["precision"]),
                float(row["recall"]),
                float(row["threshold"]),
            ),
        )
    )


def _choose_recall_target(
    rows: Sequence[Mapping[str, Any]], target: float
) -> tuple[dict[str, Any], bool]:
    satisfying = [row for row in rows if float(row["recall"]) >= target]
    if not satisfying:
        best = max(rows, key=lambda row: (float(row["recall"]), float(row["precision"])))
        return dict(best), False
    best = max(
        satisfying,
        key=lambda row: (
            float(row["precision"]),
            float(row["f2"]),
            float(row["threshold"]),
        ),
    )
    return dict(best), True


def analyze_thresholds(
    *,
    y_validation: pd.Series | Sequence[int],
    positive_probabilities: Sequence[float],
    recall_targets: Sequence[float] = (0.70, 0.80, 0.90),
    selected_scenario_id: str = "minimum_recall_0_80",
) -> dict[str, Any]:
    """Analyze educational thresholds exclusively on validation labels/scores."""

    y = pd.Series(y_validation, copy=True).astype(int).reset_index(drop=True)
    probabilities = [float(value) for value in positive_probabilities]
    if len(y) != len(probabilities) or len(y) == 0:
        raise ValueError("Validation labels and probabilities must have equal non-zero length.")
    if any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("Probabilities must be in [0, 1].")
    thresholds = sorted(set([0.0, 0.5, 1.0, *probabilities]))
    rows = _threshold_rows(y, probabilities, thresholds)
    scenarios: list[dict[str, Any]] = []

    default = _threshold_metrics(y, probabilities, 0.5)
    scenarios.append(
        {
            "scenario_id": "default_0_50",
            **default,
            "recall_target": None,
            "target_satisfied": None,
            "selection_rule": "fixed_threshold_0_50",
        }
    )
    for scenario_id, metric in (("maximum_f1", "f1"), ("maximum_f2", "f2")):
        chosen = _choose_maximum(rows, metric)
        scenarios.append(
            {
                "scenario_id": scenario_id,
                **chosen,
                "recall_target": None,
                "target_satisfied": None,
                "selection_rule": f"maximize_{metric}_then_precision_recall_threshold",
            }
        )
    for target in recall_targets:
        chosen, satisfied = _choose_recall_target(rows, float(target))
        scenario_id = f"minimum_recall_{target:.2f}".replace(".", "_")
        scenarios.append(
            {
                "scenario_id": scenario_id,
                **chosen,
                "recall_target": float(target),
                "target_satisfied": bool(satisfied),
                "selection_rule": (
                    "maximize_precision_subject_to_minimum_recall_then_f2_then_highest_threshold"
                ),
            }
        )
    selected = next(
        (scenario for scenario in scenarios if scenario["scenario_id"] == selected_scenario_id),
        None,
    )
    if selected is None:
        raise ValueError(f"Selected threshold scenario is absent: {selected_scenario_id}")
    if selected.get("target_satisfied") is False:
        raise ModelSelectionError(
            f"Selected threshold scenario did not satisfy its recall target: {selected_scenario_id}"
        )
    return {
        "partition": "validation",
        "threshold_strategy": "maximize_precision_subject_to_minimum_recall",
        "supporting_threshold_metric": "f2",
        "scenarios": _deepcopy(scenarios),
        "selected_scenario_id": selected_scenario_id,
        "selected_educational_threshold": float(selected["threshold"]),
        "selected_scenario": _deepcopy(selected),
        "operational_threshold": "unresolved",
        "test_partition_policy": "sealed",
        "threshold_grid": {
            "thresholds": [float(row["threshold"]) for row in rows],
            "precision": [float(row["precision"]) for row in rows],
            "recall": [float(row["recall"]) for row in rows],
            "f1": [float(row["f1"]) for row in rows],
            "f2": [float(row["f2"]) for row in rows],
        },
    }


def runtime_versions() -> dict[str, str]:
    """Return relevant runtime versions without machine-specific paths."""

    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.system(),
    }


def build_model_selection_manifest(
    *,
    dataset_slug: str,
    preparation_references: Mapping[str, Any],
    preparation_fingerprints: Mapping[str, Any],
    model_selection_contract: Mapping[str, Any],
    candidate_model_ids: Sequence[str],
    baseline_model_id: str,
    cv_contract: Mapping[str, Any],
    scoring_contract: Mapping[str, Any],
    search_strategies: Mapping[str, Any],
    search_spaces: Mapping[str, Any],
    random_seeds: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    readiness: Mapping[str, Any],
    limitations: Sequence[str],
) -> dict[str, Any]:
    """Build the model-selection manifest payload."""

    payload = {
        "schema_version": "model-selection-manifest.v1",
        "artifact_type": "model_selection_manifest",
        "dataset_slug": dataset_slug,
        "preparation_references": _deepcopy(preparation_references),
        "preparation_fingerprints": _deepcopy(preparation_fingerprints),
        "model_selection_contract": _deepcopy(model_selection_contract),
        "candidate_model_ids": list(candidate_model_ids),
        "baseline_model_id": baseline_model_id,
        "cv_contract": _deepcopy(cv_contract),
        "scoring_contract": _deepcopy(scoring_contract),
        "search_strategies": _deepcopy(search_strategies),
        "search_spaces": _jsonable(search_spaces),
        "random_seeds": _deepcopy(random_seeds),
        "runtime_versions": runtime_versions(),
        "artifact_paths": _deepcopy(artifact_paths),
        "artifact_fingerprints": {},
        "readiness": _deepcopy(readiness),
        "limitations": list(limitations),
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
        "operational_validity": "unconfirmed",
    }
    _validate_paths_recursively(payload)
    return payload


def build_candidate_results(
    *,
    baseline: Mapping[str, Any],
    candidate_summaries: Sequence[Mapping[str, Any]],
    validation_evaluations: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
    warnings_by_model: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Build baseline and candidate audit results without row-level data."""

    candidates: list[dict[str, Any]] = []
    eligibility = {
        row["model_id"]: row
        for row in selection.get("candidate_eligibility", [])
    }
    for summary in candidate_summaries:
        model_id = str(summary["model_id"])
        metrics = validation_evaluations[model_id]["metrics"]
        candidates.append(
            {
                **_deepcopy(summary),
                "validation_metrics_at_0_50": _deepcopy(metrics),
                "eligible": bool(eligibility[model_id]["eligible"]),
                "margin_over_dummy": float(eligibility[model_id]["margin_over_dummy"]),
                "failures": [],
                "warnings": list((warnings_by_model or {}).get(model_id, [])),
            }
        )
    return {
        "schema_version": "candidate-results.v1",
        "artifact_type": "candidate_results",
        "baseline": _deepcopy(baseline),
        "candidates": candidates,
        "selection": _deepcopy(selection),
        "test_partition_evaluated": False,
    }


def _without_probabilities(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    result = _deepcopy(dict(evaluation))
    result.pop("positive_probabilities", None)
    return result


def build_validation_evidence(
    *,
    dataset_slug: str,
    evaluations: Mapping[str, Mapping[str, Any]],
    dummy_evaluation: Mapping[str, Any],
    unknown_categories: Mapping[str, Mapping[str, Sequence[Any]]],
    selection: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Build aggregate validation evidence with no personal row data."""

    return {
        "schema_version": "validation-evidence.v1",
        "artifact_type": "validation_evidence",
        "dataset_slug": dataset_slug,
        "partition": "validation",
        "classification_threshold": float(threshold),
        "baseline": _without_probabilities(dummy_evaluation),
        "models": {
            model_id: _without_probabilities(evaluation)
            for model_id, evaluation in sorted(evaluations.items())
        },
        "unknown_categories_report": _deepcopy(unknown_categories),
        "selection": _deepcopy(selection),
        "test_partition_evaluated": False,
    }


def build_threshold_analysis(
    *,
    dataset_slug: str,
    selected_model_id: str,
    analysis: Mapping[str, Any],
    educational_recall_target: float,
) -> dict[str, Any]:
    """Build the educational validation-threshold artifact."""

    return {
        "schema_version": "threshold-analysis.v1",
        "artifact_type": "threshold_analysis",
        "dataset_slug": dataset_slug,
        "selected_model_id": selected_model_id,
        **_deepcopy(dict(analysis)),
        "educational_recall_target": float(educational_recall_target),
        "false_negative_cost": "unknown",
        "false_positive_cost": "unknown",
        "business_cost_ratio": "unavailable",
        "operational_recall_target": "unresolved",
        "operational_threshold": "unresolved",
        "test_partition_policy": "sealed",
    }


def build_model_selection_handoff(
    *,
    dataset_slug: str,
    preparation_contract_references: Mapping[str, Any],
    selected_model_family: str,
    selected_model_id: str,
    selected_hyperparameters: Mapping[str, Any],
    selected_preprocessing_contract: Mapping[str, Any],
    selected_validation_metrics: Mapping[str, Any],
    selected_educational_threshold: Mapping[str, Any],
    threshold_selection_rule: str,
    selection_rationale: str,
    feature_columns: Sequence[str],
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    target_encoding: Mapping[Any, int],
    positive_class: Any,
    random_seeds: Mapping[str, Any],
    final_training_instructions: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the declarative notebook-04 handoff without a trained model."""

    return {
        "schema_version": "model-selection-handoff.v1",
        "artifact_type": "model_selection_handoff",
        "dataset_slug": dataset_slug,
        "preparation_contract_references": _deepcopy(preparation_contract_references),
        "selected_model_family": selected_model_family,
        "selected_model_id": selected_model_id,
        "selected_hyperparameters": _jsonable(selected_hyperparameters),
        "selected_preprocessing_contract": _deepcopy(selected_preprocessing_contract),
        "selected_validation_metrics": _deepcopy(selected_validation_metrics),
        "selected_educational_threshold": _deepcopy(selected_educational_threshold),
        "threshold_selection_rule": threshold_selection_rule,
        "selection_rationale": selection_rationale,
        "feature_columns": list(feature_columns),
        "numerical_features": list(numerical_features),
        "categorical_features": list(categorical_features),
        "target_encoding": _jsonable(target_encoding),
        "positive_class": _jsonable(positive_class),
        "random_seeds": _deepcopy(random_seeds),
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
        "final_training_instructions": _deepcopy(final_training_instructions),
        "model_artifact": None,
        "final_model_trained": False,
        "model_artifact_materialized": False,
        "model_bundle_materialized": False,
        "bundle": None,
        "readiness": _deepcopy(readiness),
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
        "operational_threshold": "unresolved",
    }


def _render_artifact(filename: str, value: Any) -> bytes:
    if filename.endswith(".json"):
        return canonical_json_text(value).encode("utf-8")
    if filename.endswith(".csv"):
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"CSV artifact must be a DataFrame: {filename}")
        return value.to_csv(index=False, lineterminator="\n").encode("utf-8")
    raise ValueError(f"Unsupported artifact extension: {filename}")


def _validate_artifact_bytes(filename: str, content: bytes) -> Any:
    if filename.endswith(".json"):
        payload = json.loads(content.decode("utf-8"))
        schema, artifact_type = _JSON_ARTIFACT_SCHEMAS[filename]
        if payload.get("schema_version") != schema:
            raise ModelSelectionHandoffError(
                f"Invalid schema_version for {filename}: {payload.get('schema_version')}"
            )
        if payload.get("artifact_type") != artifact_type:
            raise ModelSelectionHandoffError(
                f"Invalid artifact_type for {filename}: {payload.get('artifact_type')}"
            )
        _validate_paths_recursively(payload)
        return payload
    frame = pd.read_csv(pd.io.common.BytesIO(content))
    required = {"model_id", "family", "search_strategy", "parameters"}
    if not required.issubset(frame.columns):
        raise ModelSelectionHandoffError(
            f"cross-validation-results.csv is missing columns: {sorted(required-set(frame.columns))}"
        )
    return frame


def _semantic_fingerprint_value(filename: str, value: Any) -> str:
    if filename.endswith(".json"):
        return semantic_fingerprint_json(value)
    return semantic_fingerprint_csv(value)


def _load_artifact(path: Path) -> Any:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def _semantic_equivalent(filename: str, existing: Any, incoming: Any) -> bool:
    return _semantic_fingerprint_value(filename, existing) == _semantic_fingerprint_value(
        filename, incoming
    )


def write_model_selection_artifacts(
    *,
    output_directory: str | Path,
    artifacts: Mapping[str, Any],
    overwrite: bool = False,
) -> ArtifactWriteResult:
    """Atomically write, validate, fingerprint, and conflict-check six artifacts."""

    supplied = set(artifacts)
    expected = set(ARTIFACT_FILENAMES)
    if supplied != expected:
        raise ModelSelectionContractError(
            f"Artifact set mismatch. Missing={sorted(expected-supplied)}, extra={sorted(supplied-expected)}"
        )
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".model-selection-staging-", dir=output.parent))
    backup = Path(tempfile.mkdtemp(prefix=".model-selection-backup-", dir=output.parent))
    payloads = {name: _deepcopy(value) for name, value in artifacts.items()}
    promoted: list[str] = []
    backed_up: list[str] = []
    try:
        staging_output = staging / output.name
        staging_output.mkdir(parents=True, exist_ok=True)
        byte_hashes: dict[str, str] = {}
        semantic_hashes: dict[str, str] = {}
        # Render non-manifest artifacts first so their fingerprints can be embedded.
        for filename in ARTIFACT_FILENAMES[1:]:
            content = _render_artifact(filename, payloads[filename])
            parsed = _validate_artifact_bytes(filename, content)
            path = staging_output / filename
            path.write_bytes(content)
            byte_hashes[filename] = sha256_file(path)
            semantic_hashes[filename] = _semantic_fingerprint_value(filename, parsed)
        manifest = _deepcopy(payloads["model-selection-manifest.json"])
        manifest["artifact_fingerprints"] = {
            filename: {
                "byte_sha256": byte_hashes[filename],
                "semantic_sha256": semantic_hashes[filename],
            }
            for filename in ARTIFACT_FILENAMES[1:]
        }
        manifest_base = _deepcopy(manifest)
        manifest_base.pop("self_semantic_sha256", None)
        manifest["self_semantic_sha256"] = semantic_fingerprint_json(manifest_base)
        payloads["model-selection-manifest.json"] = manifest
        content = _render_artifact("model-selection-manifest.json", manifest)
        parsed_manifest = _validate_artifact_bytes(
            "model-selection-manifest.json", content
        )
        manifest_path = staging_output / "model-selection-manifest.json"
        manifest_path.write_bytes(content)
        byte_hashes["model-selection-manifest.json"] = sha256_file(manifest_path)
        semantic_hashes["model-selection-manifest.json"] = semantic_fingerprint_json(
            parsed_manifest
        )

        output.mkdir(parents=True, exist_ok=True)
        existing = {
            filename: (output / filename).is_file() for filename in ARTIFACT_FILENAMES
        }
        divergent: list[str] = []
        for filename, present in existing.items():
            if not present:
                continue
            current = _load_artifact(output / filename)
            if not _semantic_equivalent(filename, current, payloads[filename]):
                divergent.append(filename)
        if divergent and not overwrite:
            raise ArtifactConflictError(
                "Existing model-selection artifacts are semantically divergent: "
                + ", ".join(sorted(divergent))
            )
        if all(existing.values()) and not divergent:
            current_byte = {
                filename: sha256_file(output / filename)
                for filename in ARTIFACT_FILENAMES
            }
            current_semantic = {
                filename: _semantic_fingerprint_value(
                    filename, _load_artifact(output / filename)
                )
                for filename in ARTIFACT_FILENAMES
            }
            return ArtifactWriteResult(
                output_directory=output,
                created=(),
                replaced=(),
                idempotent=True,
                byte_sha256=current_byte,
                semantic_sha256=current_semantic,
            )

        for filename in ARTIFACT_FILENAMES:
            destination = output / filename
            if destination.exists():
                backup_path = backup / filename
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup_path)
                backed_up.append(filename)
            os.replace(staging_output / filename, destination)
            promoted.append(filename)

        # Revalidate the complete promoted set.
        for filename in ARTIFACT_FILENAMES:
            content = (output / filename).read_bytes()
            _validate_artifact_bytes(filename, content)
        created = tuple(filename for filename in promoted if not existing[filename])
        replaced = tuple(filename for filename in promoted if existing[filename])
        final_byte = {filename: sha256_file(output / filename) for filename in ARTIFACT_FILENAMES}
        final_semantic = {
            filename: _semantic_fingerprint_value(
                filename, _load_artifact(output / filename)
            )
            for filename in ARTIFACT_FILENAMES
        }
        return ArtifactWriteResult(
            output_directory=output,
            created=created,
            replaced=replaced,
            idempotent=False,
            byte_sha256=final_byte,
            semantic_sha256=final_semantic,
        )
    except Exception:
        for filename in reversed(promoted):
            destination = output / filename
            if destination.exists():
                destination.unlink()
        for filename in reversed(backed_up):
            backup_path = backup / filename
            destination = output / filename
            if backup_path.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup_path, destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def _validate_model_selection_handoff_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "model-selection-handoff.v1":
        raise ModelSelectionHandoffError("Invalid model-selection handoff schema.")
    if payload.get("artifact_type") != "model_selection_handoff":
        raise ModelSelectionHandoffError("Invalid model-selection handoff artifact_type.")
    if payload.get("test_partition_sealed") is not True:
        raise ModelSelectionHandoffError("The test partition must remain sealed.")
    if payload.get("test_partition_evaluated") is not False:
        raise ModelSelectionHandoffError("The test partition must not be evaluated.")
    if payload.get("final_model_trained") is not False:
        raise ModelSelectionHandoffError("The final model must not be trained in notebook 03.")
    if payload.get("model_artifact") is not None:
        raise ModelSelectionHandoffError("The model artifact must be absent.")
    if payload.get("model_bundle_materialized") is not False:
        raise ModelSelectionHandoffError("The model bundle must be absent.")
    if payload.get("operational_validity") != "unconfirmed":
        raise ModelSelectionHandoffError("Operational validity must remain unconfirmed.")
    if payload.get("operational_threshold") != "unresolved":
        raise ModelSelectionHandoffError("Operational threshold must remain unresolved.")
    readiness = payload.get("readiness", {})
    required_true = {
        "educational_model_selection_completed",
        "educational_final_candidate_selected",
        "educational_threshold_selected",
        "model_selection_handoff_ready",
        "final_model_training_ready",
    }
    if any(readiness.get(key) is not True for key in required_true):
        raise ModelSelectionHandoffError("Required educational readiness is incomplete.")
    _validate_paths_recursively(payload)


def load_and_validate_model_selection_handoff(
    *, project_root: str | Path, handoff_path: str | Path
) -> dict[str, Any]:
    """Load and revalidate the declarative handoff and sibling artifact set."""

    root = Path(project_root).resolve()
    relative = _require_relative_path(handoff_path, field="handoff_path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelSelectionHandoffError("Handoff path escapes project root.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Model-selection handoff not found: {relative}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_model_selection_handoff_payload(payload)
    directory = path.parent
    manifest_path = directory / "model-selection-manifest.json"
    if not manifest_path.is_file():
        raise ModelSelectionHandoffError("Model-selection manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_slug") != payload.get("dataset_slug"):
        raise ModelSelectionHandoffError("Dataset slug differs between manifest and handoff.")
    fingerprints = manifest.get("artifact_fingerprints", {})
    for filename in ARTIFACT_FILENAMES[1:]:
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise ModelSelectionHandoffError(f"Referenced artifact is missing: {filename}")
        expected = fingerprints.get(filename)
        if not isinstance(expected, Mapping):
            raise ModelSelectionHandoffError(f"Artifact fingerprint is missing: {filename}")
        if sha256_file(artifact_path) != expected.get("byte_sha256"):
            raise ModelSelectionHandoffError(f"Artifact byte fingerprint mismatch: {filename}")
        loaded = _load_artifact(artifact_path)
        if _semantic_fingerprint_value(filename, loaded) != expected.get(
            "semantic_sha256"
        ):
            raise ModelSelectionHandoffError(
                f"Artifact semantic fingerprint mismatch: {filename}"
            )
    return _deepcopy(payload)


__all__ = [
    "ARTIFACT_FILENAMES",
    "ArtifactConflictError",
    "ArtifactWriteResult",
    "CandidateSpecificationError",
    "FeatureRoleError",
    "ModelSelectionContractError",
    "ModelSelectionError",
    "ModelSelectionHandoffError",
    "NoEligibleCandidateError",
    "PartitionRoles",
    "SearchOutcome",
    "analyze_thresholds",
    "build_candidate_pipeline",
    "build_candidate_results",
    "build_cross_validation",
    "build_model_selection_handoff",
    "build_model_selection_manifest",
    "build_preprocessing_pipeline",
    "build_scoring_contract",
    "build_threshold_analysis",
    "build_validation_evidence",
    "canonical_json_bytes",
    "canonical_json_text",
    "compute_cv_confidence_interval",
    "compute_fbeta",
    "describe_cv_folds",
    "detect_practical_tie",
    "evaluate_candidates_on_validation",
    "evaluate_probability_classifier",
    "load_and_validate_model_selection_handoff",
    "report_unknown_categories",
    "run_model_search",
    "runtime_versions",
    "select_candidate_model",
    "semantic_fingerprint_csv",
    "semantic_fingerprint_json",
    "sha256_file",
    "summarize_search_results",
    "validate_candidate_model_specs",
    "validate_feature_partition_roles",
    "validate_model_selection_contract",
    "write_model_selection_artifacts",
]
