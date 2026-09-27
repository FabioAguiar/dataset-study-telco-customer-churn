"""Regression gates for the executed study: notebooks, runtime, artifacts, figures.

Notebook and figure checks always run because notebooks and ``docs/images`` are
versioned. Checks over runtime artifacts (``artifacts/``, ``data/processed``)
are skipped in a fresh clone where the study has not been executed yet.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import nbformat
import pytest

from scripts.finalize_model import (
    load_and_validate_final_model_handoff,
    load_and_validate_inference_bundle,
)
from scripts.runtime_contract import (
    expected_runtime_versions,
    read_lock_pins,
    read_python_version,
)
from scripts.select_models import DEFERRED_OPERATION_DISPOSITIONS
from scripts.smoke_predict import (
    validate_bundle_handoff_alignment,
    validate_model_artifact_before_load,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "notebooks/01_data_understanding_and_exploration.ipynb",
    "notebooks/02_data_preparation.ipynb",
    "notebooks/03_model_selection_and_evaluation.ipynb",
    "notebooks/04_final_model_and_bundle.ipynb",
    "notebooks/05_inference_demo.ipynb",
)
PREPARATION = "artifacts/preparation/telco-customer-churn"
MODEL_SELECTION = "artifacts/model-selection/telco-customer-churn"
MODELS = "artifacts/models/telco-customer-churn"
HANDOFF = f"{MODELS}/final-model-handoff.json"
BUNDLE = f"{MODELS}/inference-bundle.json"
PERSONAL_PATH = re.compile(r"(/home/[^/\s\"']+|/Users/[^/\s\"']+|[A-Za-z]:\\\\Users\\\\)")

requires_artifacts = pytest.mark.skipif(
    not (ROOT / HANDOFF).is_file(),
    reason="Study artifacts are runtime outputs; execute notebooks 01-05 first.",
)


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook(relative: str):
    return nbformat.read(ROOT / relative, as_version=4)


def _code_cells(relative: str):
    return [cell for cell in _notebook(relative).cells if cell.cell_type == "code"]


def _output_text(cell) -> str:
    parts = []
    for output in cell.get("outputs", []):
        if output.output_type == "stream":
            parts.append(output.get("text", ""))
        else:
            parts.append(str(output.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relative", NOTEBOOKS)
def test_notebook_was_executed_linearly_in_one_clean_kernel(relative: str) -> None:
    cells = _code_cells(relative)
    assert cells, relative
    assert all(cell.source.strip() for cell in cells), "empty code cell"
    counts = [cell.execution_count for cell in cells]
    assert counts == list(range(1, len(cells) + 1)), counts


@pytest.mark.parametrize("relative", NOTEBOOKS)
def test_notebook_outputs_have_no_errors_warnings_or_personal_paths(relative: str) -> None:
    for index, cell in enumerate(_code_cells(relative)):
        for output in cell.get("outputs", []):
            assert output.output_type != "error", (relative, index)
            if output.output_type == "stream":
                assert output.name != "stderr", (relative, index, output.text[:200])
        assert not PERSONAL_PATH.search(_output_text(cell)), (relative, index)


@pytest.mark.parametrize("relative", NOTEBOOKS)
def test_notebook_kernel_matches_the_canonical_python(relative: str) -> None:
    version = _notebook(relative).metadata.get("language_info", {}).get("version")
    assert version == read_python_version(ROOT)


def test_holdout_is_established_before_any_target_aware_analysis() -> None:
    cells = _code_cells(NOTEBOOKS[0])
    sources = [cell.source for cell in cells]
    holdout = next(i for i, s in enumerate(sources) if "build_holdout_membership(" in s)
    target_aware_calls = (
        "analyze_duplicate_records(",
        "analyze_target_distribution(",
        "analyze_numerical_features(",
        "analyze_categorical_features(",
        "analyze_feature_target_relationships(",
        "analyze_data_leakage(",
    )
    for call in target_aware_calls:
        position = next(i for i, s in enumerate(sources) if call in s)
        assert position > holdout, call
    target_cell = sources[next(i for i, s in enumerate(sources) if "analyze_target_distribution(" in s)]
    assert "dataframe=eda_df" in target_cell
    assert "SPLIT-DEC-001" in sources[holdout]


def test_downstream_notebooks_carry_the_required_contracts() -> None:
    nb02 = "\n".join(cell.source for cell in _code_cells(NOTEBOOKS[1]))
    nb03 = "\n".join(cell.source for cell in _code_cells(NOTEBOOKS[2]))
    assert "verify_partitions_match_holdout(" in nb02
    assert "acquire_kaggle_source(" in nb02
    assert "resolve_deferred_operations(" in nb03
    assert "model__penalty" not in nb03
    for relative in NOTEBOOKS[:4]:
        source = "\n".join(cell.source for cell in _code_cells(relative))
        assert "require_canonical_runtime(" in source, relative


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def test_environment_contract_files_exist_at_the_root() -> None:
    for name in (".python-version", "pyproject.toml", "pylock.toml"):
        assert (ROOT / name).is_file(), name
    for legacy in ("requirements.txt", "requirements", "environment.yml", "reproducibility"):
        assert not (ROOT / legacy).exists(), f"competing dependency source: {legacy}"


def test_pylock_is_a_generated_pep751_lock_covering_every_declared_dependency() -> None:
    import tomllib

    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    lock_text = (ROOT / "pylock.toml").read_text(encoding="utf-8")
    assert lock_text.startswith("# This file was autogenerated"), "pylock.toml must be tool-generated"
    lock = tomllib.loads(lock_text)
    assert lock["lock-version"] == "1.0"

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    python_version = read_python_version(ROOT)
    assert python_version in SpecifierSet(project["project"]["requires-python"])
    assert python_version in SpecifierSet(lock["requires-python"])

    groups = project["dependency-groups"]

    def expand(group: str) -> list[str]:
        items: list[str] = []
        for item in groups[group]:
            items.extend(expand(item["include-group"]) if isinstance(item, dict) else [item])
        return items

    declared = [*project["project"]["dependencies"], *expand("study")]
    pins = read_lock_pins(ROOT)
    for text in declared:
        requirement = Requirement(text)
        name = re.sub(r"[-_.]+", "-", requirement.name).lower()
        assert name in pins, f"{requirement.name} is declared but not locked"
        assert pins[name] in requirement.specifier, (text, pins[name])


def test_lock_pins_every_runtime_component() -> None:
    pins = read_lock_pins(ROOT)
    for distribution in ("pandas", "scikit-learn", "joblib", "numpy", "scipy", "kagglehub"):
        assert distribution in pins, distribution
    assert re.fullmatch(r"\d+\.\d+\.\d+", read_python_version(ROOT))


@requires_artifacts
def test_bundle_and_manifests_record_the_canonical_runtime() -> None:
    expected = expected_runtime_versions(ROOT)
    assert _json(BUNDLE)["runtime_version_requirements"] == expected
    assert _json(f"{MODELS}/final-model-manifest.json")["runtime_versions"] == expected
    for relative in (
        f"{PREPARATION}/preparation-manifest.json",
        f"{MODEL_SELECTION}/model-selection-manifest.json",
    ):
        recorded = _json(relative)["runtime_versions"]
        for component in ("python", "pandas", "scikit_learn"):
            assert recorded[component] == expected[component], (relative, component)


@requires_artifacts
def test_readme_documents_the_recorded_runtime() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for component, version in expected_runtime_versions(ROOT).items():
        assert f"| {version} |" in readme, (component, version)


# ---------------------------------------------------------------------------
# Artifact chain
# ---------------------------------------------------------------------------


@requires_artifacts
def test_final_artifacts_form_one_consistent_chain() -> None:
    handoff = load_and_validate_final_model_handoff(project_root=ROOT, handoff_path=HANDOFF)
    bundle = load_and_validate_inference_bundle(project_root=ROOT, bundle_path=BUNDLE)
    manifest = _json(f"{MODELS}/final-model-manifest.json")
    validate_bundle_handoff_alignment(handoff, bundle, manifest=manifest)
    model_path = validate_model_artifact_before_load(
        project_root=ROOT, bundle=bundle, handoff=handoff, manifest=manifest
    )
    assert _sha256(model_path) == manifest["model_artifact_byte_sha256"]

    selection = _json(f"{MODEL_SELECTION}/model-selection-handoff.json")
    assert handoff["selected_model_id"] == selection["selected_model_id"]
    assert handoff["educational_threshold"] == selection["selected_educational_threshold"]["threshold"]
    assert handoff["test_partition_evaluation_count"] == 1
    assert handoff["test_partition_used_for_model_selection"] is False
    assert handoff["test_partition_used_for_threshold_selection"] is False


@requires_artifacts
def test_split_manifest_reproduces_the_notebook_01_holdout() -> None:
    split = _json(f"{PREPARATION}/split-manifest.json")
    holdout_path = ROOT / PREPARATION / "holdout-membership.json"
    reference = split["holdout_establishment"]
    assert reference["membership_reproduced"] is True
    assert reference["sha256"] == _sha256(holdout_path)
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    assert holdout["exploration_partition"] == "train"
    for name in ("train", "validation", "test"):
        assert holdout["partitions"][name]["row_count"] == split["row_counts"][name]
        assert (
            holdout["partitions"][name]["identifiers_sha256"]
            == reference["identifier_sha256"][name]
        )


@requires_artifacts
def test_every_deferred_operation_has_a_final_disposition() -> None:
    deferred = _json(f"{PREPARATION}/feature-manifest.json")["preprocessing_contract"][
        "deferred_operations"
    ]
    records = _json(f"{MODEL_SELECTION}/model-selection-manifest.json")[
        "deferred_operation_dispositions"
    ]
    assert [row["operation"] for row in records] == deferred
    assert all(row["disposition"] in DEFERRED_OPERATION_DISPOSITIONS for row in records)
    assert all(row["rationale"].strip() for row in records)


@requires_artifacts
def test_saved_notebook_outputs_match_the_current_artifacts() -> None:
    selection = _json(f"{MODEL_SELECTION}/model-selection-handoff.json")
    nb03 = "\n".join(_output_text(cell) for cell in _code_cells(NOTEBOOKS[2]))
    assert str(selection["selected_educational_threshold"]["threshold"]) in nb03
    assert selection["selected_model_id"] in nb03

    manifest = _json(f"{MODELS}/final-model-manifest.json")
    nb04 = "\n".join(cell.source + _output_text(cell) for cell in _code_cells(NOTEBOOKS[3]))
    assert "Idempotent reuse" not in "\n".join(_output_text(c) for c in _code_cells(NOTEBOOKS[3]))
    assert manifest["selected_model_id"] in nb04


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_visual_evidence_index_matches_the_image_directory_exactly() -> None:
    index = _json("docs/images/visual-evidence-index.json")
    on_disk = {path.name: path for path in (ROOT / "docs/images").glob("*.png")}
    indexed = {Path(item["relative_asset_path"]).name: item for item in index["visuals"]}

    assert index["excluded_assets"] == []
    assert set(indexed) == set(on_disk)
    assert index["total_assets_scanned"] == index["total_assets_indexed"] == len(on_disk)
    for name, item in indexed.items():
        assert item["sha256"] == _sha256(on_disk[name]), name
    assert "train partition" in index["data_scope"]


# ---------------------------------------------------------------------------
# Smoke inference over the generated model
# ---------------------------------------------------------------------------


@requires_artifacts
def test_smoke_inference_loads_the_generated_model_through_every_gate() -> None:
    from scripts.runtime_contract import observed_runtime_mismatches
    from scripts.smoke_predict import load_validated_inference_pipeline, predict_educational

    if observed_runtime_mismatches(ROOT):
        pytest.skip("Deserialization requires the canonical runtime (pylock.toml).")

    pipeline, handoff, bundle, runtime_report = load_validated_inference_pipeline(
        project_root=ROOT,
        handoff_path=HANDOFF,
        bundle_path=BUNDLE,
        trusted_source=True,
    )
    synthetic = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.40,
        "TotalCharges": 1569.60,
    }

    result = predict_educational(
        pipeline, synthetic, bundle=bundle, runtime_report=runtime_report
    )

    assert runtime_report.compatible
    assert 0.0 <= result["positive_class_probability"] <= 1.0
    assert result["operational_prediction_available"] is False
    assert result["model_state_fingerprint"] == bundle["model_state_fingerprint"]
