"""Top-level external evidence discovery index for the Telco study.

This module inventories the already-existing (and newly materialized)
structured evidence artifacts of this project, maps each to a logical role,
and records type/version/path/hash references so that an external reader
(Lumen, during Atlas dataset-integration authoring) can discover them
without opening every artifact individually.

This index is explicitly NOT:
  - an Atlas runtime manifest;
  - an Atlas release manifest;
  - an Atlas dataset-integration-authoring-manifest;
  - an external handoff runtime dependency.

It never copies full artifact payloads; it only carries relative paths,
declared type/version strings, and content hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


EXTERNAL_EVIDENCE_INDEX_CONTRACT_VERSION: Final[str] = "external-evidence-index.v1"
DATASET_SLUG: Final[str] = "telco-customer-churn"
LOGICAL_PRODUCER_PROJECT_ID: Final[str] = "dataset-study-telco-customer-churn"
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[str] = (
    "artifacts/telco-customer-churn/external-evidence-index.json"
)

# Each entry declares a logical role, its project-relative artifact path, and
# the logical roles it is known to depend on (used only as a lightweight
# discovery hint via ``input_references``; this index never copies payloads).
_INVENTORY: Final[tuple[dict[str, Any], ...]] = (
    {
        "logical_role": "source_identity_and_preparation",
        "relative_path": (
            "artifacts/preparation/telco-customer-churn/preparation-manifest.json"
        ),
        "depends_on": (),
    },
    {
        "logical_role": "feature_semantics",
        "relative_path": (
            "artifacts/preparation/telco-customer-churn/feature-manifest.json"
        ),
        "depends_on": (),
    },
    {
        "logical_role": "split_identity",
        "relative_path": (
            "artifacts/preparation/telco-customer-churn/split-manifest.json"
        ),
        "depends_on": ("source_identity_and_preparation",),
    },
    {
        "logical_role": "data_quality_mechanical",
        "relative_path": (
            "artifacts/preparation/telco-customer-churn/quality-evidence.json"
        ),
        "depends_on": ("source_identity_and_preparation", "split_identity"),
    },
    {
        "logical_role": "data_quality_qualitative",
        "relative_path": (
            "artifacts/preparation/telco-customer-churn/data-quality-findings.json"
        ),
        "depends_on": ("data_quality_mechanical",),
    },
    {
        "logical_role": "leakage",
        "relative_path": (
            "artifacts/preparation/telco-customer-churn/leakage-evidence.json"
        ),
        "depends_on": ("feature_semantics", "split_identity"),
    },
    {
        "logical_role": "exploratory_analysis",
        "relative_path": (
            "artifacts/preparation/telco-customer-churn/"
            "exploratory-analysis-evidence.json"
        ),
        "depends_on": ("feature_semantics",),
    },
    {
        "logical_role": "model_selection",
        "relative_path": (
            "artifacts/model-selection/telco-customer-churn/"
            "model-selection-manifest.json"
        ),
        "depends_on": ("split_identity",),
    },
    {
        "logical_role": "model_selection_candidates",
        "relative_path": (
            "artifacts/model-selection/telco-customer-churn/candidate-results.json"
        ),
        "depends_on": ("model_selection",),
    },
    {
        "logical_role": "validation",
        "relative_path": (
            "artifacts/model-selection/telco-customer-churn/validation-evidence.json"
        ),
        "depends_on": ("model_selection",),
    },
    {
        "logical_role": "threshold",
        "relative_path": (
            "artifacts/model-selection/telco-customer-churn/threshold-analysis.json"
        ),
        "depends_on": ("model_selection",),
    },
    {
        "logical_role": "model_selection_handoff",
        "relative_path": (
            "artifacts/model-selection/telco-customer-churn/"
            "model-selection-handoff.json"
        ),
        "depends_on": ("model_selection", "threshold"),
    },
    {
        "logical_role": "final_model",
        "relative_path": (
            "artifacts/models/telco-customer-churn/final-model-manifest.json"
        ),
        "depends_on": ("model_selection_handoff",),
    },
    {
        "logical_role": "final_test",
        "relative_path": (
            "artifacts/models/telco-customer-churn/final-test-evidence.json"
        ),
        "depends_on": ("final_model", "split_identity"),
    },
    {
        "logical_role": "inference_semantics",
        "relative_path": (
            "artifacts/models/telco-customer-churn/inference-bundle.json"
        ),
        "depends_on": ("final_model",),
    },
    {
        "logical_role": "final_model_handoff",
        "relative_path": (
            "artifacts/models/telco-customer-churn/final-model-handoff.json"
        ),
        "depends_on": ("final_model", "inference_semantics"),
    },
    {
        "logical_role": "readiness_and_limitations",
        "relative_path": (
            "artifacts/readiness/telco-customer-churn/"
            "readiness-and-limitations.json"
        ),
        "depends_on": (
            "data_quality_mechanical",
            "model_selection",
            "final_model",
            "final_test",
            "inference_semantics",
        ),
    },
    {
        "logical_role": "visual_evidence",
        "relative_path": "docs/images/visual-evidence-index.json",
        "depends_on": (),
    },
)


class ExternalEvidenceIndexError(ValueError):
    """Raised when an inventory path or provenance argument is invalid."""


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_artifact(root: Path, relative_path: str) -> dict[str, Any]:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ExternalEvidenceIndexError(
            f"Invalid project-relative path: {relative_path}"
        )
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ExternalEvidenceIndexError(
            f"Artifact path escapes project root: {relative_path}"
        ) from exc

    if not path.is_file():
        return {
            "status": "missing",
            "artifact_type": None,
            "artifact_version": None,
            "sha256": None,
        }

    payload: Any = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = None

    artifact_type = payload.get("artifact_type") if isinstance(payload, dict) else None
    artifact_version = (
        payload.get("schema_version") if isinstance(payload, dict) else None
    )

    return {
        "status": "present",
        "artifact_type": artifact_type,
        "artifact_version": artifact_version,
        "sha256": _sha256_file(path),
    }


def build_external_evidence_index(
    *,
    project_root: str | Path,
    dataset_slug: str = DATASET_SLUG,
    repository_revision: str | None = None,
    repository_revision_unavailable_reason: str | None = None,
    generation_timestamp: str | None = None,
    contract_version: str = EXTERNAL_EVIDENCE_INDEX_CONTRACT_VERSION,
) -> dict[str, Any]:
    """Build the top-level external evidence discovery index.

    This performs no analysis and copies no artifact payloads: it inventories
    already-existing (or already-materialized) evidence artifacts by logical
    role, records their declared type/version, a project-relative path, and
    a real SHA-256 of the file bytes when present.
    """
    if repository_revision is None and not repository_revision_unavailable_reason:
        raise ExternalEvidenceIndexError(
            "repository_revision_unavailable_reason is required when "
            "repository_revision is not supplied."
        )

    root = Path(project_root).expanduser().resolve()
    entries: list[dict[str, Any]] = []
    for item in _INVENTORY:
        inspected = _inspect_artifact(root, item["relative_path"])
        entries.append(
            {
                "logical_role": item["logical_role"],
                "relative_path": item["relative_path"],
                "artifact_type": inspected["artifact_type"],
                "artifact_version": inspected["artifact_version"],
                "status": inspected["status"],
                "sha256": inspected["sha256"],
                "input_references": list(item["depends_on"]),
            }
        )

    missing_artifacts = [
        entry["logical_role"] for entry in entries if entry["status"] == "missing"
    ]

    return {
        "schema_version": EXTERNAL_EVIDENCE_INDEX_CONTRACT_VERSION,
        "artifact_type": "external_evidence_index",
        "contract_version": contract_version,
        "dataset_slug": dataset_slug,
        "index_kind": "external_authoring_time_evidence_discovery_index",
        "not_an_atlas_runtime_manifest": True,
        "not_an_atlas_release_manifest": True,
        "not_a_dataset_integration_authoring_manifest": True,
        "not_an_external_handoff_runtime_dependency": True,
        "provenance": {
            "logical_producer_project_id": LOGICAL_PRODUCER_PROJECT_ID,
            "repository_revision": repository_revision,
            "repository_revision_unavailable_reason": (
                None
                if repository_revision is not None
                else repository_revision_unavailable_reason
            ),
            "repository_revision_note": (
                "Reflects this project's repository revision at "
                "index-generation time. artifacts/**/*.json and data/ are "
                "gitignored, untracked, local-only outputs; this revision "
                "is provenance context only, not a claim that those "
                "generated artifacts historically belong to this exact "
                "commit."
            ),
            "generation_timestamp": generation_timestamp,
        },
        "evidence_inventory": entries,
        "missing_artifacts": missing_artifacts,
    }


def write_external_evidence_index(
    index: dict[str, Any],
    *,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _current_repository_revision(root: Path) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git rev-parse HEAD failed: {exc}"
    revision = completed.stdout.strip()
    if not revision:
        return None, "git rev-parse HEAD returned an empty revision."
    return revision, None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the top-level external-evidence discovery index over "
            "already-existing and already-materialized Telco evidence "
            "artifacts."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root directory (default: repository root).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Project-relative output path (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()

    revision, reason = _current_repository_revision(root)
    index = build_external_evidence_index(
        project_root=root,
        repository_revision=revision,
        repository_revision_unavailable_reason=reason,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    destination = write_external_evidence_index(index, output_path=root / args.output)

    print(
        f"Indexed {len(index['evidence_inventory']) - len(index['missing_artifacts'])} "
        f"of {len(index['evidence_inventory'])} evidence artifacts."
    )
    if index["missing_artifacts"]:
        print(f"Missing: {index['missing_artifacts']}")
    print(f"Wrote: {destination.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
