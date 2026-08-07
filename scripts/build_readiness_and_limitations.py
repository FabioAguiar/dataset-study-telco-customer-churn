"""Consolidated, structured readiness-and-limitations artifact.

This module performs no new analysis. It transcribes the already-agreed
``README.md`` "Limitations" and "Current readiness" narrative into
structured JSON and points at (rather than copies) the readiness
sub-objects already present in the preparation, model-selection, and model
artifact families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final, Mapping, Sequence


READINESS_AND_LIMITATIONS_CONTRACT_VERSION: Final[str] = (
    "readiness-and-limitations.v1"
)
DATASET_SLUG: Final[str] = "telco-customer-churn"
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[str] = (
    "artifacts/readiness/telco-customer-churn/readiness-and-limitations.json"
)

# Transcribed verbatim from README.md's "## Limitations" section (10 bullets).
TELCO_LIMITATIONS: Final[tuple[dict[str, Any], ...]] = (
    {
        "limitation_id": "LIM-001",
        "statement": (
            "The evaluation uses a stratified random snapshot, not a "
            "temporal holdout."
        ),
        "category": "temporal",
        "blocking": True,
    },
    {
        "limitation_id": "LIM-002",
        "statement": "Associations in the exploratory analysis are not causal effects.",
        "category": "interpretation",
        "blocking": False,
    },
    {
        "limitation_id": "LIM-003",
        "statement": (
            "Production-time feature availability and latency are "
            "unconfirmed."
        ),
        "category": "operational",
        "blocking": True,
    },
    {
        "limitation_id": "LIM-004",
        "statement": "The educational threshold is not a business decision policy.",
        "category": "modeling",
        "blocking": False,
    },
    {
        "limitation_id": "LIM-005",
        "statement": (
            "False-positive and false-negative business costs are "
            "unavailable."
        ),
        "category": "operational",
        "blocking": True,
    },
    {
        "limitation_id": "LIM-006",
        "statement": (
            "No intervention-uplift or retention-effectiveness study was "
            "performed."
        ),
        "category": "scope",
        "blocking": True,
    },
    {
        "limitation_id": "LIM-007",
        "statement": (
            "No subgroup fairness or stability assessment was established "
            "for deployment."
        ),
        "category": "governance",
        "blocking": True,
    },
    {
        "limitation_id": "LIM-008",
        "statement": "No drift monitoring or scheduled retraining policy exists.",
        "category": "operational",
        "blocking": True,
    },
    {
        "limitation_id": "LIM-009",
        "statement": (
            "`api/` is a reserved scaffold and does not provide an "
            "implemented endpoint."
        ),
        "category": "implementation",
        "blocking": True,
    },
    {
        "limitation_id": "LIM-010",
        "statement": (
            "Predictions are educational and must not drive automated "
            "customer decisions."
        ),
        "category": "governance",
        "blocking": True,
    },
)

TELCO_KNOWN_UNSUPPORTED_USES: Final[tuple[str, ...]] = (
    "Automated customer-retention decisions or interventions.",
    "Any production or operational prediction serving (api/ is an "
    "unimplemented reserved scaffold).",
    "Causal claims about churn drivers.",
    "Deployment without a defined temporal/inference-time contract.",
)


def _default_readiness_pointers() -> dict[str, Any]:
    limitation_ids = [str(item["limitation_id"]) for item in TELCO_LIMITATIONS]
    return {
        "analysis_completeness": {
            "status": "completed",
            "evidence_ref": {
                "artifact_path": (
                    "artifacts/preparation/telco-customer-churn/"
                    "quality-evidence.json"
                ),
                "field": "readiness.deterministic_preparation_ready",
            },
            "note": (
                "Dataset understanding, EDA, and deterministic preparation "
                "are complete."
            ),
        },
        "educational_research_scope": {
            "status": "educational_benchmark_only",
            "evidence_ref": {
                "artifact_path": (
                    "artifacts/preparation/telco-customer-churn/"
                    "split-manifest.json"
                ),
                "field": "purpose",
            },
            "note": (
                "purpose='educational_benchmark'; this is not an "
                "operational deployment study."
            ),
        },
        "modeling_completeness": {
            "status": "completed",
            "evidence_ref": {
                "artifact_path": (
                    "artifacts/model-selection/telco-customer-churn/"
                    "model-selection-manifest.json"
                ),
                "field": "readiness.educational_model_selection_completed",
            },
            "note": (
                "Educational model selection and final model training are "
                "complete."
            ),
        },
        "final_test_status": {
            "status": "completed_single_sealed_evaluation",
            "evidence_ref": {
                "artifact_path": (
                    "artifacts/models/telco-customer-churn/"
                    "final-test-evidence.json"
                ),
                "field": "test_probability_evaluation_count",
            },
            "note": (
                "All five test_used_for_* flags are false; the sealed test "
                "partition was evaluated exactly once, after the final fit."
            ),
        },
        "inference_demo_status": {
            "status": "completed_in_recorded_compatible_runtime",
            "evidence_ref": {
                "artifact_path": (
                    "artifacts/models/telco-customer-churn/"
                    "inference-bundle.json"
                ),
                "field": "readiness.educational_inference_demo_ready",
            },
            "note": (
                "notebooks/05_inference_demo.ipynb demonstrates inference "
                "under the pinned runtime; no API endpoint is implemented."
            ),
        },
        "operational_deployment_readiness": False,
        "operational_deployment_readiness_ref": {
            "artifact_path": (
                "artifacts/models/telco-customer-churn/inference-bundle.json"
            ),
            "field": "operational_validity",
            "observed_value": "unconfirmed",
        },
        "known_limitations_ref": {
            "limitation_ids": limitation_ids,
        },
        "known_unsupported_uses": list(TELCO_KNOWN_UNSUPPORTED_USES),
        "remaining_blockers": [
            {
                "blocker": "Inference-time temporal contract unresolved.",
                "evidence_ref": {
                    "artifact_path": (
                        "artifacts/preparation/telco-customer-churn/"
                        "preparation-manifest.json"
                    ),
                    "field": "readiness.temporal_contract_status",
                },
            },
            {
                "blocker": "Feature inference-time availability unconfirmed.",
                "evidence_ref": {
                    "artifact_path": (
                        "artifacts/models/telco-customer-churn/"
                        "inference-bundle.json"
                    ),
                    "field": "feature_inference_availability",
                },
            },
            {
                "blocker": "Operational threshold unresolved.",
                "evidence_ref": {
                    "artifact_path": (
                        "artifacts/models/telco-customer-churn/"
                        "final-model-manifest.json"
                    ),
                    "field": "operational_threshold",
                },
            },
            {
                "blocker": "API implementation is not present.",
                "evidence_ref": {
                    "artifact_path": "api/README.md",
                    "field": "not_applicable_empty_reserved_scaffold",
                },
            },
        ],
    }


def build_readiness_and_limitations(
    *,
    dataset_slug: str = DATASET_SLUG,
    limitations: Sequence[Mapping[str, Any]] = TELCO_LIMITATIONS,
    readiness: Mapping[str, Any] | None = None,
    contract_version: str = READINESS_AND_LIMITATIONS_CONTRACT_VERSION,
) -> dict[str, Any]:
    """Build the structured readiness-and-limitations artifact.

    ``readiness`` and ``limitations`` default to the transcription of the
    current ``README.md`` content; callers may override them explicitly
    (used by tests) without changing the artifact's shape. Limitations and
    readiness remain separate, multi-field concepts and are never collapsed
    into a single boolean.
    """
    limitation_rows = [dict(item) for item in limitations]
    resolved_readiness = (
        dict(_default_readiness_pointers()) if readiness is None else dict(readiness)
    )
    if "operational_deployment_readiness" not in resolved_readiness:
        raise ValueError(
            "readiness must declare operational_deployment_readiness explicitly."
        )
    if resolved_readiness["operational_deployment_readiness"] is not False:
        raise ValueError(
            "operational_deployment_readiness must not be elevated to True "
            "by this artifact; the project declares no operational "
            "prediction capability."
        )

    return {
        "schema_version": READINESS_AND_LIMITATIONS_CONTRACT_VERSION,
        "artifact_type": "readiness_and_limitations",
        "contract_version": contract_version,
        "dataset_slug": dataset_slug,
        "source_narrative_reference": {
            "artifact_path": "README.md",
            "sections": ["Limitations", "Current readiness"],
        },
        "limitations": limitation_rows,
        "readiness": resolved_readiness,
    }


def write_readiness_and_limitations(
    artifact: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the consolidated readiness-and-limitations artifact "
            "from the current README narrative and existing per-artifact "
            "readiness sub-objects."
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

    artifact = build_readiness_and_limitations()
    destination = write_readiness_and_limitations(
        artifact, output_path=root / args.output
    )

    print(f"Wrote: {destination.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
