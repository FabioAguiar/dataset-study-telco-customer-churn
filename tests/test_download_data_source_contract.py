"""Tests for the pinned source contract and acquisition integrity gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.download_data as dd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTENT = b"customerID,Churn\nA,No\nB,Yes\n"
FILENAME = "dataset.csv"


def _payload(**changes):
    payload = {
        "schema_version": "source-contract.v1",
        "dataset_slug": "sample",
        "provider": "kaggle",
        "handle": "owner/sample",
        "version": 3,
        "versioned_handle": "owner/sample/versions/3",
        "destination": "data/raw/sample",
        "files": [
            {
                "filename": FILENAME,
                "sha256": hashlib.sha256(CONTENT).hexdigest(),
                "size_bytes": len(CONTENT),
            }
        ],
    }
    payload.update(changes)
    return payload


def _project(tmp_path: Path, content: bytes = CONTENT) -> Path:
    project = tmp_path / "study"
    destination = project / "data" / "raw" / "sample"
    destination.mkdir(parents=True)
    (destination / FILENAME).write_bytes(content)
    return project


def test_repository_source_contract_pins_an_immutable_kaggle_version() -> None:
    contract = dd.load_source_contract(PROJECT_ROOT / "contracts" / "source.json")

    assert contract.handle == "blastchar/telco-customer-churn"
    assert contract.version == 1
    assert contract.versioned_handle == "blastchar/telco-customer-churn/versions/1"
    expected = contract.file("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    assert expected.sha256 == (
        "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
    )
    assert expected.size_bytes == 977501


def test_verify_accepts_the_expected_file(tmp_path: Path) -> None:
    project = _project(tmp_path)
    contract = dd.parse_source_contract(_payload())

    verified = dd.verify_acquired_files(project / "data/raw/sample", contract)

    assert [path.name for path in verified] == [FILENAME]


def test_verify_rejects_a_checksum_mismatch_with_a_clear_error(tmp_path: Path) -> None:
    tampered = CONTENT.replace(b"Yes", b"Nop")
    assert len(tampered) == len(CONTENT)
    project = _project(tmp_path, tampered)
    contract = dd.parse_source_contract(_payload())

    with pytest.raises(dd.DatasetIntegrityError, match="SHA-256 mismatch"):
        dd.verify_acquired_files(project / "data/raw/sample", contract)


def test_verify_rejects_a_size_mismatch(tmp_path: Path) -> None:
    project = _project(tmp_path, CONTENT + b"C,No\n")
    contract = dd.parse_source_contract(_payload())

    with pytest.raises(dd.DatasetIntegrityError, match="Size mismatch"):
        dd.verify_acquired_files(project / "data/raw/sample", contract)


def test_verify_rejects_unexpected_files(tmp_path: Path) -> None:
    project = _project(tmp_path)
    (project / "data/raw/sample/extra.csv").write_text("x\n", encoding="utf-8")
    contract = dd.parse_source_contract(_payload())

    with pytest.raises(dd.DatasetIntegrityError, match="Unexpected files"):
        dd.verify_acquired_files(project / "data/raw/sample", contract)


def test_verify_rejects_a_missing_expected_file(tmp_path: Path) -> None:
    project = _project(tmp_path)
    (project / "data/raw/sample" / FILENAME).rename(
        project / "data/raw/sample/renamed.csv"
    )
    contract = dd.parse_source_contract(_payload())

    with pytest.raises(dd.DatasetIntegrityError, match="Unexpected files"):
        dd.verify_acquired_files(project / "data/raw/sample", contract)
    with pytest.raises(dd.DatasetIntegrityError, match="missing"):
        dd.verify_acquired_files(
            project / "data/raw/sample", contract, allow_unexpected_files=True
        )


def test_verify_ignores_hidden_acquisition_markers(tmp_path: Path) -> None:
    project = _project(tmp_path)
    marker = project / "data/raw/sample/.complete/datasets/owner/sample/3"
    marker.mkdir(parents=True)
    (marker / "bundle.complete").write_text("", encoding="utf-8")
    contract = dd.parse_source_contract(_payload())

    assert dd.verify_acquired_files(project / "data/raw/sample", contract)


def test_acquire_reuses_a_verified_local_copy_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    contract = dd.parse_source_contract(_payload())

    def fail_download(**_kwargs):
        raise AssertionError("network acquisition must not run")

    monkeypatch.setattr(dd, "download_kaggle_dataset", fail_download)
    acquisition = dd.acquire_kaggle_source(contract, project_root=project)

    assert acquisition.source_reference == "owner/sample/versions/3"
    assert acquisition.relative_files == ("data/raw/sample/dataset.csv",)


def test_acquire_never_silently_replaces_a_divergent_local_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, CONTENT.replace(b"Yes", b"Nop"))
    contract = dd.parse_source_contract(_payload())
    monkeypatch.setattr(
        dd,
        "download_kaggle_dataset",
        lambda **_kwargs: pytest.fail("download must not run without force"),
    )

    with pytest.raises(dd.DatasetIntegrityError):
        dd.acquire_kaggle_source(contract, project_root=project)


def test_acquire_downloads_the_versioned_handle_and_verifies_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "study"
    project.mkdir()
    contract = dd.parse_source_contract(_payload())
    requested: list[str] = []

    def fake_download(*, handle, destination, **_kwargs):
        requested.append(handle)
        Path(destination).mkdir(parents=True, exist_ok=True)
        (Path(destination) / FILENAME).write_bytes(b"different content!!!!!!!!!")
        return Path(destination)

    monkeypatch.setattr(dd, "download_kaggle_dataset", fake_download)

    with pytest.raises(dd.DatasetIntegrityError):
        dd.acquire_kaggle_source(contract, project_root=project)
    assert requested == ["owner/sample/versions/3"]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"schema_version": "source-contract.v0"}, "schema"),
        ({"version": 0}, "version"),
        ({"versioned_handle": "owner/sample/versions/4"}, "versioned_handle"),
        ({"handle": "owner/sample/versions/3"}, "must not embed"),
        ({"files": []}, "non-empty"),
        ({"files": [{"filename": FILENAME, "sha256": "abc", "size_bytes": 1}]}, "SHA-256"),
        ({"destination": "/abs/path"}, "project-relative"),
    ],
)
def test_malformed_source_contracts_are_rejected(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        dd.parse_source_contract(_payload(**changes))


def test_kaggle_handle_accepts_versions_and_rejects_malformed_values(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="owner/dataset"):
        dd.download_kaggle_dataset("owner", tmp_path / "x", project_root=tmp_path)
    with pytest.raises(ValueError, match="owner/dataset"):
        dd.download_kaggle_dataset(
            "owner/sample/versions/x", tmp_path / "x", project_root=tmp_path
        )


def test_cli_source_command_reports_integrity_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _project(tmp_path, CONTENT.replace(b"Yes", b"Nop"))
    contract_path = project / "contracts" / "source.json"
    contract_path.parent.mkdir()
    contract_path.write_text(json.dumps(_payload()), encoding="utf-8")

    exit_code = dd.main(
        ["source", "contracts/source.json", "--project-root", str(project)]
    )

    assert exit_code == 1
    assert "SHA-256 mismatch" in capsys.readouterr().err
