from pathlib import Path

import pytest

from scripts.download_data import (
    DatasetAcquisition,
    discover_dataset_files,
    resolve_project_path,
)


def test_discover_dataset_files_excludes_hidden_metadata(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / "raw" / "sample"
    hidden_dir = dataset_dir / ".complete"
    hidden_dir.mkdir(parents=True)
    (dataset_dir / "dataset.csv").write_text("a\n1\n", encoding="utf-8")
    (hidden_dir / "bundle.complete").write_text("ok", encoding="utf-8")

    files = discover_dataset_files(dataset_dir)

    assert [path.name for path in files] == ["dataset.csv"]


def test_require_one_file_uses_explicit_selector(tmp_path: Path) -> None:
    project = tmp_path / "study"
    destination = project / "data" / "raw" / "sample"
    destination.mkdir(parents=True)
    source_file = destination / "dataset.csv"
    source_file.write_text("a\n1\n", encoding="utf-8")

    acquisition = DatasetAcquisition(
        source_kind="kaggle",
        source_reference="owner/sample",
        destination=destination,
        resolved_path=destination,
        files=(source_file,),
        project_root=project,
    )

    assert acquisition.require_one_file("dataset.csv") == source_file
    assert acquisition.display_destination == "data/raw/sample"


def test_resolve_project_path_rejects_outside_destination(
    tmp_path: Path,
) -> None:
    project = tmp_path / "study"
    project.mkdir()

    with pytest.raises(ValueError, match="inside the project root"):
        resolve_project_path("../outside", project_root=project)
