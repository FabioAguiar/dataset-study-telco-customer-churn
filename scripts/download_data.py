"""Reusable dataset acquisition helpers for dataset-study repositories.

This module keeps operational download logic outside notebooks while leaving
study-specific choices visible in the notebook, such as the source identifier,
destination directory, and selected data file.

Supported sources
-----------------
- Kaggle datasets through ``kagglehub``;
- direct HTTP, HTTPS, or FTP file URLs through Python's standard library.

Recommended notebook usage
--------------------------
Import the high-level acquisition function after the project root has been
added to ``sys.path``::

    from pathlib import Path
    from scripts.download_data import acquire_kaggle_dataset

    DATASET_HANDLE = "blastchar/telco-customer-churn"
    RAW_DATA_RELATIVE_DIR = Path("data/raw/telco-customer-churn")

    acquisition = acquire_kaggle_dataset(
        handle=DATASET_HANDLE,
        destination=RAW_DATA_RELATIVE_DIR,
    )

    RAW_DATA_DIR = acquisition.destination

    print(f"Dataset source: {acquisition.source_reference}")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    for file_path in acquisition.relative_files:
        print(f"- {file_path}")

The command-line interface remains available::

    python scripts/download_data.py kaggle \
        blastchar/telco-customer-churn \
        --destination data/raw/telco-customer-churn

    python scripts/download_data.py url \
        "ftp://example.org/path/dataset.csv" \
        --destination data/raw/my-dataset
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
DEFAULT_KAGGLE_HANDLE: Final[str] = "blastchar/telco-customer-churn"
DEFAULT_TELCO_DESTINATION: Final[Path] = (
    DEFAULT_RAW_DATA_DIR / "telco-customer-churn"
)
SUPPORTED_URL_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https", "ftp"})
DEFAULT_CHUNK_SIZE: Final[int] = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS: Final[int] = 120

SourceKind = Literal["kaggle", "url"]


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded or validated."""


@dataclass(frozen=True, slots=True)
class DatasetAcquisition:
    """Describe the materialized result of one dataset acquisition.

    Attributes
    ----------
    source_kind:
        Acquisition mechanism: ``kaggle`` or ``url``.
    source_reference:
        Kaggle handle or direct source URL used for the acquisition.
    destination:
        Absolute directory in which the raw source files are stored.
    resolved_path:
        Path returned by the underlying acquisition implementation.
    files:
        Absolute visible files found in the destination. Internal hidden files,
        such as ``kagglehub`` completion markers, are excluded.
    project_root:
        Repository root used to resolve relative destinations.
    """

    source_kind: SourceKind
    source_reference: str
    destination: Path
    resolved_path: Path
    files: tuple[Path, ...]
    project_root: Path = PROJECT_ROOT

    @property
    def relative_files(self) -> tuple[str, ...]:
        """Return project-relative POSIX paths when possible."""
        relative: list[str] = []

        for file_path in self.files:
            try:
                display_path = file_path.relative_to(self.project_root)
            except ValueError:
                display_path = file_path
            relative.append(display_path.as_posix())

        return tuple(relative)

    def require_files(self, pattern: str) -> tuple[Path, ...]:
        """Return files matching a glob pattern or raise a clear error."""
        matches = tuple(
            file_path
            for file_path in self.files
            if file_path.match(pattern) or file_path.name == pattern
        )

        if not matches:
            raise FileNotFoundError(
                f"No acquired file matches '{pattern}' in {self.destination}. "
                f"Available files: {list(self.relative_files)}"
            )

        return matches

    def require_one_file(self, pattern: str) -> Path:
        """Return exactly one matching file or raise a clear error."""
        matches = self.require_files(pattern)

        if len(matches) != 1:
            rendered = [path.as_posix() for path in matches]
            raise RuntimeError(
                f"Expected exactly one file matching '{pattern}', "
                f"found {len(matches)}: {rendered}"
            )

        return matches[0]


def resolve_project_path(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolve a path relative to the repository root when needed."""
    root = Path(project_root).expanduser().resolve()
    candidate = Path(path).expanduser()

    if not candidate.is_absolute():
        candidate = root / candidate

    return candidate.resolve()


def discover_dataset_files(
    directory: str | Path,
    *,
    include_hidden: bool = False,
) -> tuple[Path, ...]:
    """Return deterministic recursive file paths from a dataset directory.

    Hidden files and files under hidden directories are excluded by default.
    This keeps implementation metadata such as ``.complete`` out of notebook
    acquisition summaries.
    """
    root = Path(directory).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative_parts = path.relative_to(root).parts
        if not include_hidden and any(part.startswith(".") for part in relative_parts):
            continue

        files.append(path.resolve())

    return tuple(sorted(files, key=lambda item: item.as_posix()))


def _validate_filename(filename: str) -> str:
    """Reject empty or path-like filenames supplied for direct downloads."""
    candidate = filename.strip()

    if not candidate or candidate in {".", ".."}:
        raise ValueError("The destination filename cannot be empty.")

    if Path(candidate).name != candidate:
        raise ValueError("The destination filename must not contain directories.")

    return candidate


def calculate_sha256(file_path: str | Path) -> str:
    """Return the SHA-256 digest for a local file."""
    path = Path(file_path)
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while chunk := source.read(DEFAULT_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def _verify_sha256(file_path: Path, expected_sha256: str | None) -> None:
    """Validate a file checksum when an expected digest was supplied."""
    if expected_sha256 is None:
        return

    expected = expected_sha256.strip().lower()

    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters.")

    observed = calculate_sha256(file_path)

    if observed != expected:
        raise DatasetDownloadError(
            f"SHA-256 mismatch for {file_path}: "
            f"expected {expected}, observed {observed}."
        )


def download_kaggle_dataset(
    handle: str,
    destination: str | Path,
    *,
    dataset_file: str | None = None,
    force: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Download a Kaggle dataset or one dataset file.

    This is the low-level Kaggle operation. Notebook code should normally call
    :func:`acquire_kaggle_dataset`, which also validates and inventories the
    resulting files.
    """
    normalized_handle = handle.strip()

    if not normalized_handle or "/" not in normalized_handle:
        raise ValueError("Kaggle handle must use the 'owner/dataset' format.")

    output_dir = resolve_project_path(destination, project_root=project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError as exc:
        raise DatasetDownloadError(
            "kagglehub is not installed in the active Python environment. "
            "Install it with: python -m pip install kagglehub"
        ) from exc

    try:
        resolved_path = kagglehub.dataset_download(
            normalized_handle,
            path=dataset_file,
            output_dir=str(output_dir),
            force_download=force,
        )
    except TypeError as exc:
        raise DatasetDownloadError(
            "The installed kagglehub version does not support output_dir. "
            "Upgrade it with: python -m pip install --upgrade kagglehub"
        ) from exc
    except Exception as exc:
        raise DatasetDownloadError(
            f"Kaggle download failed for '{normalized_handle}': {exc}"
        ) from exc

    result = Path(resolved_path).expanduser().resolve()

    if not result.exists():
        raise DatasetDownloadError(
            f"kagglehub returned a path that does not exist: {result}"
        )

    return result


def acquire_kaggle_dataset(
    handle: str,
    destination: str | Path,
    *,
    dataset_file: str | None = None,
    force: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> DatasetAcquisition:
    """Acquire a Kaggle dataset and return a notebook-friendly result."""
    root = Path(project_root).expanduser().resolve()
    output_dir = resolve_project_path(destination, project_root=root)

    resolved_path = download_kaggle_dataset(
        handle=handle,
        destination=output_dir,
        dataset_file=dataset_file,
        force=force,
        project_root=root,
    )

    files = discover_dataset_files(output_dir)

    if not files:
        raise DatasetDownloadError(
            f"The Kaggle acquisition completed but no visible files were found "
            f"in {output_dir}."
        )

    return DatasetAcquisition(
        source_kind="kaggle",
        source_reference=handle.strip(),
        destination=output_dir,
        resolved_path=resolved_path,
        files=files,
        project_root=root,
    )


def download_url_file(
    url: str,
    destination: str | Path,
    *,
    filename: str | None = None,
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sha256: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Download one file from an HTTP, HTTPS, or FTP URL atomically.

    Existing files are reused unless ``force=True``. When ``sha256`` is
    supplied, both reused and newly downloaded files are validated.
    """
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme not in SUPPORTED_URL_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_URL_SCHEMES))
        raise ValueError(
            f"Unsupported URL scheme '{scheme}'. Supported schemes: {supported}."
        )

    resolved_filename = filename or unquote(Path(parsed.path).name)
    resolved_filename = _validate_filename(resolved_filename)

    output_dir = resolve_project_path(destination, project_root=project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / resolved_filename

    if output_path.exists() and not force:
        _verify_sha256(output_path, sha256)
        return output_path.resolve()

    temporary_path = output_path.with_name(f".{output_path.name}.part")
    temporary_path.unlink(missing_ok=True)

    request: str | Request

    if scheme in {"http", "https"}:
        request = Request(
            url,
            headers={"User-Agent": "dataset-study-downloader/1.0"},
        )
    else:
        request = url

    try:
        with (
            urlopen(request, timeout=timeout) as response,
            temporary_path.open("wb") as target,
        ):
            shutil.copyfileobj(response, target, length=DEFAULT_CHUNK_SIZE)

        _verify_sha256(temporary_path, sha256)
        os.replace(temporary_path, output_path)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise DatasetDownloadError(f"Download failed for '{url}': {exc}") from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return output_path.resolve()


def acquire_url_file(
    url: str,
    destination: str | Path,
    *,
    filename: str | None = None,
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sha256: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> DatasetAcquisition:
    """Acquire one URL file and return a notebook-friendly result."""
    root = Path(project_root).expanduser().resolve()
    output_dir = resolve_project_path(destination, project_root=root)

    resolved_path = download_url_file(
        url=url,
        destination=output_dir,
        filename=filename,
        force=force,
        timeout=timeout,
        sha256=sha256,
        project_root=root,
    )

    files = discover_dataset_files(output_dir)

    return DatasetAcquisition(
        source_kind="url",
        source_reference=url,
        destination=output_dir,
        resolved_path=resolved_path,
        files=files,
        project_root=root,
    )


def _print_acquisition(acquisition: DatasetAcquisition) -> None:
    """Render a compact deterministic acquisition summary for the CLI."""
    print(f"Source type: {acquisition.source_kind}")
    print(f"Source: {acquisition.source_reference}")
    print(f"Destination: {acquisition.destination}")
    print("Files:")

    for file_path in acquisition.relative_files:
        print(f"- {file_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire source datasets into the study repository.",
    )
    subparsers = parser.add_subparsers(dest="source", required=True)

    kaggle_parser = subparsers.add_parser(
        "kaggle",
        help="Acquire a dataset through kagglehub.",
    )
    kaggle_parser.add_argument(
        "handle",
        nargs="?",
        default=DEFAULT_KAGGLE_HANDLE,
        help=f"Kaggle handle (default: {DEFAULT_KAGGLE_HANDLE}).",
    )
    kaggle_parser.add_argument(
        "--destination",
        default=str(DEFAULT_TELCO_DESTINATION),
        help="Output directory, absolute or relative to the repository root.",
    )
    kaggle_parser.add_argument(
        "--dataset-file",
        help="Optional path to a single file inside the Kaggle dataset.",
    )
    kaggle_parser.add_argument(
        "--force",
        action="store_true",
        help="Download again instead of reusing an existing materialization.",
    )

    url_parser = subparsers.add_parser(
        "url",
        help="Acquire one file from an HTTP, HTTPS, or FTP URL.",
    )
    url_parser.add_argument("url", help="Source HTTP, HTTPS, or FTP URL.")
    url_parser.add_argument(
        "--destination",
        default=str(DEFAULT_RAW_DATA_DIR),
        help="Output directory, absolute or relative to the repository root.",
    )
    url_parser.add_argument(
        "--filename",
        help="Local filename. Defaults to the final URL path component.",
    )
    url_parser.add_argument(
        "--force",
        action="store_true",
        help="Download again and replace an existing local file.",
    )
    url_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Connection/read timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    url_parser.add_argument(
        "--sha256",
        help="Optional expected SHA-256 checksum for integrity validation.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.source == "kaggle":
            acquisition = acquire_kaggle_dataset(
                handle=args.handle,
                destination=args.destination,
                dataset_file=args.dataset_file,
                force=args.force,
            )
        else:
            acquisition = acquire_url_file(
                url=args.url,
                destination=args.destination,
                filename=args.filename,
                force=args.force,
                timeout=args.timeout,
                sha256=args.sha256,
            )
    except (DatasetDownloadError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_acquisition(acquisition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
