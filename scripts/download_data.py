"""Reusable dataset acquisition helpers for dataset-study repositories.

Study-specific choices such as source identifiers and destination paths are
intentionally passed by callers. The module supports Kaggle datasets and
direct HTTP, HTTPS, or FTP file downloads.

A study pins its source through a versioned JSON source contract (see
``contracts/source.json``). :func:`acquire_kaggle_source` downloads the pinned
Kaggle version and refuses to continue unless every expected file matches its
recorded size and SHA-256 and no unexpected file is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import os
import shutil
import sys
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
SUPPORTED_URL_SCHEMES: Final[frozenset[str]] = frozenset(
    {"http", "https", "ftp"}
)
DEFAULT_CHUNK_SIZE: Final[int] = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS: Final[int] = 120
SOURCE_CONTRACT_SCHEMA_VERSION: Final[str] = "source-contract.v1"
_KAGGLE_HANDLE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:/versions/[1-9][0-9]*)?$"
)

SourceKind = Literal["kaggle", "url"]


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded or validated."""


class DatasetIntegrityError(DatasetDownloadError):
    """Raised when acquired files do not match the pinned source contract."""


class SourceContractError(ValueError):
    """Raised when a source contract is malformed."""


@dataclass(frozen=True, slots=True)
class DatasetAcquisition:
    """Describe the materialized result of one dataset acquisition."""

    source_kind: SourceKind
    source_reference: str
    destination: Path
    resolved_path: Path
    files: tuple[Path, ...]
    project_root: Path

    @property
    def display_destination(self) -> str:
        """Return a project-relative destination for safe presentation."""
        try:
            return self.destination.relative_to(self.project_root).as_posix()
        except ValueError:
            return self.destination.name

    @property
    def relative_files(self) -> tuple[str, ...]:
        """Return safe project-relative POSIX file paths."""
        rendered: list[str] = []

        for file_path in self.files:
            try:
                display_path = file_path.relative_to(self.project_root)
            except ValueError:
                rendered.append(file_path.name)
            else:
                rendered.append(display_path.as_posix())

        return tuple(rendered)

    def require_files(self, pattern: str) -> tuple[Path, ...]:
        """Return acquired files matching a filename or glob pattern."""
        normalized_pattern = pattern.strip()
        if not normalized_pattern:
            raise ValueError("File selection pattern cannot be empty.")

        matches = tuple(
            file_path
            for file_path in self.files
            if file_path.match(normalized_pattern)
            or file_path.name == normalized_pattern
        )

        if not matches:
            raise FileNotFoundError(
                f"No acquired file matches '{normalized_pattern}' in "
                f"{self.display_destination}. Available files: "
                f"{list(self.relative_files)}"
            )

        return matches

    def require_one_file(self, pattern: str) -> Path:
        """Return exactly one matching file or raise a clear error."""
        matches = self.require_files(pattern)

        if len(matches) != 1:
            rendered = [path.name for path in matches]
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
    """Resolve a path relative to the supplied project root."""
    root = Path(project_root).expanduser().resolve()
    candidate = Path(path).expanduser()

    if not candidate.is_absolute():
        candidate = root / candidate

    candidate = candidate.resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Dataset destination must remain inside the project root."
        ) from exc

    return candidate


def discover_dataset_files(
    directory: str | Path,
    *,
    include_hidden: bool = False,
) -> tuple[Path, ...]:
    """Return deterministic recursive file paths from a dataset directory."""
    root = Path(directory).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root.name}"
        )

    files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative_parts = path.relative_to(root).parts
        if not include_hidden and any(
            part.startswith(".") for part in relative_parts
        ):
            continue

        files.append(path.resolve())

    return tuple(sorted(files, key=lambda item: item.as_posix()))


@contextmanager
def _suppress_console_output(enabled: bool):
    """Temporarily suppress third-party output, warnings, and logging.

    The implementation is cross-platform because ``os.devnull`` resolves to
    the operating system's null device (for example, ``/dev/null`` on POSIX
    systems and ``NUL`` on Windows).
    """
    if not enabled:
        yield
        return

    previous_logging_disable_level = logging.root.manager.disable

    with (
        open(os.devnull, "w", encoding="utf-8") as output_sink,
        warnings.catch_warnings(),
    ):
        # kagglehub imports tqdm.auto, which can emit this warning when the
        # optional Jupyter widget integration is not installed.
        warnings.filterwarnings(
            "ignore",
            message=r"IProgress not found.*",
        )

        logging.disable(logging.CRITICAL)

        try:
            with (
                redirect_stdout(output_sink),
                redirect_stderr(output_sink),
            ):
                yield
        finally:
            logging.disable(previous_logging_disable_level)


def _validate_filename(filename: str) -> str:
    """Reject empty or path-like filenames for direct downloads."""
    candidate = filename.strip()

    if not candidate or candidate in {".", ".."}:
        raise ValueError("The destination filename cannot be empty.")

    if Path(candidate).name != candidate:
        raise ValueError(
            "The destination filename must not contain directories."
        )

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
        raise ValueError(
            "sha256 must contain exactly 64 hexadecimal characters."
        )

    observed = calculate_sha256(file_path)

    if observed != expected:
        raise DatasetIntegrityError(
            f"SHA-256 mismatch for {file_path.name}: "
            f"expected {expected}, observed {observed}. "
            "The acquired file is not the pinned source snapshot."
        )


def verify_sha256(file_path: str | Path, expected_sha256: str) -> None:
    """Fail with :class:`DatasetIntegrityError` unless the digest matches."""
    _verify_sha256(Path(file_path), expected_sha256)


def download_kaggle_dataset(
    handle: str,
    destination: str | Path,
    *,
    dataset_file: str | None = None,
    force: bool = False,
    show_progress: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Download a Kaggle dataset or one file into a project directory.

    Parameters
    ----------
    show_progress:
        When ``True``, preserve ``kagglehub`` progress and diagnostic output.
        The default suppresses third-party console output so absolute local
        paths are not exposed in notebook results.
    """
    normalized_handle = handle.strip()

    if not _KAGGLE_HANDLE_PATTERN.fullmatch(normalized_handle):
        raise ValueError(
            "Kaggle handle must use the 'owner/dataset' or "
            "'owner/dataset/versions/<n>' format."
        )

    output_dir = resolve_project_path(
        destination,
        project_root=project_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # The import itself is kept inside the suppression context because
        # kagglehub imports tqdm.auto, which may emit notebook widget warnings.
        with _suppress_console_output(enabled=not show_progress):
            import kagglehub

            resolved_path = kagglehub.dataset_download(
                normalized_handle,
                path=dataset_file,
                output_dir=str(output_dir),
                force_download=force,
            )
    except ImportError as exc:
        raise DatasetDownloadError(
            "kagglehub is not installed in the active Python environment. "
            "Install the project dependencies with: "
            "python -m pip install -e ."
        ) from exc
    except TypeError as exc:
        raise DatasetDownloadError(
            "The installed kagglehub version does not support output_dir. "
            "Upgrade the project dependencies."
        ) from exc
    except Exception as exc:
        raise DatasetDownloadError(
            f"Kaggle download failed for '{normalized_handle}': {exc}"
        ) from exc

    result = Path(resolved_path).expanduser().resolve()

    if not result.exists():
        raise DatasetDownloadError(
            "kagglehub returned a path that does not exist."
        )

    return result


def acquire_kaggle_dataset(
    handle: str,
    destination: str | Path,
    *,
    dataset_file: str | None = None,
    force: bool = False,
    show_progress: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> DatasetAcquisition:
    """Acquire a Kaggle dataset and return a notebook-friendly result.

    Third-party progress output is hidden by default to avoid exposing
    absolute paths in notebook output. Set ``show_progress=True`` when
    interactive download diagnostics are needed.
    """
    root = Path(project_root).expanduser().resolve()
    output_dir = resolve_project_path(destination, project_root=root)

    resolved_path = download_kaggle_dataset(
        handle=handle,
        destination=output_dir,
        dataset_file=dataset_file,
        force=force,
        show_progress=show_progress,
        project_root=root,
    )

    files = discover_dataset_files(output_dir)

    if not files:
        raise DatasetDownloadError(
            "The Kaggle acquisition completed but no visible files "
            "were found in the destination directory."
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
    """Download one HTTP, HTTPS, or FTP file atomically."""
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme not in SUPPORTED_URL_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_URL_SCHEMES))
        raise ValueError(
            f"Unsupported URL scheme '{scheme}'. "
            f"Supported schemes: {supported}."
        )

    resolved_filename = filename or unquote(Path(parsed.path).name)
    resolved_filename = _validate_filename(resolved_filename)

    output_dir = resolve_project_path(
        destination,
        project_root=project_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / resolved_filename

    if output_path.exists() and not force:
        _verify_sha256(output_path, sha256)
        return output_path.resolve()

    temporary_path = output_path.with_name(
        f".{output_path.name}.part"
    )
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
            shutil.copyfileobj(
                response,
                target,
                length=DEFAULT_CHUNK_SIZE,
            )

        _verify_sha256(temporary_path, sha256)
        os.replace(temporary_path, output_path)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"Download failed for '{url}': {exc}"
        ) from exc
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

    return DatasetAcquisition(
        source_kind="url",
        source_reference=url,
        destination=output_dir,
        resolved_path=resolved_path,
        files=discover_dataset_files(output_dir),
        project_root=root,
    )


@dataclass(frozen=True, slots=True)
class SourceFileContract:
    """Expected identity of one acquired source file."""

    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceContract:
    """Immutable identity of a pinned Kaggle dataset version."""

    dataset_slug: str
    handle: str
    version: int
    versioned_handle: str
    destination: str
    files: tuple[SourceFileContract, ...]

    def file(self, filename: str) -> SourceFileContract:
        """Return the contract for one expected filename."""
        for item in self.files:
            if item.filename == filename:
                return item
        raise KeyError(f"File is not declared by the source contract: {filename}")


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SourceContractError(
            f"{field} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def parse_source_contract(payload: Mapping[str, Any]) -> SourceContract:
    """Validate a decoded source contract and return its typed form."""
    if payload.get("schema_version") != SOURCE_CONTRACT_SCHEMA_VERSION:
        raise SourceContractError(
            f"Unsupported source contract schema: {payload.get('schema_version')!r}."
        )
    if payload.get("provider") != "kaggle":
        raise SourceContractError("Only the 'kaggle' provider is supported.")

    handle = payload.get("handle")
    version = payload.get("version")
    versioned_handle = payload.get("versioned_handle")
    if not isinstance(handle, str) or not _KAGGLE_HANDLE_PATTERN.fullmatch(handle):
        raise SourceContractError("handle must use the 'owner/dataset' format.")
    if "/versions/" in handle:
        raise SourceContractError("handle must not embed a version.")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SourceContractError("version must be a positive integer.")
    if versioned_handle != f"{handle}/versions/{version}":
        raise SourceContractError(
            "versioned_handle must equal '<handle>/versions/<version>'."
        )

    destination = payload.get("destination")
    if not isinstance(destination, str) or not destination.strip():
        raise SourceContractError("destination must be a project-relative path.")
    if Path(destination).is_absolute():
        raise SourceContractError("destination must be project-relative.")

    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise SourceContractError("files must be a non-empty list.")
    files: list[SourceFileContract] = []
    for index, item in enumerate(raw_files):
        if not isinstance(item, Mapping):
            raise SourceContractError(f"files[{index}] must be an object.")
        filename = _validate_filename(str(item.get("filename", "")))
        size_bytes = item.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise SourceContractError(f"files[{index}].size_bytes must be positive.")
        files.append(
            SourceFileContract(
                filename=filename,
                sha256=_require_sha256(item.get("sha256"), field=f"files[{index}].sha256"),
                size_bytes=size_bytes,
            )
        )
    if len({item.filename for item in files}) != len(files):
        raise SourceContractError("files must not repeat a filename.")

    return SourceContract(
        dataset_slug=str(payload.get("dataset_slug", "")),
        handle=handle,
        version=version,
        versioned_handle=versioned_handle,
        destination=destination,
        files=tuple(files),
    )


def load_source_contract(path: str | Path) -> SourceContract:
    """Load and validate a JSON source contract."""
    with Path(path).open("r", encoding="utf-8") as source:
        return parse_source_contract(json.load(source))


def verify_acquired_files(
    directory: str | Path,
    contract: SourceContract,
    *,
    allow_unexpected_files: bool = False,
) -> tuple[Path, ...]:
    """Verify that a directory holds exactly the pinned source files.

    Every declared file must exist once, with the declared byte size and
    SHA-256. Visible files that the contract does not declare are rejected
    unless ``allow_unexpected_files`` is true. Hidden acquisition markers
    (such as kagglehub's ``.complete`` directory) are ignored.
    """
    root = Path(directory).expanduser().resolve()
    files = discover_dataset_files(root)
    by_name: dict[str, list[Path]] = {}
    for path in files:
        by_name.setdefault(path.name, []).append(path)

    declared = {item.filename for item in contract.files}
    unexpected = sorted(
        path.relative_to(root).as_posix()
        for path in files
        if path.name not in declared
    )
    if unexpected and not allow_unexpected_files:
        raise DatasetIntegrityError(
            f"Unexpected files in {root.name}: {unexpected}. The acquisition "
            "does not match the pinned source contract."
        )

    verified: list[Path] = []
    for item in contract.files:
        matches = by_name.get(item.filename, [])
        if not matches:
            raise DatasetIntegrityError(
                f"Expected source file is missing: {item.filename}."
            )
        if len(matches) != 1:
            raise DatasetIntegrityError(
                f"Expected exactly one {item.filename}, found {len(matches)}."
            )
        path = matches[0]
        observed_size = path.stat().st_size
        if observed_size != item.size_bytes:
            raise DatasetIntegrityError(
                f"Size mismatch for {item.filename}: expected "
                f"{item.size_bytes} bytes, observed {observed_size}."
            )
        _verify_sha256(path, item.sha256)
        verified.append(path)
    return tuple(verified)


def acquire_kaggle_source(
    contract: SourceContract,
    *,
    force: bool = False,
    show_progress: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> DatasetAcquisition:
    """Acquire the pinned Kaggle version and verify it before returning.

    A local copy that already satisfies the contract is reused without network
    access. A local copy that exists but does not match is never silently
    replaced; the caller must pass ``force=True`` to download again.
    """
    root = Path(project_root).expanduser().resolve()
    output_dir = resolve_project_path(contract.destination, project_root=root)

    has_local_files = output_dir.is_dir() and bool(discover_dataset_files(output_dir))
    if has_local_files and not force:
        verify_acquired_files(output_dir, contract)
        resolved_path = output_dir
    else:
        resolved_path = download_kaggle_dataset(
            handle=contract.versioned_handle,
            destination=output_dir,
            force=force,
            show_progress=show_progress,
            project_root=root,
        )
        verify_acquired_files(output_dir, contract)

    return DatasetAcquisition(
        source_kind="kaggle",
        source_reference=contract.versioned_handle,
        destination=output_dir,
        resolved_path=resolved_path,
        files=discover_dataset_files(output_dir),
        project_root=root,
    )


def _print_acquisition(acquisition: DatasetAcquisition) -> None:
    """Render a compact deterministic acquisition summary."""
    print(f"Source type: {acquisition.source_kind}")
    print(f"Source: {acquisition.source_reference}")
    print(f"Destination: {acquisition.display_destination}")
    print("Files:")

    for file_path in acquisition.relative_files:
        print(f"- {file_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire source datasets into a study repository.",
    )
    subparsers = parser.add_subparsers(dest="source", required=True)

    source_parser = subparsers.add_parser(
        "source",
        help=(
            "Acquire the pinned Kaggle version declared by a source contract "
            "and verify every file's size and SHA-256."
        ),
    )
    source_parser.add_argument(
        "contract",
        nargs="?",
        default="contracts/source.json",
        help="Project-relative source contract (default: contracts/source.json).",
    )
    source_parser.add_argument(
        "--force",
        action="store_true",
        help="Download again instead of verifying the existing local copy.",
    )
    source_parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root directory (default: repository root).",
    )
    source_parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show kagglehub progress and diagnostic output.",
    )

    kaggle_parser = subparsers.add_parser(
        "kaggle",
        help="Acquire a dataset through kagglehub.",
    )
    kaggle_parser.add_argument(
        "handle",
        help="Kaggle dataset handle in owner/dataset form.",
    )
    kaggle_parser.add_argument(
        "--destination",
        required=True,
        help="Project-relative output directory.",
    )
    kaggle_parser.add_argument(
        "--dataset-file",
        help="Optional path to one file inside the Kaggle dataset.",
    )
    kaggle_parser.add_argument(
        "--force",
        action="store_true",
        help="Download again instead of reusing an existing result.",
    )
    kaggle_parser.add_argument(
        "--show-progress",
        action="store_true",
        help=(
            "Show kagglehub progress and diagnostic output. "
            "Disabled by default to avoid exposing absolute local paths."
        ),
    )

    url_parser = subparsers.add_parser(
        "url",
        help="Acquire one file from an HTTP, HTTPS, or FTP URL.",
    )
    url_parser.add_argument(
        "url",
        help="Source HTTP, HTTPS, or FTP URL.",
    )
    url_parser.add_argument(
        "--destination",
        required=True,
        help="Project-relative output directory.",
    )
    url_parser.add_argument(
        "--filename",
        help="Local filename; defaults to the URL path component.",
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
        help=(
            "Connection/read timeout in seconds "
            f"(default: {DEFAULT_TIMEOUT_SECONDS})."
        ),
    )
    url_parser.add_argument(
        "--sha256",
        help="Optional expected SHA-256 checksum.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.source == "source":
            contract = load_source_contract(
                resolve_project_path(args.contract, project_root=args.project_root)
            )
            acquisition = acquire_kaggle_source(
                contract,
                force=args.force,
                show_progress=args.show_progress,
                project_root=args.project_root,
            )
        elif args.source == "kaggle":
            acquisition = acquire_kaggle_dataset(
                handle=args.handle,
                destination=args.destination,
                dataset_file=args.dataset_file,
                force=args.force,
                show_progress=args.show_progress,
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
    except (DatasetDownloadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_acquisition(acquisition)
    if args.source == "source":
        print("Integrity: verified against the source contract (size and SHA-256).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
