"""Cross-platform project context utilities for dataset study notebooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator


PROJECT_ROOT_ENV: Final[str] = "DATASET_STUDY_ROOT"
PROJECT_MARKER: Final[Path] = Path("scripts") / "download_data.py"


class ProjectContextError(RuntimeError):
    """Raised when the dataset study project cannot be located."""


@dataclass(frozen=True)
class ProjectContext:
    """Resolved and validated context for a dataset study project."""

    root: Path

    @property
    def name(self) -> str:
        """Return only the project directory name."""
        return self.root.name

    def path(self, *parts: str | Path) -> Path:
        """Build an absolute path inside the project."""
        return self.root.joinpath(*parts)

    def display(self, path: str | Path | None = None) -> str:
        """Return a safe project-relative path for notebook output.

        Absolute parent directories such as ``/home/user`` or
        ``C:\\Users\\user`` are never displayed.
        """
        if path is None:
            return self.name

        candidate = Path(path).expanduser()

        if not candidate.is_absolute():
            candidate = self.root / candidate

        candidate = candidate.resolve()

        if candidate == self.root:
            return self.name

        try:
            relative_path = candidate.relative_to(self.root)
        except ValueError:
            # Avoid exposing an absolute path outside the project.
            return candidate.name

        # Use forward slashes only for presentation. Filesystem operations
        # continue using pathlib and remain operating-system aware.
        return relative_path.as_posix()

    def require_file(self, *parts: str | Path) -> Path:
        """Return a required project file or raise a clear error."""
        file_path = self.path(*parts)

        if not file_path.is_file():
            relative_path = Path(*parts).as_posix()
            raise FileNotFoundError(
                f"Required project file not found: {relative_path}"
            )

        return file_path

    def require_directory(self, *parts: str | Path) -> Path:
        """Return a required project directory or raise a clear error."""
        directory_path = self.path(*parts)

        if not directory_path.is_dir():
            relative_path = Path(*parts).as_posix()
            raise FileNotFoundError(
                f"Required project directory not found: {relative_path}"
            )

        return directory_path


def _candidate_roots(start: str | Path | None = None) -> Iterator[Path]:
    """Yield possible project roots without repeating candidates."""
    seeds: list[Path] = []

    configured_root = os.getenv(PROJECT_ROOT_ENV)
    if configured_root:
        seeds.append(Path(configured_root).expanduser())

    # This file is expected at:
    # project/scripts/project_context.py
    module_location = Path(__file__).resolve()
    seeds.append(module_location.parents[1])

    start_path = Path(start).expanduser() if start else Path.cwd()
    seeds.append(start_path)

    observed: set[Path] = set()

    for seed in seeds:
        resolved_seed = seed.resolve()

        for candidate in (resolved_seed, *resolved_seed.parents):
            if candidate not in observed:
                observed.add(candidate)
                yield candidate


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the project root using a stable repository marker."""
    for candidate in _candidate_roots(start):
        if (candidate / PROJECT_MARKER).is_file():
            return candidate

    raise ProjectContextError(
        "Dataset study project root not found. "
        f"Expected the project marker '{PROJECT_MARKER.as_posix()}'. "
        f"You may define the {PROJECT_ROOT_ENV} environment variable "
        "with the project root when running the notebook externally."
    )


def get_project_context(
    start: str | Path | None = None,
) -> ProjectContext:
    """Resolve, validate, and return the current project context."""
    root = find_project_root(start)

    context = ProjectContext(root=root)
    context.require_file("scripts", "download_data.py")

    return context
