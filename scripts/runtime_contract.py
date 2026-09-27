"""Canonical study runtime derived from ``.python-version`` and the lock file.

The exact environment that executes the notebooks is declared once:

* ``.python-version`` holds the CPython version;
* ``requirements/lock.txt`` pins every installed distribution.

Notebooks call :func:`require_canonical_runtime` before producing evidence, and
tests compare persisted artifact metadata against :func:`expected_runtime_versions`,
so the bundle, manifests, and documentation cannot silently drift apart.
"""

from __future__ import annotations

import platform
import re
from importlib import metadata
from pathlib import Path
from typing import Final, Mapping


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PYTHON_VERSION_FILE: Final[str] = ".python-version"
LOCK_FILE: Final[str] = "requirements/lock.txt"

# Runtime components recorded in model artifacts, mapped to their
# distribution names in the lock file.
RECORDED_COMPONENTS: Final[Mapping[str, str]] = {
    "pandas": "pandas",
    "scikit_learn": "scikit-learn",
    "joblib": "joblib",
}
# Additional distributions whose version changes numerical results.
NUMERICAL_DISTRIBUTIONS: Final[tuple[str, ...]] = ("numpy", "scipy")

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)")


class RuntimeContractError(RuntimeError):
    """Raised when the active interpreter is not the canonical study runtime."""


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock_pins(project_root: str | Path = PROJECT_ROOT) -> dict[str, str]:
    """Return ``{normalized distribution name: version}`` from the lock file."""
    pins: dict[str, str] = {}
    lock_path = Path(project_root) / LOCK_FILE
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line.strip())
        if match:
            pins[_normalize(match.group(1))] = match.group(2)
    return pins


def read_python_version(project_root: str | Path = PROJECT_ROOT) -> str:
    """Return the canonical CPython version (``major.minor.patch``)."""
    value = (Path(project_root) / PYTHON_VERSION_FILE).read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise RuntimeContractError(f"{PYTHON_VERSION_FILE} must hold X.Y.Z, got {value!r}.")
    return value


def expected_runtime_versions(project_root: str | Path = PROJECT_ROOT) -> dict[str, str]:
    """Return the runtime mapping that model artifacts must record."""
    pins = read_lock_pins(project_root)
    versions = {"python": read_python_version(project_root)}
    for component, distribution in RECORDED_COMPONENTS.items():
        key = _normalize(distribution)
        if key not in pins:
            raise RuntimeContractError(f"{LOCK_FILE} does not pin {distribution}.")
        versions[component] = pins[key]
    return versions


def observed_runtime_mismatches(project_root: str | Path = PROJECT_ROOT) -> list[str]:
    """Compare the active interpreter with the canonical runtime."""
    mismatches: list[str] = []
    expected_python = read_python_version(project_root)
    if platform.python_version() != expected_python:
        mismatches.append(
            f"python: expected {expected_python}, observed {platform.python_version()}"
        )
    pins = read_lock_pins(project_root)
    distributions = [*RECORDED_COMPONENTS.values(), *NUMERICAL_DISTRIBUTIONS]
    for distribution in distributions:
        expected = pins.get(_normalize(distribution))
        try:
            observed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            observed = "not installed"
        if expected != observed:
            mismatches.append(f"{distribution}: expected {expected}, observed {observed}")
    return mismatches


def require_canonical_runtime(project_root: str | Path = PROJECT_ROOT) -> dict[str, str]:
    """Fail unless the active interpreter matches ``.python-version`` and the lock."""
    mismatches = observed_runtime_mismatches(project_root)
    if mismatches:
        raise RuntimeContractError(
            "The active environment is not the canonical study runtime: "
            + "; ".join(mismatches)
            + ". Install it with: python -m pip install -r requirements/lock.txt"
        )
    return expected_runtime_versions(project_root)
