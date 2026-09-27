"""Canonical study runtime derived from ``.python-version`` and ``pylock.toml``.

The Dataset Study environment contract has exactly two normative sources:

* ``.python-version`` holds the exact CPython version (``X.Y.Z``);
* ``pylock.toml`` (PEP 751, machine-generated from ``pyproject.toml``) pins
  every resolved distribution.

Nothing in this module keeps its own copy of a version. Notebooks call
:func:`require_canonical_runtime` before producing evidence, tests compare
persisted artifact metadata (observed-run evidence) against
:func:`expected_runtime_versions`, and ``python -m scripts.runtime_contract``
verifies an installed environment against the lock.
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Mapping

from packaging.markers import Marker


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PYTHON_VERSION_FILE: Final[str] = ".python-version"
LOCK_FILE: Final[str] = "pylock.toml"

# Runtime components recorded in model artifacts, mapped to their
# distribution names in the lock.
RECORDED_COMPONENTS: Final[Mapping[str, str]] = {
    "pandas": "pandas",
    "scikit_learn": "scikit-learn",
    "joblib": "joblib",
}
# Additional distributions whose version changes numerical results.
NUMERICAL_DISTRIBUTIONS: Final[tuple[str, ...]] = ("numpy", "scipy")


class RuntimeContractError(RuntimeError):
    """Raised when the active interpreter is not the canonical study runtime."""


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load and minimally validate the PEP 751 lock."""
    path = Path(project_root) / LOCK_FILE
    with path.open("rb") as handle:
        lock = tomllib.load(handle)
    if lock.get("lock-version") != "1.0":
        raise RuntimeContractError(f"{LOCK_FILE} must declare lock-version 1.0.")
    if not isinstance(lock.get("packages"), list) or not lock["packages"]:
        raise RuntimeContractError(f"{LOCK_FILE} declares no packages.")
    return lock


def read_lock_pins(
    project_root: str | Path = PROJECT_ROOT,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return ``{normalized name: version}`` for packages that apply here.

    Packages whose PEP 508 ``marker`` excludes the evaluated environment (for
    example Windows-only packages on Linux) are omitted. ``environment``
    overrides marker variables; by default the running interpreter is used.
    """
    pins: dict[str, str] = {}
    for package in read_lock(project_root)["packages"]:
        version = package.get("version")
        if version is None:
            continue
        marker = package.get("marker")
        if marker and not Marker(marker).evaluate(dict(environment or {})):
            continue
        pins[_normalize(package["name"])] = str(version)
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


def _installed_versions() -> dict[str, str]:
    return {
        _normalize(distribution.metadata["Name"]): distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata["Name"]
    }


def observed_runtime_mismatches(
    project_root: str | Path = PROJECT_ROOT,
    *,
    all_locked: bool = False,
) -> list[str]:
    """Compare the active interpreter with the canonical runtime.

    By default only the interpreter and the result-relevant distributions are
    compared; ``all_locked=True`` compares every applicable locked package.
    """
    mismatches: list[str] = []
    expected_python = read_python_version(project_root)
    if platform.python_version() != expected_python:
        mismatches.append(
            f"python: expected {expected_python}, observed {platform.python_version()}"
        )
    pins = read_lock_pins(project_root)
    installed = _installed_versions()
    if all_locked:
        names = sorted(pins)
    else:
        names = [
            _normalize(name)
            for name in (*RECORDED_COMPONENTS.values(), *NUMERICAL_DISTRIBUTIONS)
        ]
    for name in names:
        expected = pins.get(name)
        observed = installed.get(name, "not installed")
        if expected != observed:
            mismatches.append(f"{name}: expected {expected}, observed {observed}")
    return mismatches


def require_canonical_runtime(project_root: str | Path = PROJECT_ROOT) -> dict[str, str]:
    """Fail unless the active interpreter matches ``.python-version`` and the lock."""
    mismatches = observed_runtime_mismatches(project_root)
    if mismatches:
        raise RuntimeContractError(
            "The active environment is not the canonical study runtime: "
            + "; ".join(mismatches)
            + ". Install it with: python -m pip install -r pylock.toml"
        )
    return expected_runtime_versions(project_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the active environment against .python-version and every "
            "applicable package pinned in pylock.toml."
        ),
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    mismatches = observed_runtime_mismatches(args.project_root, all_locked=True)
    applicable = len(read_lock_pins(args.project_root))
    if mismatches:
        print("Environment does NOT match the canonical runtime:", file=sys.stderr)
        for line in mismatches:
            print(f"- {line}", file=sys.stderr)
        return 1
    print(
        f"Environment matches .python-version ({read_python_version(args.project_root)}) "
        f"and all {applicable} applicable packages in {LOCK_FILE}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
