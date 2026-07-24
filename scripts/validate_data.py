"""Reusable, non-mutating validation helpers for tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class DataValidationError(ValueError):
    """Raised when a declared dataset expectation is not satisfied."""


@dataclass(frozen=True, slots=True)
class ObservationUnitReport:
    """Summarize identifier completeness and uniqueness for a dataset."""

    identifier: str
    row_count: int
    non_null_identifier_count: int
    unique_identifier_count: int
    missing_identifier_count: int
    duplicate_identifier_count: int
    duplicated_row_count: int
    duplicated_rows: pd.DataFrame

    @property
    def has_missing_identifiers(self) -> bool:
        """Return whether at least one identifier is absent."""
        return self.missing_identifier_count > 0

    @property
    def has_duplicates(self) -> bool:
        """Return whether at least one non-null identifier is repeated."""
        return self.duplicate_identifier_count > 0

    @property
    def is_complete(self) -> bool:
        """Return whether every row contains an identifier."""
        return not self.has_missing_identifiers

    @property
    def is_unique(self) -> bool:
        """Return whether every non-null identifier occurs once."""
        return not self.has_duplicates

    def summary_frame(self) -> pd.DataFrame:
        """Return a notebook-friendly metric table."""
        return pd.DataFrame(
            {
                "Metric": [
                    "Rows",
                    "Non-null identifiers",
                    "Unique identifiers",
                    "Missing identifiers",
                    "Duplicate identifiers",
                    "Rows involved in duplication",
                ],
                "Value": [
                    self.row_count,
                    self.non_null_identifier_count,
                    self.unique_identifier_count,
                    self.missing_identifier_count,
                    self.duplicate_identifier_count,
                    self.duplicated_row_count,
                ],
            }
        )

    def raise_if_invalid(
        self,
        *,
        require_complete: bool = True,
        require_unique: bool = True,
    ) -> None:
        """Raise one consolidated error for violated expectations."""
        failures: list[str] = []

        if require_complete and self.has_missing_identifiers:
            failures.append(
                f"'{self.identifier}' contains "
                f"{self.missing_identifier_count} missing value(s)"
            )

        if require_unique and self.has_duplicates:
            failures.append(
                f"'{self.identifier}' contains "
                f"{self.duplicate_identifier_count} duplicated "
                "identifier value(s) across "
                f"{self.duplicated_row_count} row(s)"
            )

        if failures:
            raise DataValidationError("; ".join(failures) + ".")


def analyze_observation_unit(
    dataframe: pd.DataFrame,
    identifier: str,
) -> ObservationUnitReport:
    """Analyze whether one identifier supports a row-level observation unit.

    The input DataFrame is never modified. Missing identifiers and repeated
    non-null identifiers are reported separately.
    """
    normalized_identifier = identifier.strip()

    if not normalized_identifier:
        raise ValueError("Observation identifier cannot be empty.")

    if normalized_identifier not in dataframe.columns:
        raise KeyError(
            f"Observation identifier not found: {normalized_identifier}"
        )

    identifier_series = dataframe[normalized_identifier]
    non_null_identifiers = identifier_series.dropna()
    identifier_counts = non_null_identifiers.value_counts(dropna=True)
    duplicated_identifier_values = identifier_counts[
        identifier_counts > 1
    ].index

    duplicated_mask = (
        identifier_series.notna()
        & identifier_series.duplicated(keep=False)
    )
    duplicated_rows = (
        dataframe.loc[duplicated_mask]
        .sort_values(normalized_identifier, kind="stable")
        .copy()
    )

    return ObservationUnitReport(
        identifier=normalized_identifier,
        row_count=int(len(dataframe)),
        non_null_identifier_count=int(identifier_series.notna().sum()),
        unique_identifier_count=int(
            non_null_identifiers.nunique(dropna=True)
        ),
        missing_identifier_count=int(identifier_series.isna().sum()),
        duplicate_identifier_count=int(
            len(duplicated_identifier_values)
        ),
        duplicated_row_count=int(len(duplicated_rows)),
        duplicated_rows=duplicated_rows,
    )
