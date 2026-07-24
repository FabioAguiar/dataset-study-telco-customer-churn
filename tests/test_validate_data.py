import pandas as pd
import pytest

from scripts.validate_data import (
    DataValidationError,
    analyze_observation_unit,
)


def test_observation_report_for_complete_unique_identifier() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C"],
            "value": [1, 2, 3],
        }
    )

    report = analyze_observation_unit(dataframe, "customer_id")

    assert report.row_count == 3
    assert report.non_null_identifier_count == 3
    assert report.unique_identifier_count == 3
    assert report.missing_identifier_count == 0
    assert report.duplicate_identifier_count == 0
    assert report.duplicated_row_count == 0
    assert report.is_complete
    assert report.is_unique
    report.raise_if_invalid()


def test_observation_report_separates_missing_and_duplicate_counts() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", None],
            "value": [1, 2, 3, 4],
        }
    )

    report = analyze_observation_unit(dataframe, "customer_id")

    assert report.missing_identifier_count == 1
    assert report.duplicate_identifier_count == 1
    assert report.duplicated_row_count == 2
    assert list(report.duplicated_rows["customer_id"]) == ["A", "A"]

    with pytest.raises(DataValidationError, match="missing value"):
        report.raise_if_invalid()


def test_observation_report_does_not_modify_source_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["B", "A", "A"],
            "value": [1, 2, 3],
        }
    )
    original = dataframe.copy(deep=True)

    analyze_observation_unit(dataframe, "customer_id")

    pd.testing.assert_frame_equal(dataframe, original)
