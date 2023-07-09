import numpy as np
import pandas as pd

from preprint_revisions.generate_revision_statistics import (
    calculate_proportion_with_revision,
    calculate_time_between_versions,
    drop_incomplete_revision_history,
    find_changes,
    get_most_revised_preprints,
)


def test_drop_incomplete_revision_history():
    df = pd.DataFrame(
        {
            "doi": ["doi1", "doi1", "doi1", "doi2", "doi2", "doi3"],
            "version": [1, 2, 3, 2, 3, 1],
        }
    )

    result = drop_incomplete_revision_history(df)
    expected_result = pd.DataFrame(
        {
            "doi": ["doi1", "doi1", "doi1", "doi3"],
            "version": [1, 2, 3, 1],
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected_result.reset_index(drop=True)
    )


def test_calculate_proportion_with_revision():
    df = pd.DataFrame(
        {
            "doi": ["doi1", "doi1", "doi2", "doi3", "doi3", "doi3", "doi4"],
            "has_revision": [True, False, False, True, True, False, False],
            "date_month": pd.Series(
                [
                    "2023-01",
                    "2023-02",
                    "2023-02",
                    "2023-03",
                    "2023-03",
                    "2023-04",
                    "2023-05",
                ],
                dtype="period[M]",
            ),
        }
    )

    prop, prop_over_time = calculate_proportion_with_revision(df)

    assert prop == 50.0

    expected_prop_over_time = pd.Series(
        {
            "2023-01": 100.0,  # no revisions in Jan
            "2023-02": 0.0,  # one out of two DOIs in Feb has a revision, so 50% proportion
            "2023-03": 100.0,  # all DOIs in Mar have a revision, so 100% proportion
            "2023-04": 0.0,  # all DOIs in Apr have a revision, so 100% proportion
            "2023-05": 0.0,  # all DOIs in May have a revision, so 100% proportion
        },
        dtype=np.float64,
        name="has_revision",
    )

    expected_prop_over_time.index = pd.PeriodIndex(
        expected_prop_over_time.index, freq="M", name="date_month"
    )

    pd.testing.assert_series_equal(prop_over_time, expected_prop_over_time)


class UniqueNaN:
    def __eq__(self, other):
        return isinstance(other, self.__class__)


def test_calculate_time_between_versions():
    df = pd.DataFrame(
        {
            "doi": ["doi1", "doi1", "doi1", "doi2", "doi2", "doi3"],
            "version": [1, 2, 3, 1, 2, 1],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-01-01",
                    "2023-02-15",
                    "2023-01-01",
                ]
            ),
        }
    )

    updated_df, revision_times = calculate_time_between_versions(df)

    expected_df = df.copy()
    expected_df["revision_time"] = [np.nan, 31, 28, np.nan, 45, np.nan]

    pd.testing.assert_frame_equal(
        updated_df.reset_index(drop=True), expected_df.reset_index(drop=True)
    )

    expected_revision_times = {
        "doi1": [31.0, 28.0],
        "doi2": [45.0],
        "doi3": [UniqueNaN()],
    }

    # Replace np.nan values with UniqueNaN() objects
    for doi, times in revision_times.items():
        revision_times[doi] = [UniqueNaN() if np.isnan(t) else t for t in times]

    assert revision_times == expected_revision_times


def test_find_changes():
    df = pd.DataFrame(
        {
            "doi": ["doi1", "doi1", "doi2", "doi2"],
            "version": [1, 2, 1, 2],
            "title": ["Title A", "Title B", "Title C", "Title D"],
            "abstract": ["Abstract A", "Abstract A", "Abstract C", "Abstract D"],
            "authors": ["Author A", "Author B", "Author C", "Author C"],
        }
    )

    result_df = find_changes(df)

    expected_df = df.copy()
    expected_df["title_changed"] = [None, True, None, True]
    expected_df["abstract_changed"] = [None, False, None, True]
    expected_df["authors_changed"] = [None, True, None, False]

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_get_most_revised_preprints():
    revision_traces = {
        "doi1": [31, 28, 15, 10],
        "doi2": [45, 20],
        "doi3": [60, 30, 25],
        "doi4": [30, 10, 10, 10, 20],
    }

    # Test with n = 2
    most_revised_preprints = get_most_revised_preprints(revision_traces, 2)
    assert most_revised_preprints == [
        ("doi4", [30, 10, 10, 10, 20]),
        ("doi1", [31, 28, 15, 10]),
    ]

    # Test with n = 1
    most_revised_preprints = get_most_revised_preprints(revision_traces, 1)
    assert most_revised_preprints == [("doi4", [30, 10, 10, 10, 20])]
