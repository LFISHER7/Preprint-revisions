import pandas as pd

from preprint_revisions.process_data import generate_preprint_df, generate_revisions_df


def test_generate_revisions_df():
    revision_data = {
        "https://www.medrxiv.org/content/10.1101/2022.12.20.22283713v2": "text 1",
        "https://www.medrxiv.org/content/10.1101/2022.12.20.22283713v3": "text 2",
        "https://www.medrxiv.org/content/10.1101/2022.12.16.22283559v2": "text 3",
    }

    result = generate_revisions_df(revision_data)
    expected_result = pd.DataFrame(
        {
            "doi": [
                "10.1101/2022.12.20.22283713",
                "10.1101/2022.12.20.22283713",
                "10.1101/2022.12.16.22283559",
            ],
            "version": [2, 3, 2],
            "text": ["text 1", "text 2", "text 3"],
        }
    )
    pd.testing.assert_frame_equal(result, expected_result)


def test_generate_preprint_df():
    preprint_data = [
        {"doi": "10.1101/2022.12.20.22283713", "version": "1", "date": "2023-01-01"},
        {"doi": "10.1101/2022.12.20.22283713", "version": "2", "date": "2023-02-02"},
        {"doi": "10.1101/2022.12.16.22283559", "version": "1", "date": "2023-03-03"},
    ]

    result = generate_preprint_df(preprint_data)
    expected_result = pd.DataFrame(
        {
            "doi": [
                "10.1101/2022.12.20.22283713",
                "10.1101/2022.12.20.22283713",
                "10.1101/2022.12.16.22283559",
            ],
            "version": pd.Series([1, 2, 1], dtype="int32"),
            "date": pd.to_datetime(["2023-01-01", "2023-02-02", "2023-03-03"]),
            "date_month": pd.Series(
                ["2023-01", "2023-02", "2023-03"], dtype="period[M]"
            ),
            "max_versions": [2, 2, 1],
            "has_revision": [True, False, False],
        }
    )
    pd.testing.assert_frame_equal(result, expected_result)
