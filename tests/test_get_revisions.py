import pytest
import requests_mock

from preprint_revisions.get_revisions import get_revision_text, get_versions

# Mock data for testing
mock_data = [
    {"doi": "doi1", "version": "1", "date": "2023-01-01"},
    {"doi": "doi1", "version": "2", "date": "2023-01-02"},
    {"doi": "doi2", "date": "2023-01-01"},
]

# Mock BeautifulSoup content for testing
mock_html_content = """
<li class="fn-group-summary-of-updates">
    <p>Mock revision text</p>
</li>
"""


@pytest.fixture
def mock_load_json():
    def _load_json(path):
        return mock_data

    return _load_json


def test_get_versions(mock_load_json, monkeypatch):
    monkeypatch.setattr("preprint_revisions.get_revisions.load_json", mock_load_json)

    # Run the function
    versions_dict = get_versions("biorxiv")

    # Check that the function returns the correct data
    assert versions_dict == {"doi1": 2}


def test_get_revision_text():
    with requests_mock.Mocker() as m:
        # Mock the requests.get call in the function
        m.get("https://www.medrxiv.org/content/doi1v2", text=mock_html_content)

        # Run the function
        revision_text = get_revision_text("https://www.medrxiv.org/content/doi1v2")

    # Check that the function returns the correct data
    assert revision_text == "Mock revision text"
