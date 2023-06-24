import pytest
import requests_mock

from preprint_revisions.get_preprint_data import get_data


# Define a fixture for the mock data and the URL
@pytest.fixture
def mock_setup():
    # Mock data
    data = {
        "collection": [{"id": 1, "name": "Paper1"}, {"id": 2, "name": "Paper2"}],
        "messages": [{"status": "ok", "count": 2}],
    }

    # Define the server and dates for the test
    server = "biorxiv"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    url = f"https://api.biorxiv.org/details/{server}/{start_date}/{end_date}"

    return data, url


# Use the fixture in the test
def test_get_data(mock_setup):
    mock_data, preprint_url = mock_setup

    with requests_mock.Mocker() as m:
        # Mock the requests.get call in the function
        m.get(f"{preprint_url}/0", json=mock_data)
        m.get(f"{preprint_url}/2", json={"messages": [{"status": "error"}]})

        # Run the function
        data = get_data("biorxiv", "2023-01-01", "2023-12-31")

    # Check that the function returns the correct data
    assert data == mock_data["collection"]
