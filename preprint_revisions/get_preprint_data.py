import requests
import time
import argparse
from util.json_loader import save_to_json
from util.config import DATA_DIR

# make call to medrxiv API and fetch all preprints


# turn while loop into a function
def get_data(server, start_date, end_date):
    """
    This function takes a url, start date, and end date and returns a list of dictionaries
    Args:
        server (str): server to make API call to
        start_date (str): start date in format YYYY-MM-DD
        end_date (str): end date in format YYYY-MM-DD
    Returns:
        extracted_data (list): list of dictionaries
    """

    preprint_url = f"https://api.biorxiv.org/pubs/{server}/{start_date}/{end_date}"
    extracted_data = []
    page = 1
    item = 0
    while True:
        print(f"Fetching page {page}")
        page += 1
        response = requests.get(f"{preprint_url}/{item}")
        response_json = response.json()
        if not response_json["messages"][0]["status"] == "ok":
            break

        extracted_data.extend(response_json["collection"])
        item += response_json["messages"][0]["count"]
        time.sleep(1)
    return extracted_data


def parse_args():
    """
    This function parses arguments
    """

    parser = argparse.ArgumentParser(
        description="Get preprint data via biorxiv api from biorxiv and medrxiv"
    )
    parser.add_argument(
        "--server", type=str, help="preprint server to get sitemap tags for"
    )
    parser.add_argument(
        "--start_date", type=str, help="start date in format YYYY-MM-DD"
    )
    parser.add_argument("--end_date", type=str, help="end date in format YYYY-MM-DD")
    parser.add_argument(
        "--filename", type=str, help="name of json file to save data to"
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(f"Querying {args.server} API...")
    extracted_data = get_data(args.server, args.start_date, args.end_date)
    save_to_json(extracted_data, DATA_DIR / args.filename)


if __name__ == "__main__":
    main()
