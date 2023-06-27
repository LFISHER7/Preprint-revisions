import argparse
import time

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from util.config import DATA_DIR
from util.json_loader import load_json, save_to_json


def get_versions(data_file):
    """
    This function takes a server and returns a dictionary of all preprint dois and their versions
    Args:
        data_file (str): data file to load - contains extracted preprint data
    Returns:
        versions_dict (dict): dictionary of all preprint dois and their versions
    """
    data = load_json(f"{DATA_DIR}/{data_file}")

    # order data based on "date"
    data = sorted(data, key=lambda k: k["date"])

    versions_dict = {}

    for d in data:
        if d["doi"] in versions_dict.keys():
            if "version" in d.keys():
                if int(d["version"]) > versions_dict[d["doi"]]:
                    versions_dict[d["doi"]] = int(d["version"])
        else:
            if "version" in d.keys():
                versions_dict[d["doi"]] = int(d["version"])
            else:
                versions_dict[d["doi"]] = "Missing"

    # drop any key, value pairs from versions_dict that have only 1 version or
    # are missing versions
    versions_dict = {
        k: v for k, v in versions_dict.items() if v != 1 and v != "Missing"
    }

    return versions_dict


def build_urls(version_dict):
    """
    Builds urls for each version of each preprint from the version dictionary
    Args:
        version_dict (dict): dictionary where the key is the doi and the value is the number of versions
    Returns:
        urls (list): list of urls
    """
    urls = []
    for key, value in version_dict.items():
        for version in range(1, value + 1):
            urls.append(f"https://www.medrxiv.org/content/{key}v{version}")
    return urls


def get_revision_text(url):
    """
    You can't get the revision text using the api, so scrape it from the provided url.
    Args:
        url (str): url of the revision
    Returns:
        revision_text (str): the revision text
    """

    page_revision_text = requests.get(url)
    soup_revision_text = BeautifulSoup(page_revision_text.content, "html.parser")
    revision_text_header = soup_revision_text.find(
        "li", class_="fn-group-summary-of-updates"
    )

    if revision_text_header:
        revision_text = revision_text_header.find("p").text
    else:
        revision_text = None

    time.sleep(0.5)

    return revision_text


def parse_args():

    parser = argparse.ArgumentParser(
        description="Get revision text associated with each updated version of a preprint"
    )
    parser.add_argument(
        "--preprint-data-file",
        type=str,
        help="file containing preprint data",
        required=True,
    )
    args = parser.parse_args()

    return args


def main():
    """
    For all dois in extracted data, find all relevant urls
    """
    args = parse_args()
    data_file = args.preprint_data_file

    versions_dict = get_versions(data_file)

    urls = build_urls(versions_dict)
    urls = urls[::-1]

    text_dict = {}
    print("Extracting revision text...")

    for url in tqdm(urls):
        # exclude urls that are the first version of the preprint
        if not url.endswith("v1"):
            text_dict[url] = get_revision_text(url)

    if "medrxiv" in data_file:
        server = "medrxiv"
    else:
        server = "biorxiv"

    save_to_json(text_dict, f"{DATA_DIR}/revision_dict_{server}.json")


if __name__ == "__main__":
    main()
