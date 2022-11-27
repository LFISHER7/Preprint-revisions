import re
import argparse
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from util.json_loader import load_json, save_to_json
from util.config import DATA_DIR

def get_versions(server):
    """
    This function takes a server and returns a dictionary of all preprint dois and their versions
    Args:
        server (str): server to make API call to
    Returns:
        versions_dict (dict): dictionary of all preprint dois and their versions
    """
    data = load_json(f"{DATA_DIR}/extracted_{server}.json")

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

    # drop any key, value pairs from versions_dict that have only 1 version or are missing versions
    versions_dict = {
        k: v for k, v in versions_dict.items() if v != 1 and v != "Missing"
    }

    return versions_dict

def parse_args():
    """
    This function parses arguments
    """

    parser = argparse.ArgumentParser(
        description="Get sitemap tags from biorxiv and medrxiv"
    )
    parser.add_argument(
        "--server", type=str, help="preprint server to get sitemap tags for"
    )
    args = parser.parse_args()

    return args


def get_revision_text(url):
    """
    Gets the revision text from the provided url
    args:
        url (str): url of the revision
    returns:
        revision_text (str): the revision text
    """

    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    header = soup.find("li", class_="fn-group-summary-of-updates")

    if header:
        revision_text = header.find("p").text
    else:
        revision_text = None
    time.sleep(1)

    return revision_text


def parse_args():
    """
    This function parses arguments
    """

    parser = argparse.ArgumentParser(
        description="Get sitemap tags from biorxiv and medrxiv"
    )
    parser.add_argument(
        "--server", type=str, help="preprint server to get sitemap tags for"
    )
    args = parser.parse_args()

    return args


def main():
    """
    For all dois in extracted data, find all relevant urls
    """
    args = parse_args()
    server = args.server

    data = load_json(f"{DATA_DIR}/extracted_{server}.json")
    dois = get_dois(data)

    url_data = load_json(f"{DATA_DIR}/{server}_urls.json")
    urls = get_urls(url_data)

    matches_dict = {}

    for pre_doi in dois:
        for i in urls:
            if re.search(pre_doi, i):

                if pre_doi not in matches_dict.keys():
                    matches_dict[pre_doi] = [i]
                else:
                    matches_dict[pre_doi].append(i)

    text_dict = {}
    print("Extracting revision text...")

    for key, value in tqdm(matches_dict.items()):

        # get revision text for each url with > 1 versions
        if len(value) > 1:
            for i in value:
                # exclude urls that are the first version of the preprint
                if not i.endswith("v1"):
                    text_dict[i] = get_revision_text(i)

    save_to_json(text_dict, f"{DATA_DIR}/revision_dict_{server}.json")


if __name__ == "__main__":
    main()
