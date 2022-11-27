import re
import argparse
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from util.json_loader import load_json, save_to_json
from util.config import DATA_DIR
from datetime import datetime


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


def build_urls(version_dict):
    """
    Builds urls from the version dictionary
    args:
        version_dict (dict): dictionary of versions
    returns:
        urls (list): list of urls
    """
    urls = []
    for key, value in version_dict.items():
        for version in range(1, value + 1):
            urls.append(f"https://www.medrxiv.org/content/{key}v{version}")
    return urls


def get_revision_text(url):
    """
    Gets the revision text from the provided url
    args:
        url (str): url of the revision
    returns:
        revision_text (str): the revision text
    """
    
    page = requests.get(url + ".article-info")
    soup = BeautifulSoup(page.content, "html.parser")
    header = soup.find("li", class_="fn-group-summary-of-updates")

    if header:
        revision_text = header.find("p").text
    else:
        revision_text = None
   
    revision_time = soup.find("div", id="highwire-versions-wrapper").find(class_="hw-version-previous-no-link")
   
    # find li element that is parent of revisions times and get "date" attribute
    revision_time = int(revision_time.find_parent("li")["date"])
    
    # convert revision time to datetime object
    revision_time = datetime.fromtimestamp(revision_time).strftime("%Y-%m-%d %H:%M:%S")
    
    time.sleep(1)
    

    return revision_text, revision_time


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

    versions_dict = get_versions(server)

    urls = build_urls(versions_dict)

    text_dict = {}
    print("Extracting revision text...")

    for url in tqdm(urls):
        # exclude urls that are the first version of the preprint
        if not url.endswith("v1"):
            text_dict[url] = get_revision_text(url)

    save_to_json(text_dict, f"{DATA_DIR}/revision_dict_{server}.json")


if __name__ == "__main__":
    main()
