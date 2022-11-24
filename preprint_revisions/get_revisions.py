import re
import argparse
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from util.json_loader import load_json, save_to_json
from util.config import DATA_DIR

def get_dois(data):
    """
    Extracts all DOIs from a JSON file.
    """
    dois = []
    for preprint in data:
        dois.append(preprint['preprint_doi'])
    return dois

def get_urls(data):
    """
    Extracts all URLs from a JSON file.
    """
    urls = []
    for key, value in data.items():
        for url in value:

            # regex to catch anything that looks like "http://www.medrxiv.org/content/10.1101/2020.04.09.20059790v4"
            if re.match(r'^https?:\/\/(?:www\.)?(?:medrxiv|biorxiv|arxiv)\.org\/content\/10\.1101\/\d{4}\.\d{2}\.\d{2}\.\d{8}v\d{1,2}$', url):
                urls.append(url)
    return urls

def parse_args():
    """
    This function parses arguments
    """

    parser = argparse.ArgumentParser(description="Get sitemap tags from biorxiv and medrxiv")
    parser.add_argument("--server", type=str, help="preprint server to get sitemap tags for")
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
    header = soup.find('li', class_="fn-group-summary-of-updates")
    
    if header:
        revision_text = header.find('p').text
    else:
        revision_text = None
    time.sleep(1)

    return revision_text

def parse_args():
    """
    This function parses arguments
    """

    parser = argparse.ArgumentParser(description="Get sitemap tags from biorxiv and medrxiv")
    parser.add_argument("--server", type=str, help="preprint server to get sitemap tags for")
    args = parser.parse_args()

    return args

def main():
    """
    For all dois in extracted data, find all relevant urls
    """
    args = parse_args()
    server = args.server
    
    data = load_json(f'{DATA_DIR}/extracted_{server}.json')
    dois = get_dois(data)

    url_data = load_json(f'{DATA_DIR}/{server}_urls.json')
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
        if len(value)>1:    
            for i in value:
                # exclude urls that are the first version of the preprint
                if not i.endswith("v1"):
                    text_dict[i] = get_revision_text(i)
                
    save_to_json(text_dict, f'{DATA_DIR}/revision_dict_{server}.json')


if __name__ == "__main__":
    main()

