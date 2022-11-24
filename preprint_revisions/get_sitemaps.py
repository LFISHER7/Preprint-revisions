from bs4 import BeautifulSoup
import requests
import argparse
from util.json_loader import save_to_json
from util.config import DATA_DIR


def get_sitemap_urls(server):
    """
    This functions gets the sitemap urls from the sitemap for the given server (biorxiv or medrxiv)
    Args:
        server (str): preprint server to get sitemap tags for
    Returns:
        urls (list): list of urls
    """
    urls = []
    sitemap_url = f"https://www.{server}.org/sitemap.xml"
    response = requests.get(sitemap_url)

    soup = BeautifulSoup(response.text, "xml")
    sitemap_tags = soup.find_all("sitemap")

    print(f"The number of sitemaps are: {len(sitemap_tags)}")

    for sitemap in sitemap_tags:
        urls.append(sitemap.findNext("loc").text)

    return urls


def get_urls_from_sitemap(sitemap_tag):
    """
    This function takes a sitemap tag and returns a list of urls
    Args:
        sitemap_tag (str): sitemap tag to get urls from
    Returns:
        urls (list): list of urls
    """

    urls = []
    response = requests.get(sitemap_tag)
    soup = BeautifulSoup(response.text, "xml")
    url_tags = soup.find_all("url")
    for url_tag in url_tags:
        urls.append(url_tag.findNext("loc").text)
    return urls


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
    This function runs the get_sitemaps.py script
    """
    args = parse_args()
    server = args.server
    print(f"Getting sitemaps for {server}...")
    sitemap_urls = get_sitemap_urls(server)
    urls = {}
    for url in sitemap_urls:
        urls[url] = get_urls_from_sitemap(url)

    save_to_json(urls, f"{DATA_DIR}/{server}_urls.json")


if __name__ == "__main__":
    main()
