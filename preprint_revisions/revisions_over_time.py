import argparse
from util import json_loader
from util.config import DATA_DIR

def get_revisions_times(revision_dict):
    """
    This function returns the times for each revision
    """
    


def parse_args():
    """
    This function parses arguments
    """
    parser = argparse.ArgumentParser(
        description="This script clusters the revisions of a preprint",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input file containing the revisions",
        required=True,
    )
    args = parser.parse_args()
    return args

def main():
    revision_dict = json_loader.load_json(f"{DATA_DIR}/revision_dict_{server}.json")

if __name__ == "__main__":
    main()