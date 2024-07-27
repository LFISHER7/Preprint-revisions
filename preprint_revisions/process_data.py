import argparse
import re

import pandas as pd
from util.config import DATA_DIR
from util.json_loader import load_json


def generate_revisions_df(revision_data):
    """
    Conerts revision data into a dataframe. Gets version and doi from url.
    Args:
        revision_data (dict): dictionary of revision data
    Returns:
        df (pd.DataFrame): dataframe of revision data
    """
    revision_text_dfs = []

    for url, text in revision_data.items():
        doi = re.search(r"(?<=content\/)(.*)(?=v)", url).group(0)
        version = int(re.search(r"(?<=v)(.*)", url.split("/")[-1]).group(0))
        row_dict = {"doi": doi, "version": version, "text": text}
        row = pd.DataFrame(row_dict, index=[0])
        revision_text_dfs.append(row)

    return pd.concat(revision_text_dfs, axis=0).reset_index(drop=True)


def generate_preprint_df(preprint_data):
    """
    Converts preprint data into a dataframe. Adds columns for max versions and whether a preprint has a revision.
    Args:
        preprint_data (dict): dictionary of preprint data
    Returns:
        df (pd.DataFrame): dataframe of preprint data
    """
    dfs = [pd.DataFrame(d, index=[0]) for d in preprint_data]
    df = pd.concat(dfs)

    df["version"] = df["version"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    df["date_month"] = df["date"].dt.to_period("M")

    num_versions = df.groupby("doi")[["version"]].count()
    df["max_versions"] = df["doi"].map(num_versions["version"].to_dict())

    # remove any doi where the dates of the versions are not in order. This is a problem with the data from the API
    problematic_dois = df.groupby("doi").apply(
        lambda x: (x.sort_values("version")["date"].diff().dt.days < 0).any(),
        include_groups=False,
    )
    problematic_dois = problematic_dois[problematic_dois].index.tolist()

    if len(problematic_dois) > 0:
        print(
            f"Some DOIs have versions where the dates are not in order: {problematic_dois}"
        )

    # Remove problematic DOIs
    df = df[~df["doi"].isin(problematic_dois)]

    # a preprint has a revision if it has more than one version and the version number is not the same as version in num_versions
    df["has_revision"] = (df["max_versions"] > 1) & (
        df["version"] != df["max_versions"]
    )
    return df.reset_index(drop=True)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Joins preprint data with revision text"
    )
    parser.add_argument(
        "--preprint-data-file",
        type=str,
        help="file name of preprint data",
        required=True,
    )
    parser.add_argument(
        "--revision-data-file",
        type=str,
        help="file name of revision data",
        required=True,
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    server = "medrxiv" if "medrxiv" in args.preprint_data_file else "biorxiv"

    data = load_json(f"{DATA_DIR}/{args.preprint_data_file}")
    texts = load_json(f"{DATA_DIR}/{args.revision_data_file}")

    revision_text = generate_revisions_df(texts)

    df = generate_preprint_df(data)

    merged_df = df.merge(revision_text, on=["doi", "version"], how="left")
    merged_df.to_csv(f"{DATA_DIR}/preprint_revision_df_{server}.csv", index=False)


if __name__ == "__main__":
    main()
