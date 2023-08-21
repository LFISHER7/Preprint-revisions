import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from util.config import DATA_DIR, RESULTS_DIR
from util.json_loader import save_to_json


def drop_incomplete_revision_history(df):
    """
    Drops DOIs for which we don't have full revision history. This includes, preprints where the
    first version is not within the study period and preprints where the first version is within
    12 months of the end of the study period.
    """
    latest_date = df["date"].max()

    # Drop DOIs where the first version is not within the study period
    df = df.groupby("doi").filter(lambda x: x["version"].max() == len(x))

    # Drop DOIs where the first version is within 12 months of the end of the study period
    df = df.groupby("doi").filter(lambda x: (latest_date - x["date"].min()).days > 365)
    return df


def plot_preprints_over_time(df, output_dir, figsize=(10, 5)):
    """
    Plots the unique DOIs in the DataFrame over time and saves the figure to a PNG file.

    Args:
        df (pandas.DataFrame): A pandas DataFrame with at least two columns 'date_month' and 'doi'.
        figsize (tuple): A tuple indicating the size of the figure. Default is (10, 5).
        output_dir (pathlib.Path): The directory to save the figure to.
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    preprints_over_time = df.groupby("date_month")["doi"].nunique()
    sns.lineplot(data=preprints_over_time)
    plt.xlabel("Date")
    plt.ylabel("Number of preprints")
    plt.xticks(rotation=90)

    plt.savefig(f"{output_dir}/preprints_over_time.png", transparent=True)
    plt.tight_layout()
    plt.clf()


def calculate_proportion_with_revision(df):
    """
    Calculates the proportion of preprints that have a revision.

    Args:
        df (pandas.DataFrame): DataFrame with at least three columns 'doi', 'has_revision', and 'date_month'.
                          'doi' is the identifier of the revision, 'has_revision' is a boolean column indicating
                          if a revision exists, 'date_month' is the date of the revision.
        output_dir (pathlib.Path): The directory to save the figure to.
    Returns:
        proportion_with_revision (float): The mean proportion of preprints that have a revision.
        proportion_with_revision_over_time (pandas.Series): The proportion of preprints that have a revision over time.
    """
    proportion_with_revision = df.groupby("doi")["has_revision"].any().mean() * 100
    proportion_with_revision_over_time = (
        df.groupby(["doi", "date_month"])["has_revision"]
        .any()
        .groupby("date_month")
        .mean()
    ) * 100

    return proportion_with_revision, proportion_with_revision_over_time


def plot_proportion_with_revision(df, output_dir, figsize=(10, 5)):
    """
    Plot the proportion of preprints that have a revision.

    Args:
        df (pandas.DataFrame): DataFrame with at least three columns 'doi', 'has_revision', and 'date_month'.
                          'doi' is the identifier of the revision, 'has_revision' is a boolean column indicating
                          if a revision exists, 'date_month' is the date of the revision.
        output_dir (pathlib.Path): The directory to save the figure to.
    Returns:
        float: The mean proportion of preprints that have a revision.
    """
    (
        proportion_with_revision,
        proportion_with_revision_over_time,
    ) = calculate_proportion_with_revision(df)

    plt.figure(figsize=figsize)
    sns.lineplot(data=proportion_with_revision_over_time)
    plt.title("Proportion of preprints that have a revision")
    plt.ylabel("Proportion (%)")
    plt.xlabel("Date")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/proportion_revisions.png", transparent=True)
    plt.clf()

    return proportion_with_revision


def calculate_time_between_versions(df):
    """
    Calculates the time between different versions for each DOI.

    Args:
        df (pandas.DataFrame): DataFrame with at least three columns 'doi', 'version', and 'date'.
                          'doi' is the identifier of the revision, 'version' is the version number
                          of each revision, 'date' is the date of the revision.
    Returns:
        tuple: A tuple containing the updated DataFrame and a dictionary mapping DOIs to revision times.
    """
    doi_revision_times = {}
    updated_dfs = []

    for doi, group in df.groupby("doi"):
        group = group.sort_values("version")
        group["revision_time"] = (
            group["date"].diff().dt.days.astype(int, errors="ignore")
        )
        revision_times = group["revision_time"].tolist()

        if len(revision_times) > 1:
            revision_times = revision_times[1:]

        doi_revision_times[doi] = revision_times
        updated_dfs.append(group)

    updated_df = pd.concat(updated_dfs)

    return updated_df, doi_revision_times


def plot_num_revisions_distribution(revision_traces, output_dir, bins=100):
    """
    Plots the distribution of the number of revisions and saves the plot to a PNG file.

    Args:
        revision_traces (dict): A dictionary mapping DOIs to lists of revision times.
        bins (int): The number of bins to use in the histogram.
    Returns:
        None
    """
    num_revisions = [len(times) for times in revision_traces.values()]
    sns.histplot(num_revisions, bins=bins, kde=False)
    plt.title(f"Number of revisions (n={len(revision_traces)})")
    plt.xlabel("Number of revisions")
    plt.ylabel("Number of preprints")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/num_revisions_distribution.png", transparent=True)
    plt.clf()


def revision_time_stats(revision_traces, output_dir):
    """
    Computes statistics of revision times and saves the results to a JSON file.

    Args:
        revision_traces (dict): A dictionary mapping DOIs to lists of revision times.
    Returns:
        list: A list of times between revisions, excluding NaNs.
    """
    times_between = [
        time
        for times in revision_traces.values()
        for time in times
        if not np.isnan(time)
    ]
    stats = {
        "mean": np.mean(times_between),
        "median": np.median(sorted(times_between)),
        "max": np.max(times_between),
        "min": np.min(times_between),
    }

    save_to_json(stats, f"{output_dir}/revision_time_stats.json")

    return times_between


def plot_revision_time_distribution(times_between, output_dir, bins=25):
    """
    Plots the distribution of times between revisions and saves the plot to a PNG file.

    Args:
        times_between (list): A list of times between revisions.
        bins (int): The number of bins to use in the histogram.
    Returns:
        None
    """
    sns.histplot(times_between, bins=bins)
    plt.title(f"Revision times (n={len(times_between)})")
    plt.xlabel("Days between revisions")
    plt.ylabel("Number of revisions")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/revision_times.png", transparent=True)
    plt.clf()


def find_changes(df):
    """
    Identifies changes in title, abstract, and authors between different versions of each DOI in a DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with at least five columns: 'doi', 'version', 'title', 'abstract', and 'authors'.
            'doi' is the identifier of the revision, 'version' is the version number of each revision,
            'title', 'abstract', and 'authors' are the corresponding contents of each revision.
    Returns:
        pandas.DataFrame: An updated DataFrame with new boolean columns indicating if the title, abstract, and authors have changed.
    """
    updated_dfs = []

    for _, group in df.groupby("doi"):
        group = group.sort_values("version")
        group["title_changed"] = group["title"].ne(group["title"].shift())
        group["abstract_changed"] = group["abstract"].ne(group["abstract"].shift())
        group["authors_changed"] = group["authors"].ne(group["authors"].shift())
        group.loc[
            group["version"] == 1,
            ["title_changed", "abstract_changed", "authors_changed"],
        ] = None
        updated_dfs.append(group)

    return pd.concat(updated_dfs)


def changes_stats(df, output_dir):
    """
    Computes statistics of changes in title, abstract, and authors and saves the results to a JSON file.

    Args:
        df (pandas.DataFrame): DataFrame returned by the 'find_changes' function.
    Returns:
        None
    """
    stats = {
        "title_changed": float(df[df["version"] > 1]["title_changed"].mean()),
        "abstract_changed": float(df[df["version"] > 1]["abstract_changed"].mean()),
        "authors_changed": float(df[df["version"] > 1]["authors_changed"].mean()),
        "num_revisions": float(df.shape[0]),
        "num_revisions_with_text": float(df["text"].notnull().sum()),
    }

    save_to_json(stats, f"{output_dir}/changes_stats.json")


def plot_revisions_with_text_over_time(df, output_dir):
    """
    Plots the proportion of revisions with text over time and saves the plot.

    Args:
        df (pandas.DataFrame): Data frame containing revision data.
    Returns:
        None
    """
    has_revision = df.loc[(df["version"] > 1), :]

    plt.figure(figsize=(10, 5))
    proportion_with_text = has_revision.groupby("date_month")["text"].apply(
        lambda x: (x.notnull().sum() / len(x)) * 100
    )
    sns.lineplot(data=proportion_with_text)
    plt.ylabel("Proportion of revisions with text")
    plt.xlabel("Date")
    plt.title("Proportion of revisions with text")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/proportion_revisions_with_text.png", transparent=True)
    plt.clf()


def get_most_revised_preprints(revision_traces, n=50):
    """
    Returns the DOIs of the n most revised preprints.

    Args:
        revision_traces (dict): A dictionary mapping DOIs to lists of revision times.
        n (int): The number of most revised preprints to return.
    Returns:
        list: A list of DOIs.
    """
    return sorted(revision_traces.items(), key=lambda x: len(x[1]), reverse=True)[:n]


def plot_most_revised_preprints(revision_traces, output_dir):
    """
    Creates and saves a plot of the revision times for the most revised preprints.

    Args:
        revision_traces (dict): A dictionary mapping DOIs to lists of revision times.
    Returns:
        None
    """
    most_revised = get_most_revised_preprints(revision_traces)
    cumulative_times = []

    for _, times in most_revised:
        cumulative_times.append([sum(times[:j]) for j in range(len(times))])

    cumulative_times = sorted(cumulative_times, key=lambda x: x[-1])

    plt.figure(figsize=(15, 12))
    for i, times in enumerate(cumulative_times):
        plt.plot(
            times,
            i * np.ones(len(times)),
            marker="o",
            markersize=5,
            markerfacecolor="red",
            linewidth=3,
            color="blue",
        )

    plt.yticks([])
    plt.title(f"Revision times (n={len(most_revised)})")
    plt.xlabel("Days since first version", fontsize=20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/revision_times_most_revised.png", transparent=True)
    plt.clf()


def revision_text_stats(df, output_dir):
    """
    Computes statistics about revision text lengths and saves them to a JSON file.

    Args:
        df (pandas.DataFrame): Data frame containing revision data.
    Returns:
        None
    """
    texts = df.loc[df.text.notnull(), "text"]
    mean_text_length = texts.str.len().mean()
    median_text_length = texts.str.len().median()

    save_to_json(
        {"mean": mean_text_length, "median": median_text_length},
        f"{output_dir}/revision_text_stats.json",
    )

    sns.histplot([len(t) for t in texts], bins=50)
    plt.ylabel("Number of revisions with revision text")
    plt.xlabel("Length of revision text")
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/revision_text_length.png", transparent=True)
    plt.clf()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots of revision statistics"
    )
    parser.add_argument(
        "--data-file", type=Path, help="file name of preprint data", required=True
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sns.set_style("darkgrid")
    server = "medrxiv" if "medrxiv" in args.data_file.stem else "biorxiv"
    RESULTS_DIR.mkdir(exist_ok=True)

    output_dir = RESULTS_DIR / server
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(DATA_DIR / args.data_file, parse_dates=["date"])
    num_versions = len(df)
    df = drop_incomplete_revision_history(df)
    num_versions_complete_history = len(df)
    num_preprints = df["doi"].nunique()

    plot_preprints_over_time(df, output_dir)

    proportion_with_revision = plot_proportion_with_revision(df, output_dir)

    save_to_json(
        {
            "num_versions": num_versions,
            "num_versions_complete_history": num_versions_complete_history,
            "num_preprints": num_preprints,
            "proportion_preprint_with_revision": proportion_with_revision,
        },
        f"{output_dir}/num_versions.json",
    )

    df, revision_traces = calculate_time_between_versions(df)

    times_between = revision_time_stats(revision_traces, output_dir)
    save_to_json(revision_traces, f"{output_dir}/revision_traces.json")

    plot_revision_time_distribution(times_between, output_dir)

    df = find_changes(df)

    changes_stats(df, output_dir)

    plot_revisions_with_text_over_time(df, output_dir)

    plot_most_revised_preprints(revision_traces, output_dir)

    revision_text_stats(df, output_dir)


if __name__ == "__main__":
    main()
