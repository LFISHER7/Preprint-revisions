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
    12 months of the end of the study period. Also drops any preprints where the type is "WITHDRAWN".
    """

    df = df.copy()

    # remove any preprints where type="WITHDRAWN". As opposed to missing or PUBLISHEDAHEADOFPRINT
    df = df[df["type"] != "WITHDRAWN"]

    df["date"] = pd.to_datetime(df["date"])

    latest_date = df["date"].max()

    # dates in get_perprint_data_medrxiv are inclusive. if max date is 1st of month, remove
    if latest_date.is_month_start:
        df = df[df["date"] != latest_date]

    num_preprints = df["doi"].nunique()

    # Drop DOIs where the max version number is not the same as
    df = df.groupby("doi").filter(lambda x: x["version"].max() == len(x))

    # Drop DOIs where the first version is within 12 months of the end of the study period
    df_dropped_end = df.groupby("doi").filter(
        lambda x: (latest_date - x["date"].min()).days > 365
    )

    num_preprints_post_drop = df_dropped_end["doi"].nunique()
    print(
        f"Dropped {num_preprints - num_preprints_post_drop} preprints with incomplete revision history"
    )
    return df, df_dropped_end


def calculate_time_between_v1_v2(df):
    """
    Calculates the time in days between the first and second version of each preprint.

    Args:
        df (pandas.DataFrame): A pandas DataFrame with at least two columns 'date_month' and 'doi'.
        output_dir (pathlib.Path): The directory to save the figure to.
    """
    # select the first verion of each preprint
    second_versions = df.loc[df["version"] == 2, :]

    first_versions = df.loc[
        (df["doi"].isin(second_versions["doi"])) & (df["version"] == 1), :
    ].reset_index()

    # merge on doi
    first_versions = first_versions.merge(
        second_versions, on="doi", suffixes=("_v1", "_v2")
    )[["doi", "date_v1", "date_v2"]]
    first_versions["time_between_v1_v2"] = (
        first_versions["date_v2"] - first_versions["date_v1"]
    ).dt.days

    return first_versions["time_between_v1_v2"].tolist()


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

    # select the first verion of each preprint
    first_versions = df.loc[df["version"] == 1, :]
    preprints_over_time = first_versions.groupby("date_month")["doi"].nunique()

    plt.plot(preprints_over_time, color="#EF4444")
    plt.title("Monthly number of new preprints")
    plt.xlabel("Date")
    plt.ylabel("n")
    plt.xticks(rotation=90)

    ax = plt.gca()
    ax.set_yticks(np.arange(0, preprints_over_time.max() + 200, 200))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/preprints_over_time.png", transparent=True)
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
    first_versions = df[df["version"] == 1]
    proportion_with_revision = (
        first_versions.groupby("doi")["has_revision"].any().mean() * 100
    )
    proportion_with_revision_over_time = (
        first_versions.groupby(["doi", "date_month"])["has_revision"]
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
    plt.plot(proportion_with_revision_over_time, color="#EF4444")
    plt.title("Proportion of preprints that have a revision")
    plt.ylabel("Proportion (%)")
    plt.xlabel("Date")
    plt.xticks(rotation=90)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(f"{output_dir}/proportion_revisions.png", transparent=True)
    plt.clf()

    return proportion_with_revision


def num_versions_distribution(df, output_dir, nbins=5):
    """
    Plots the distribution of the number of revisions and saves the plot to a PNG file.

    Args:
        revision_traces (dict): A dictionary mapping DOIs to lists of revision times.
        bins (int): The number of bins to use in the histogram.
    Returns:
        None
    """

    num_versions = pd.Series(df.groupby("doi")["version"].max().tolist()).value_counts()

    # aggregate all versions greater than 5
    num_versions.loc[5] = num_versions.loc[5:].sum()
    num_versions = num_versions[:5]

    # set the last x tick label to 5+
    num_versions.index = ["1", "2", "3", "4", "5+"]

    # convert to percentages
    num_versions = num_versions / num_versions.sum() * 100

    plt.figure(figsize=(10, 5))
    sns.barplot(x=num_versions.index, y=num_versions.values, color="#EF4444", width=1)

    # make outline of bars black
    for i, bar in enumerate(plt.gca().patches):
        bar.set_edgecolor("black")
        bar.set_linewidth(1)

    plt.title("Distribution of number of versions")
    plt.xlabel("Number of versions")
    plt.ylabel("Proportion (%)")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/num_versions_distribution.png", transparent=True)
    plt.clf()

    # version not including preprints with only 1 version
    num_versions = num_versions[1:]
    num_versions = num_versions / num_versions.sum() * 100

    plt.figure(figsize=(10, 5))
    sns.barplot(x=num_versions.index, y=num_versions.values, color="#EF4444", width=1)

    for i, bar in enumerate(plt.gca().patches):
        bar.set_edgecolor("black")
        bar.set_linewidth(1)

    plt.title("Distribution of number of versions")
    plt.xlabel("Number of versions")
    plt.ylabel("Proportion (%)")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/num_versions_distribution_no_v1.png", transparent=True)
    plt.clf()


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


def plot_revision_time_distribution(times_between, output_dir):
    """
    Plots the distribution of times between revisions and saves the plot to a PNG file.

    Args:
        times_between (list): A list of times between revisions.
        bins (int): The number of bins to use in the histogram.
    Returns:
        None
    """

    min_time = min(times_between)
    max_time = max(times_between)

    bins = np.arange(min_time, max_time, 50)
    plt.figure(figsize=(10, 5))
    sns.histplot(times_between, bins=bins, color="#EF4444", stat="percent")
    plt.title(f"Revision times (n={len(times_between):,})")
    plt.xlabel("Days between revisions")
    plt.ylabel("%")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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
        group["title_changed"] = (
            group["title"].ne(group["title"].shift()).astype("boolean")
        )
        group["abstract_changed"] = (
            group["abstract"].ne(group["abstract"].shift()).astype("boolean")
        )
        group["authors_changed"] = (
            group["authors"].ne(group["authors"].shift()).astype("boolean")
        )
        group.loc[
            group["version"] == 1,
            ["title_changed", "abstract_changed", "authors_changed"],
        ] = np.nan
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
    plt.plot(proportion_with_text, color="#EF4444")
    plt.title("Proportion of revisions with a revision summary")
    plt.ylabel("Proportion with text (%)")
    plt.xlabel("Date")
    plt.xticks(rotation=90)

    plt.yticks(np.arange(0, 110, 10))
    plt.ylim(0, 100)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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
            markerfacecolor="#EF4444",
            linewidth=3,
            color="#d4d8d4",
        )

    plt.yticks([])
    plt.title(f"Revision history of the {len(most_revised)} most revised preprints")
    plt.xlabel("Days since first version")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
    plt.figure(figsize=(10, 5))
    sns.histplot([len(t) for t in texts], bins=25, color="#EF4444")
    plt.title(f"Revision text length (n={len(texts)})")
    plt.ylabel("n")
    plt.xlabel("Length of revision text (characters)")
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

    plt.style.use("preprint_revisions/util/plots.mplstyle")

    server = "medrxiv" if "medrxiv" in args.data_file.stem else "biorxiv"
    RESULTS_DIR.mkdir(exist_ok=True)

    output_dir = RESULTS_DIR / server
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(DATA_DIR / args.data_file, parse_dates=["date"]).sort_values(
        ["date"]
    )
    print(f"Loaded {len(df)} preprint versions from {args.data_file.stem}")

    num_versions = len(df)
    df, df_dropped = drop_incomplete_revision_history(df)
    num_versions_complete_history = len(df_dropped)
    num_preprints = df["doi"].nunique()

    delta_v1_v2 = np.median(calculate_time_between_v1_v2(df_dropped))

    plot_preprints_over_time(df, output_dir)

    proportion_with_revision = plot_proportion_with_revision(df_dropped, output_dir)

    num_versions_distribution(df_dropped, output_dir)

    save_to_json(
        {
            "num_versions": num_versions,
            "num_versions_complete_history": num_versions_complete_history,
            "num_preprints": num_preprints,
            "proportion_preprint_with_revision": proportion_with_revision,
            "delta_v1_v2": delta_v1_v2,
        },
        f"{output_dir}/num_versions.json",
    )

    df_dropped, revision_traces = calculate_time_between_versions(df_dropped)

    times_between = revision_time_stats(revision_traces, output_dir)
    save_to_json(revision_traces, f"{DATA_DIR}/{server}_revision_traces.json")

    plot_revision_time_distribution(times_between, output_dir)

    df_dropped = find_changes(df_dropped)

    changes_stats(df_dropped, output_dir)

    plot_revisions_with_text_over_time(df_dropped, output_dir)

    plot_most_revised_preprints(revision_traces, output_dir)

    revision_text_stats(df_dropped, output_dir)


if __name__ == "__main__":
    main()
