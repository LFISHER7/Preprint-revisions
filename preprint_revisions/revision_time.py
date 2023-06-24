import argparse
import pandas as pd
import matplotlib.pyplot as plt
from util.config import DATA_DIR, RESULTS_DIR
from util.json_loader import load_json, save_to_json
import re

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
    args = parse_args()

    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir()

    data = load_json(f"{DATA_DIR}/extracted_{args.server}.json")

    texts = load_json(f"{DATA_DIR}/revision_dict_{args.server}.json")

    revision_text_dfs = []
    for url, text in texts.items():
       
        # re to get doi (10.1101/2021.04.19.21255414) from url that looks like this - 0.1101/2021.04.19.21255414v2
        doi = re.search(r"(?<=content\/)(.*)(?=v)", url).group(0)
        
        # get version number from url
        version = re.search(r"(?<=v)(.*)", url.split("/")[-1]).group(0)
        version = int(version)
        row_dict = {
            "doi": doi,
            "version": version,
            "text": text
        }
        row = pd.DataFrame(row_dict, index=[0])
        revision_text_dfs.append(row)
    
    revision_text = pd.concat(revision_text_dfs, axis=0)
        

    dfs = [pd.DataFrame(d, index=[0]) for d in data]
    df = pd.concat(dfs)
    
 
    df["version"] = df["version"].astype(int)
    # convert date columns to datetime
    df["date"] = pd.to_datetime(df["date"])
    df["date_month"] = df["date"].dt.to_period("M")

    
    # plot the proportion of preprints with first version each month that go on to have a revision
    has_revision = df.groupby("doi")[["version"]].count()
    df["has_revision"] = df["doi"].isin(has_revision[has_revision["version"] > 1].index)

    # plot monthly proportion of preprints that have a revision
    df.groupby("date_month")["has_revision"].mean().plot()
    plt.title("Proportion of preprints that have a revision")
    # save
    print(f"{RESULTS_DIR}/proportion_revisions_{args.server}.png")
    plt.savefig(f"{RESULTS_DIR}/proportion_revisions_{args.server}.png")



    count_over_time = df.groupby("date")[["version"]].count()
    count_over_time = count_over_time.resample("W").sum()


    #plot count over time
    plt.plot(count_over_time)
    plt.xlabel("Date")
    plt.ylabel("Number of revisions")
    # rotate x labels
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/revisions_over_time_{args.server}.png")
    plt.clf()


    df = df.groupby("doi").filter(lambda x: (x["version"].max() > 1))


    # remove any dois for which we don't have full revision history (ie previous versions outside study period)
    # groupby doi, find max version, filter out any dois with max version not equal to number in groupby
    df = df.groupby("doi").filter(lambda x: (x["version"].max() == len(x)))

    

    revision_traces = {}

    

    updated_df = []

    # group df by doi, order rows by version and then calculate time between versions
    for doi, group in df.groupby("doi"):
        group = group.sort_values(by="version")

     
        

        group["revision_time"] = group["date"].diff().dt.days.astype(int, errors="ignore")
        
        # drop the first row, as we don't have a previous version to compare it to
        revision_times = group["revision_time"].tolist()
        
        if len(revision_times) > 1:
            revision_times = revision_times[1:]

        revision_traces[doi] = revision_times

        # see if the title or abstract changed
        group["title_changed"] = group["title"].ne(group["title"].shift())

        group["abstract_changed"] = group["abstract"].ne(group["abstract"].shift())

        # see if the author list changed
        group["authors_changed"] = group["authors"].ne(group["authors"].shift())

        # for the first row, set all of these to NA
        group.loc[group["version"] == 1, ["title_changed", "abstract_changed", "authors_changed"]] = None

        group = group.merge(revision_text[["doi", "version", "text"]], on=["doi", "version"], how="left")
        group.rename(columns={"text": "revision_text"}, inplace=True)

        updated_df.append(group)
    
    updated_df = pd.concat(updated_df)

    # print the proportion of revisions that change the title, abstract or authors (dont include first version)
    print(f"Proportion of revisions that change title: {updated_df[updated_df['version'] > 1]['title_changed'].mean()}")
    print(f"Proportion of revisions that change abstract: {updated_df[updated_df['version'] > 1]['abstract_changed'].mean()}")
    print(f"Proportion of revisions that change authors: {updated_df[updated_df['version'] > 1]['authors_changed'].mean()}")

    print(f"Number of revisions: {updated_df.shape[0]}")
    print(f"Number of revisions with text: {updated_df['revision_text'].notnull().sum()}")
    updated_df.to_csv(f"{RESULTS_DIR}/revision_stats_{args.server}.csv", index=False)
    times_combined = [time for doi, times in revision_traces.items() for time in times]

    # summary stats for times_combined
    print(f"Mean time between revisions: {sum(times_combined) / len(times_combined)}")
    print(f"Median time between revisions: {sorted(times_combined)[len(times_combined) // 2]}")
    print(f"Max time between revisions: {max(times_combined)}")
    print(f"Min time between revisions: {min(times_combined)}")
    

    # plot distribution of revision times
    plt.hist(times_combined, bins=100)
    plt.title(f"Revision times for {args.server} (n={len(times_combined)})")
    plt.xlabel("Days between revisions")
    plt.ylabel("Number of revisions")
    plt.savefig(f"{RESULTS_DIR}/revision_times_{args.server}.png")
    plt.clf()

    # plot distribution of number of revisions
    plt.hist([len(times) for doi, times in revision_traces.items()], bins=100, density=False, width=0.8)
    plt.title(f"Number of revisions for {args.server} (n={len(revision_traces)})")
    plt.xlabel("Number of revisions")
    plt.ylabel("Number of preprints")
    plt.yticks([])
    plt.savefig(f"{RESULTS_DIR}/number_of_revisions_{args.server}.png")
    plt.clf()

    # find the longest 100 lists in revision_traces
    most_revised = sorted(revision_traces.items(), key=lambda x: len(x[1]), reverse=True)[:100]
    cumulative_times = []
    # for each of these, plot the cumulative sum of revision times on the x axis and the index of the revision on the y axis
    for i, (doi, times) in enumerate(most_revised):
        cumulative_times.append([sum(times[:j]) for j in range(len(times))])

    # order cumulative times by the last value
    cumulative_times = sorted(cumulative_times, key=lambda x: x[-1])

    plt.figure(figsize=(15,12))
    for i, (times) in enumerate(cumulative_times):

        # plot with different colours between points
        # change marker color to red

        plt.plot(times,[i]* len(times), label=doi, marker="o", markersize=5, markerfacecolor="red", linewidth=3, color="blue")
        plt.title(f"Revision times for {args.server} (n={len(most_revised)})")
        plt.xlabel("Days")
       
    plt.savefig(f"{RESULTS_DIR}/revision_times_{args.server}_most_revised.png")
    plt.clf()


    save_to_json(revision_traces, f"{RESULTS_DIR}/revision_traces_{args.server}.json")


if __name__ == "__main__":
    main()