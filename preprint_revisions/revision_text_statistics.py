import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
from util import json_loader
from util.config import DATA_DIR, RESULTS_DIR

def parse_args():
    """
    This function parses arguments
    """
    parser = argparse.ArgumentParser(
        description="This script clusters the revisions of a preprint",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--server", type=str, help="preprint server to get sitemap tags for")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    revision_dict = json_loader.load_json(f"{DATA_DIR}/revision_dict_{args.server}.json")
    
    revision_dfs = []
    for url, text in revision_dict.items():
        
        # regular expression to get doi (10.1101/2021.04.19.21255414) from url that looks like this - https://www.medrxiv.org/content/10.1101/2021.04.19.21255414v2
        doi = re.search(r"(?<=content\/)(.*)(?=v)", url).group(0)
        
        # get version number from url
        version = re.search(r"(?<=v)(.*)", url.split("/")[-1]).group(0)

        # make df row from doi, version, and text
        revision_dfs.append(pd.DataFrame({"doi": doi, "version": version, "text": text}, index=[0]))

        

    revision_df = pd.concat(revision_dfs, axis=0)
   

    preprint_data = json_loader.load_json(f"{DATA_DIR}/extracted_{args.server}.json")
    preprint_df = pd.DataFrame(preprint_data)
    
    # merge preprint_df and revision_df on doi
    merged_df = pd.merge(preprint_df, revision_df, on=["doi", "version"])
    # print(merged_df.index)
    # merged_df["version"] = merged_df["version"].astype(int)
    # merged_df = merged_df.groupby("doi").filter(lambda x: (x["version"].max() > 1))
    # merged_df = merged_df.groupby("doi").filter(lambda x: (x["version"].max() == len(x)))


    merged_df["revision_text_length"] = merged_df["text"].apply(lambda x: len(x.split()) if x is not None else 0)


    with_revisions = merged_df[merged_df["revision_text_length"] > 0]
    
    
  
    print(f"Average number of words per revision: {with_revisions['revision_text_length'].sum()/len(with_revisions['revision_text_length'])}")
    print(f"Median number of words per revision: {with_revisions['revision_text_length'].median()}")
    print(f"Number of versions with revision text: {len(with_revisions['revision_text_length'])}/{len(merged_df)}")

    # plot proportion of revisions with text over time
    merged_df["has_revision_text"] = merged_df["revision_text_length"] > 0
    merged_df["date_month"] = pd.to_datetime(merged_df["date"]).dt.to_period("M")
    merged_df.groupby("date_month")["has_revision_text"].mean().plot()
    merged_df.to_csv(f"{DATA_DIR}/merged_df_{args.server}.csv")
    # first recorded date where there is a revision with text longer than 0
    first_date = merged_df[merged_df["has_revision_text"] == True]["date"].min()
    print(f"First recorded revision date: {first_date}")
    
    

    plt.title("Proportion of revisions with text")
    plt.savefig(f"{RESULTS_DIR}/proportion_revisions_with_text_{args.server}.png")


    # plot distribution of number of words
    with_revisions["revision_text_length"].hist(bins=100)
    plt.xlabel("Number of words")
    plt.ylabel("Number of revisions")
    plt.title(f"Distribution of number of words per revision for {args.server}")
    plt.savefig(f"{RESULTS_DIR}/revision_text_statistics_{args.server}.png")

if __name__ == "__main__":
    main()