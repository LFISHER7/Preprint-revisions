# Preprint Revisions Analysis

Preprint servers like medRxiv and bioRxiv support preprint revisions, which can represent changes to the original preprint based on feedback, new data, or further analysis. This project aims to analyse the revision data from these servers and provide insights into how preprints are revised and evolve during the early stages of the publication process.

You can find the results of the analysis in the `results/` directory.

## Setup

To set up the project, you need to have [just](https://github.com/casey/just) installed.

Then, run the setup command:

```
just setup
```

## Run

The project uses DVC (Data Version Control) to manage the data pipeline. To run the entire pipeline, use:

```
dvc repro
```

To run individual sections of the pipeline, use:

```
dvc repro {stage_name}
```

where `{stage_name}` is any stage defined in the `dvc.yaml` file.
