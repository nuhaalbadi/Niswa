"""Sample Saudi-filtered tweets for annotation.

REQUIREMENTS
------------
Python 3 and pandas:

    pip install pandas

INPUT
-----
The input CSV must contain:

``period``
    Study-period value used for stratified sampling.

``tweet_id``
    Tweet identifier retained in the annotation file.

``tweet_text_annotation``
    Prepared tweet text used for annotation.

ARGUMENTS
---------
--input PATH
    Path to the preprocessed Saudi-filtered tweet CSV.

--output PATH
    Path for the annotation sample CSV.

HOW TO RUN
----------
    python scripts/04_sample_annotation.py \
        --input /path/to/preprocessed_tweets.csv \
        --output /path/to/labelstudio_sample_5000_tweets.csv

OUTPUT
------
The output CSV contains ``tweet_id`` and ``text``. It contains 1,000 randomly
sampled tweets from each study period before empty-text removal.
"""

import argparse
from pathlib import Path

import pandas as pd


SAMPLE_SIZE_PER_PERIOD = 1000
RANDOM_STATE = 42


def parse_arguments():
    """Define command-line inputs."""
    parser = argparse.ArgumentParser(
        description="Sample Saudi-filtered tweets for annotation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Preprocessed Saudi-filtered tweet CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV for the annotation sample.",
    )
    return parser.parse_args()


def main():
    """Sample tweets and save the annotation file."""
    args = parse_arguments()
    tweets = pd.read_csv(args.input)

    # Sample tweets from each period.
    annotation_tweets = (
        tweets.groupby("period", group_keys=False)
        .sample(n=SAMPLE_SIZE_PER_PERIOD, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    # Keep the Label Studio columns.
    annotation_tweets = annotation_tweets[
        ["tweet_id", "tweet_text_annotation"]
    ].copy()
    annotation_tweets = annotation_tweets.rename(
        columns={"tweet_text_annotation": "text"}
    )

    # Remove empty text.
    annotation_tweets["text"] = annotation_tweets["text"].astype(str).str.strip()
    annotation_tweets = annotation_tweets[annotation_tweets["text"] != ""]

    print(annotation_tweets.shape)
    print(
        tweets.groupby("period", group_keys=False)
        .sample(n=SAMPLE_SIZE_PER_PERIOD, random_state=RANDOM_STATE)["period"]
        .value_counts()
        .sort_index()
    )

    # Save the annotation sample.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotation_tweets.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
