"""Prepare the Saudi-filtered tweets for annotation.

REQUIREMENTS
------------
Python 3 and pandas:

    pip install pandas

INPUT
-----
The input CSV must contain:

``tweet_text``
    Tweet text used for duplicate removal and annotation-text preparation.

``period``
    Study-period value used to print the number of retained rows per period.

All other input columns are retained in the output.

ARGUMENTS
---------
--input PATH
    Path to the Saudi-filtered tweet CSV.

--output PATH
    Path for the annotation-ready CSV.

HOW TO RUN
----------
    python scripts/03_preprocess_tweets.py \
        --input /path/to/saudi_filtered_tweets.csv \
        --output /path/to/annotation_ready_tweets.csv

OUTPUT
------
The output CSV contains the original columns and ``tweet_text_annotation``.
Duplicate rows based on whitespace-normalized tweet text are removed.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_arguments():
    """Define command-line inputs."""
    parser = argparse.ArgumentParser(
        description="Prepare Saudi-filtered tweets for annotation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Saudi-filtered tweet CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV for annotation-ready tweets.",
    )
    return parser.parse_args()


def clean_text_for_model(text):
    """Replace URLs and mentions and normalize spaces."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "URL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    """Remove duplicate tweet text and create annotation text."""
    args = parse_arguments()

    # Read the Saudi-filtered tweets.
    tweets = pd.read_csv(
        args.input,
        lineterminator="\n",
        low_memory=False,
    )

    # Normalize whitespace for duplicate detection.
    tweets["tweet_text_normalized"] = (
        tweets["tweet_text"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    original_unique = tweets["tweet_text"].nunique()
    normalized_unique = tweets["tweet_text_normalized"].nunique()

    # Remove duplicate tweet text and keep the first row.
    annotation_tweets = (
        tweets.drop_duplicates(
            subset=["tweet_text_normalized"],
            keep="first",
        )
        .copy()
        .drop(columns=["tweet_text_normalized"])
    )

    # Create the annotation text column.
    annotation_tweets["tweet_text_annotation"] = annotation_tweets[
        "tweet_text"
    ].apply(clean_text_for_model)

    print(f"Unique original texts: {original_unique:,}")
    print(f"Unique normalized texts: {normalized_unique:,}")
    print(f"Additional duplicates found: {original_unique - normalized_unique:,}")
    print(f"Original rows: {len(tweets):,}")
    print(f"Rows after deduplication: {len(annotation_tweets):,}")
    print(f"Removed: {len(tweets) - len(annotation_tweets):,}")
    print(annotation_tweets["period"].value_counts().sort_index())

    # Save the annotation-ready tweets.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotation_tweets.to_csv(args.output, index=False)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
