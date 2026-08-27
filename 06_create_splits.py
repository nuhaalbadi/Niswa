"""Create stratified train, validation, and test splits.

REQUIREMENTS
------------
Python 3, pandas, and scikit-learn:

    pip install pandas scikit-learn

INPUT
-----
The annotation JSON must contain 5,000 records with:

``tweet_id``
    Unique tweet identifier.

``period``
    Study period used for stratified splitting.

``stance``
    Gold stance label.

``topics``
    Gold topic labels.

The metadata CSV must contain ``tweet_id`` and ``tweet_created_at``.

ARGUMENTS
---------
--input PATH
    Path to the gold-annotation JSON file.

--metadata PATH
    Path to the Saudi-filtered tweet CSV containing creation timestamps.

--output-dir PATH
    Directory where the split files will be created.

HOW TO RUN
----------
    python scripts/06_create_splits.py \
        --input /path/to/gold_annotations.json \
        --metadata /path/to/preprocessed_tweets.csv \
        --output-dir /path/to/splits

OUTPUT
------
The output directory contains ``women_gold_train.json`` with 3,500 records,
``women_gold_validation.json`` with 750 records, and ``women_gold_test.json``
with 750 records.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
REQUIRED_COLUMNS = ["tweet_id", "period", "stance", "topics"]
EXPECTED_SIZES = {"train": 3500, "validation": 750, "test": 750}
EXPECTED_PER_PERIOD = {"train": 700, "validation": 150, "test": 150}
OUTPUT_FILENAMES = {
    "train": "women_gold_train.json",
    "validation": "women_gold_validation.json",
    "test": "women_gold_test.json",
}


def parse_arguments():
    """Define command-line inputs."""
    parser = argparse.ArgumentParser(
        description="Create stratified train, validation, and test splits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Gold-annotation JSON file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Saudi-filtered tweet CSV containing tweet_created_at.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the split JSON files will be created.",
    )
    return parser.parse_args()


def load_annotations(input_path):
    """Load and check the gold annotations."""
    with input_path.open("r", encoding="utf-8") as input_file:
        records = json.load(input_file)

    annotations = pd.DataFrame(records)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in annotations.columns
    ]
    if missing_columns:
        raise ValueError(f"Input file is missing columns: {missing_columns}")
    if len(annotations) != 5000:
        raise ValueError(f"Expected 5,000 records, but found {len(annotations):,}.")
    if annotations[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("The input contains missing required values.")
    if annotations["tweet_id"].duplicated().any():
        raise ValueError("The input contains duplicate tweet_id values.")
    return annotations


def add_creation_timestamps(annotations, metadata_path):
    """Add tweet creation timestamps to the annotation records."""
    tweet_metadata = pd.read_csv(
        metadata_path,
        usecols=["tweet_id", "tweet_created_at"],
        dtype={"tweet_id": "string"},
    )

    annotations["_tweet_id_key"] = annotations["tweet_id"].astype("string")
    tweet_metadata["_tweet_id_key"] = tweet_metadata["tweet_id"].astype("string")

    if tweet_metadata["_tweet_id_key"].duplicated().any():
        raise ValueError("The metadata file contains duplicate tweet_id values.")

    original_record_count = len(annotations)
    annotations = annotations.merge(
        tweet_metadata[["_tweet_id_key", "tweet_created_at"]],
        on="_tweet_id_key",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(annotations) != original_record_count:
        raise ValueError("Merging tweet_created_at changed the record count.")
    if annotations["tweet_created_at"].isna().any():
        missing_count = int(annotations["tweet_created_at"].isna().sum())
        raise ValueError(
            f"Missing tweet_created_at for {missing_count:,} annotated tweets."
        )

    annotations = annotations.drop(columns="_tweet_id_key")
    column_order = list(annotations.columns)
    column_order.remove("tweet_created_at")
    tweet_id_position = column_order.index("tweet_id")
    column_order.insert(tweet_id_position + 1, "tweet_created_at")
    return annotations[column_order]


def create_splits(annotations):
    """Create the train, validation, and test dataframes."""
    train, temporary = train_test_split(
        annotations,
        test_size=0.30,
        stratify=annotations["period"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        stratify=temporary["period"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def check_splits(splits, annotations):
    """Check split sizes, period counts, and tweet IDs."""
    for name, split in splits.items():
        if len(split) != EXPECTED_SIZES[name]:
            raise ValueError(f"Unexpected {name} size: {len(split)}")
        period_counts = split["period"].value_counts()
        if not period_counts.eq(EXPECTED_PER_PERIOD[name]).all():
            raise ValueError(f"The {name} split is not balanced by period.")

    train_ids = set(splits["train"]["tweet_id"])
    validation_ids = set(splits["validation"]["tweet_id"])
    test_ids = set(splits["test"]["tweet_id"])

    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("A tweet_id appears in more than one split.")
    if len(train_ids | validation_ids | test_ids) != len(annotations):
        raise ValueError("The splits do not contain every source tweet_id exactly once.")

    print("Validation passed: all tweet IDs are unique across splits.")


def save_splits(splits, output_files):
    """Save the split files as JSON arrays."""
    for name, split in splits.items():
        with output_files[name].open("w", encoding="utf-8") as output_file:
            json.dump(
                split.to_dict(orient="records"),
                output_file,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved {name}: {output_files[name]} ({len(split):,} records)")


def main():
    """Create, check, and save the data splits."""
    args = parse_arguments()
    annotations = load_annotations(args.input)
    annotations = add_creation_timestamps(annotations, args.metadata)

    print("Total records:", len(annotations))
    print("Period distribution:")
    print(annotations["period"].value_counts().sort_index().rename("records").to_frame())

    splits = create_splits(annotations)
    split_summary = pd.DataFrame(
        {
            name: split["period"].value_counts().sort_index()
            for name, split in splits.items()
        }
    ).T
    split_summary["total"] = split_summary.sum(axis=1)
    print(split_summary)

    check_splits(splits, annotations)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        name: args.output_dir / filename
        for name, filename in OUTPUT_FILENAMES.items()
    }
    save_splits(splits, output_files)


if __name__ == "__main__":
    main()
