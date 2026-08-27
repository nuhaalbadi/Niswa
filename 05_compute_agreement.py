"""Compute agreement for stance and topic annotations.

REQUIREMENTS
------------
Python 3, pandas, and statsmodels:

    pip install pandas statsmodels

INPUT
-----
The input CSV must contain:

``tweet_id``
    Tweet identifier used to group annotations.

``annotator``
    Annotator identifier used to count annotators.

``stance``
    Stance label used for exact agreement and Fleiss' kappa.

``topic``
    Topic annotation stored as plain text or as JSON with a ``choices`` list.

ARGUMENTS
---------
--input PATH
    Path to the CSV containing three annotations per tweet.

HOW TO RUN
----------
    python scripts/05_compute_agreement.py \
        --input /path/to/annotations.csv

OUTPUT
------
The script prints annotation counts, stance agreement, Fleiss' kappa, exact
topic-set agreement, and pairwise Jaccard similarity statistics.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa


def parse_arguments():
    """Define command-line inputs."""
    parser = argparse.ArgumentParser(
        description="Compute agreement for stance and topic annotations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV containing three annotations per tweet.",
    )
    return parser.parse_args()


def agreement_level(labels):
    """Classify agreement among three labels."""
    unique = len(set(labels))

    if unique == 1:
        return "Full agreement (3/3)"
    if unique == 2:
        return "Majority agreement (2/3)"
    return "No agreement (1/3)"


def interpret_kappa(kappa):
    """Return the interpretation for a kappa value."""
    if kappa < 0:
        return "Poor agreement"
    if kappa < 0.20:
        return "Slight agreement"
    if kappa < 0.40:
        return "Fair agreement"
    if kappa < 0.60:
        return "Moderate agreement"
    if kappa < 0.80:
        return "Substantial agreement"
    return "Almost perfect agreement"


def parse_topic(topic_value):
    """Convert a topic annotation to a sorted tuple."""
    if pd.isna(topic_value):
        return tuple()

    topic_value = str(topic_value).strip()
    if topic_value == "":
        return tuple()

    if topic_value.startswith("{"):
        try:
            parsed_json = json.loads(topic_value)
            topics = parsed_json.get("choices", [])
            return tuple(
                sorted(
                    {
                        topic.strip()
                        for topic in topics
                        if topic.strip()
                    }
                )
            )
        except json.JSONDecodeError:
            print(f"Could not parse JSON value: {topic_value}")
            return tuple()

    return (topic_value,)


def classify_topic_set_agreement(topic_sets):
    """Classify agreement among three topic sets."""
    number_of_unique_sets = len(set(topic_sets))

    if number_of_unique_sets == 1:
        return "Full agreement (3/3)"
    if number_of_unique_sets == 2:
        return "Majority agreement (2/3)"
    return "No agreement (1/3)"


def jaccard_similarity(topic_set_1, topic_set_2):
    """Calculate Jaccard similarity between two topic sets."""
    set_1 = set(topic_set_1)
    set_2 = set(topic_set_2)
    union = set_1 | set_2

    if len(union) == 0:
        return 1.0

    intersection = set_1 & set_2
    return len(intersection) / len(union)


def calculate_tweet_jaccard(topic_sets):
    """Calculate mean pairwise Jaccard similarity for one tweet."""
    annotator_pairs = list(combinations(topic_sets, 2))
    pairwise_scores = [
        jaccard_similarity(first_set, second_set)
        for first_set, second_set in annotator_pairs
    ]
    return sum(pairwise_scores) / len(pairwise_scores)


def compute_stance_agreement(annotations):
    """Compute stance agreement statistics."""
    stance_per_tweet = (
        annotations.groupby("tweet_id")["stance"]
        .apply(list)
        .reset_index()
    )
    stance_per_tweet["agreement"] = stance_per_tweet["stance"].apply(
        agreement_level
    )

    agreement_counts = stance_per_tweet["agreement"].value_counts()
    agreement_summary = (
        stance_per_tweet["agreement"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(agreement_counts)
    print("Agreement percentages:")
    print(agreement_summary)

    rating_matrix = (
        annotations.groupby(["tweet_id", "stance"])
        .size()
        .unstack(fill_value=0)
    )
    print("Rating matrix shape:", rating_matrix.shape)

    row_sums = rating_matrix.sum(axis=1)
    print("Unique row sums:", row_sums.unique())
    assert (row_sums == 3).all(), "Some tweets do not have exactly three annotations."
    print("✓ Every tweet has exactly three ratings.")

    kappa = fleiss_kappa(rating_matrix.values)
    print(f"Fleiss' Kappa: {kappa:.4f}")
    print(f"Interpretation: {interpret_kappa(kappa)}")


def compute_topic_agreement(annotations):
    """Compute topic agreement statistics."""
    annotations["topic_set"] = annotations["topic"].apply(parse_topic)

    topic_per_tweet = (
        annotations.groupby("tweet_id")["topic_set"]
        .apply(list)
        .reset_index(name="topic_sets")
    )
    topic_per_tweet["number_of_annotations"] = topic_per_tweet[
        "topic_sets"
    ].apply(len)

    invalid_topic_counts = topic_per_tweet[
        topic_per_tweet["number_of_annotations"] != 3
    ]
    print(
        "Tweets without exactly three topic annotations:",
        len(invalid_topic_counts),
    )
    if not invalid_topic_counts.empty:
        print(invalid_topic_counts)

    topic_per_tweet["set_agreement"] = topic_per_tweet["topic_sets"].apply(
        classify_topic_set_agreement
    )
    topic_set_agreement_summary = (
        topic_per_tweet["set_agreement"]
        .value_counts()
        .rename_axis("agreement")
        .reset_index(name="count")
    )
    topic_set_agreement_summary["percentage"] = (
        topic_set_agreement_summary["count"]
        / len(topic_per_tweet)
        * 100
    ).round(2)
    print(topic_set_agreement_summary)

    topic_per_tweet["mean_pairwise_jaccard"] = topic_per_tweet[
        "topic_sets"
    ].apply(calculate_tweet_jaccard)

    overall_mean_jaccard = topic_per_tweet["mean_pairwise_jaccard"].mean()
    overall_median_jaccard = topic_per_tweet["mean_pairwise_jaccard"].median()
    overall_std_jaccard = topic_per_tweet["mean_pairwise_jaccard"].std()

    print(f"Mean pairwise Jaccard similarity   : {overall_mean_jaccard:.4f}")
    print(f"Median pairwise Jaccard similarity : {overall_median_jaccard:.4f}")
    print(f"Standard deviation                 : {overall_std_jaccard:.4f}")
    print(
        topic_per_tweet["mean_pairwise_jaccard"]
        .describe()
        .to_frame(name="value")
    )


def main():
    """Load annotations and compute agreement statistics."""
    args = parse_arguments()
    annotations = pd.read_csv(args.input, encoding="utf-8")

    total_annotations = len(annotations)
    unique_tweets = annotations["tweet_id"].nunique()
    num_annotators = annotations["annotator"].nunique()

    print(f"Total annotation records : {total_annotations}")
    print(f"Unique tweets            : {unique_tweets}")
    print(f"Number of annotators     : {num_annotators}")
    print(f"Expected annotations     : {unique_tweets * 3}")

    compute_stance_agreement(annotations)
    compute_topic_agreement(annotations)


if __name__ == "__main__":
    main()
