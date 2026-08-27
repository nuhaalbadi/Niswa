"""Normalize profile locations and classify them geographically.

This script performs geographic normalization and classification of profile locations:

1. Read the collected tweet files and count unique profile-location values.
2. Normalize profile locations and geographic mapping resources.
3. Apply the Albadi et al. location-to-country mapping by exact match.
4. Classify still-unmapped locations containing exact tokens associated with
   other Arab countries as ``other``.
5. Classify remaining locations using the Saudi geographic location map,
   which combines Saudi city names and manually reviewed expressions.
6. Save a location-frequency table with the final geographic classification.

REQUIREMENTS
------------
Install Python 3 and pandas:

    pip install pandas

Required inputs
---------------
--tweets PATH [PATH ...]
    The collected-tweet CSV files. Each file must contain the column
    ``profile_location``.

--location-maps PATH [PATH ...]
    Two CSV files containing the Albadi et al. location-to-country
    resource. Each file must contain ``location`` and ``country`` columns.
    Values labelled ``Saudi`` are retained; every other mapped country is
    converted to ``other``.

--saudi-location-map PATH
    CSV containing the Saudi geographic location mapping. This file combines
    Arabic and English city names derived from the Manateq resource with 127
    manually identified Saudi-related expressions. It must contain a ``location``
    column unless a different column is specified with
    ``--saudi-location-column``.

--output PATH
    Destination CSV for the unique normalized locations, their tweet
    frequencies, and their final classification: ``Saudi``, ``other``, or
    ``unmapped``.

Optional arguments
------------------
--chunksize INTEGER
    Number of tweet rows read at a time. The default is 250,000, which avoids
    loading all collected tweets into memory simultaneously.

--saudi-location-column TEXT
    Name of the location column in the Saudi geographic mapping CSV. The
    default is ``location``.

Example
-------
Run this command from the repository root, adjusting input paths as needed:

    python scripts/02_geographic_filtering.py \
        --tweets /path/to/tweets_period_1.csv \
                 /path/to/tweets_period_2.csv \
                 /path/to/tweets_period_3.csv \
                 /path/to/tweets_period_4.csv \
                 /path/to/tweets_period_5.csv \
        --location-maps resources/hate_mapped.csv \
                        resources/no_hate_mapped.csv \
        --saudi-location-map /path/to/saudi_geographic_location_map.csv \
        --output /path/to/location_country_mapping.csv

"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# Arabic letter normalization.
ARABIC_LETTER_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
    }
)
ARABIC_DIACRITICS_AND_TATWEEL = re.compile("[ًٌٍَُِّْـ]")
HTTPS_PATTERN = re.compile(r"\bhttps:[^\s]+\b")
HTTP_PATTERN = re.compile(r"\bhttp:[^\s]+\b")
WWW_PATTERN = re.compile(r"\bwww.[^\s]+\b")
REPEATED_CHARACTER_PATTERN = re.compile(r"(\w)\1+")
NON_WORD_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
ARABIC_NUMBER_PATTERN = re.compile(r"[\u0660-\u0669]+")
WESTERN_NUMBER_PATTERN = re.compile(r"[0-9]+")
MULTIPLE_SPACE_PATTERN = re.compile(r"\s+")

# Other-country geographic indicators.
OTHER_COUNTRY_INDICATORS = [
    "kuwait", "الكويت", "kuwaitsomenumber", "egypt", "مصر", "uae",
    "الامارات", "tunisia", "تونس", "emirates", "cairo", "ابوظبي",
    "dubai", "kwt", "turkey", "تركياsudan", "السودان",
    "amanjordan", "عمان", "الاردن", "aman", "oman", "khartoum",
    "الخرطوم", "abudhabi", "benghazi", "بنغازي", "العراق", "iraq",
    "egyptcairo", "كويت", "palestine", "فلطسين", "غزه", "gaza",
    "dohaqatar", "قطر", "qatar", "البحرين", "bahrain", "المنامه",
    "manama", "libya", "ليبا", "طرابلس", "trablus", "algeria",
    "الجزائر", "morocco", "المغرب", "الرباط", "rabat", "مسقط",
    "muscat", "الدوحه", "doha", "yemen", "u a e", "اليمن", "صنعاء",
    "sanaa", "jordan", "لبنان", "سوريا", "بيروت", "دمشق", "القدس",
    "بغداد", "baghdad", "lebanon", "syria", "jerusalem", "beirut",
    "damascus", "israel", "العين", "بوظبي", "jordanaman", "emirate",
    "ismailia", "istanbul", "elmansoura", "mansoura", "hashemite",
    "mansouraegypt", "sharjah", "palestinegaza", "زايد",
    "beirutlebanon", "moroco", "damieta", "alexandriaegypt",
    "egyptalexandria", "benha", "kuw", "جابر", "irbid", "nablus",
    "dxb", "nasr", "benghazilibya", "libyabenghazi", "aswan",
    "casablanca", "omdurman", "الجيزه", "الخليج", "damanhur", "البصره",
    "uaedubai", "kuwaiti", "sura", "kuwaitsura", "uaeshj", "dubaiuae",
    "المحرق", "elmahala", "اسيوط", "tantaegypt", "kuwaitqortuba",
    "الزقازيق", "الاردنيه",
]


def parse_arguments():
    """Define command-line inputs."""
    parser = argparse.ArgumentParser(
        description="Normalize profile locations and map them geographically."
    )
    parser.add_argument(
        "--tweets",
        type=Path,
        nargs="+",
        required=True,
        help="Collected-tweet CSV files containing profile_location.",
    )
    parser.add_argument(
        "--location-maps",
        type=Path,
        nargs="+",
        required=True,
        help="Location-to-country CSV files containing location and country.",
    )
    parser.add_argument(
        "--saudi-location-map",
        type=Path,
        required=True,
        help="Combined Saudi geographic location mapping CSV.",
    )
    parser.add_argument(
        "--saudi-location-column",
        default="location",
        help="Location column in the Saudi mapping CSV (default: location).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV for the classified unique-location mapping.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Tweet rows processed per chunk (default: 250000).",
    )
    return parser.parse_args()


def normalize_location(value):
    """Normalize profile and reference location strings."""
    if pd.isna(value):
        return "unavailable"

    location = str(value).translate(ARABIC_LETTER_MAP)

    location = location.replace("\U0001f1f8", " Saudi_flag ")
    location = location.lower()

    # Replace URLs with a common placeholder.
    location = HTTPS_PATTERN.sub(" link ", location)
    location = HTTP_PATTERN.sub(" link ", location)
    location = WWW_PATTERN.sub(" link ", location)
    location = REPEATED_CHARACTER_PATTERN.sub(r"\1", location)
    location = ARABIC_DIACRITICS_AND_TATWEEL.sub("", location)
    location = location.replace("_", " ")
    location = NON_WORD_PATTERN.sub(" ", location)
    location = ARABIC_NUMBER_PATTERN.sub(" somenumber ", location)
    location = WESTERN_NUMBER_PATTERN.sub(" somenumber ", location)

    return MULTIPLE_SPACE_PATTERN.sub(" ", location).strip()


def count_tweet_locations(tweet_paths, chunksize):
    """Count normalized locations while reading large tweet files in chunks."""
    normalized_counts = Counter()
    raw_nonempty_locations = set()
    total_tweets = 0

    for tweet_path in tweet_paths:
        for chunk in pd.read_csv(
            tweet_path,
            usecols=["profile_location"],
            chunksize=chunksize,
            low_memory=False,
        ):
            total_tweets += len(chunk)

            # Record unique non-empty locations before preprocessing.
            nonempty = chunk["profile_location"].dropna().astype(str)
            raw_nonempty_locations.update(
                value for value in nonempty if value.strip()
            )

            normalized = chunk["profile_location"].map(normalize_location)
            normalized_counts.update(normalized)

    locations = pd.DataFrame(
        normalized_counts.items(), columns=["unique_location", "counts"]
    )
    locations["country"] = "unmapped"
    locations = locations.sort_values("counts", ascending=False).reset_index(drop=True)
    return locations, len(raw_nonempty_locations), total_tweets


def prepare_location_country_map(map_paths):
    """Normalize and combine the Albadi et al. location-to-country files."""
    mappings = []
    for map_path in map_paths:
        mapping = pd.read_csv(map_path, usecols=["location", "country"])
        mapping["unique_location"] = mapping["location"].map(normalize_location)
        mapping["country"] = mapping["country"].where(
            mapping["country"].eq("Saudi"), "other"
        )
        mappings.append(mapping[["unique_location", "country"]])

    # Remove duplicate normalized locations.
    return pd.concat(mappings, ignore_index=True).drop_duplicates(
        subset="unique_location", keep="first"
    )


def prepare_saudi_locations(mapping_path, location_column):
    """Normalize the combined Saudi geographic location mapping."""
    mapping = pd.read_csv(mapping_path, usecols=[location_column])
    return {
        normalize_location(location)
        for location in mapping[location_column]
        if normalize_location(location) != "unavailable"
    }


def prepare_other_indicators():
    """Normalize the embedded list of non-Saudi geographic indicators."""
    return {
        normalize_location(indicator)
        for indicator in OTHER_COUNTRY_INDICATORS
        if normalize_location(indicator) != "unavailable"
    }


def contains_exact_token(location, indicators):
    """Match any whitespace-delimited token exactly."""
    return bool(set(location.split()).intersection(indicators))


def classify_locations(locations, reference_map, other_indicators, saudi_locations):
    """Apply the reference, other-country, and Saudi mappings in sequence."""
    classified = locations.merge(
        reference_map,
        on="unique_location",
        how="left",
        suffixes=("", "_reference"),
    )
    exact_reference_match = classified["country_reference"].notna()
    classified.loc[exact_reference_match, "country"] = classified.loc[
        exact_reference_match, "country_reference"
    ]
    classified = classified.drop(columns="country_reference")

    # Classify other-country indicators before checking Saudi locations.
    unmapped = classified["country"].eq("unmapped")
    other_match = classified.loc[unmapped, "unique_location"].map(
        lambda location: contains_exact_token(location, other_indicators)
    )
    classified.loc[other_match.index[other_match], "country"] = "other"

    unmapped = classified["country"].eq("unmapped")
    saudi_match = classified.loc[unmapped, "unique_location"].map(
        lambda location: (
            location in saudi_locations
            or contains_exact_token(location, saudi_locations)
        )
    )
    classified.loc[saudi_match.index[saudi_match], "country"] = "Saudi"
    stage_counts = {
        "reference_matches": int(exact_reference_match.sum()),
        "unmapped_after_reference": int(
            len(classified)
            - exact_reference_match.sum()
        ),
        "additional_other_matches": int(other_match.sum()),
        "additional_saudi_matches": int(saudi_match.sum()),
        "final_unmapped": int(classified["country"].eq("unmapped").sum()),
    }
    classified = classified.sort_values("counts", ascending=False).reset_index(drop=True)
    return classified, stage_counts


def print_summary(
    classified,
    raw_unique_count,
    total_tweets,
    reference_map,
    saudi_locations,
    stage_counts,
):
    """Print location counts."""
    print(f"Tweets processed: {total_tweets:,}")
    print(f"Unique non-empty locations before preprocessing: {raw_unique_count:,}")
    print(f"Unique normalized locations: {len(classified):,}")
    print(f"Unique normalized reference-map entries: {len(reference_map):,}")
    print(
        "Unique normalized entries in Saudi geographic map: "
        f"{len(saudi_locations):,}"
    )
    print(f"Exact reference-map matches: {stage_counts['reference_matches']:,}")
    print(
        "Unmapped after reference-map matching: "
        f"{stage_counts['unmapped_after_reference']:,}"
    )
    print(
        "Additional other-country matches: "
        f"{stage_counts['additional_other_matches']:,}"
    )
    print(
        "Additional Saudi matches: "
        f"{stage_counts['additional_saudi_matches']:,}"
    )
    print(f"Final unmapped locations: {stage_counts['final_unmapped']:,}")

    for country in ["Saudi", "other", "unmapped"]:
        subset = classified[classified["country"].eq(country)]
        print(
            f"{country}: unique_locations={len(subset):,}, "
            f"tweets={subset['counts'].sum():,}"
        )


def main():
    """Run the complete geographic normalization and classification pipeline."""
    args = parse_arguments()
    locations, raw_unique_count, total_tweets = count_tweet_locations(
        args.tweets, args.chunksize
    )
    reference_map = prepare_location_country_map(args.location_maps)
    other_indicators = prepare_other_indicators()
    saudi_locations = prepare_saudi_locations(
        args.saudi_location_map, args.saudi_location_column
    )
    classified, stage_counts = classify_locations(
        locations, reference_map, other_indicators, saudi_locations
    )

    # Save the final location mapping.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(args.output, index=False)
    print_summary(
        classified,
        raw_unique_count,
        total_tweets,
        reference_map,
        saudi_locations,
        stage_counts,
    )


if __name__ == "__main__":
    main()
