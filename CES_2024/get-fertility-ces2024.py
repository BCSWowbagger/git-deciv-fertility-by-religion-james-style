import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ces_fertility_core import build_grouping_options, print_analysis_report, run_analysis


DATASET_KEY = "CES_2024"
WEIGHT_VAR = "commonweight"
CURRENT_YEAR = 2024
MIN_AGE = 44
MAX_AGE = 55

# Choose:
# - "numchildren" for lifetime children ever had
# - "child18" for whether there are children under 18 in the household
ANALYSIS_MODE = "numchildren"

# Set to None for everyone, or one of:
# - "Man"
# - "Woman"
# - "Non-binary"
# - "Other"
GENDER_FILTER = "Woman"
GROUPING_PRESET = "indifferentist_split"
GROUPING_OVERRIDES = {
    "indifferentist_label": "Indifferentist",
    "excluded_religions": set(),
}


def main():
    grouping_options = build_grouping_options(GROUPING_PRESET, GROUPING_OVERRIDES)
    result = run_analysis(
        DATASET_KEY,
        analysis_mode=ANALYSIS_MODE,
        min_age=MIN_AGE,
        max_age=MAX_AGE,
        gender_filter=GENDER_FILTER,
        weight_var=WEIGHT_VAR,
        grouping_options=grouping_options,
    )
    print_analysis_report(result)


if __name__ == "__main__":
    main()
