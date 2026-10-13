"""
CodeOrbit Tech Data Science Internship
Project: Student Performance Data Cleaning and Exploratory Data Analysis

Run:
    python main.py

Input:
    sample_student_performance_raw.csv

Outputs:
    sample_student_performance_cleaned.csv
    outputs/study_hours_histogram.png
    outputs/study_hours_vs_score.png
    outputs/average_score_by_group.png
    outputs/summary_statistics.csv
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "sample_student_performance_raw.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, standardize numeric columns, validate ranges, and impute missing values."""
    df = df.drop_duplicates().copy()

    numeric_cols = [
        "Study_Hours",
        "Attendance_Percent",
        "Assignment_Score",
        "Final_Score",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validate values against reasonable ranges.
    df.loc[~df["Study_Hours"].between(0, 24), "Study_Hours"] = float("nan")
    df.loc[~df["Attendance_Percent"].between(0, 100), "Attendance_Percent"] = float("nan")
    df.loc[~df["Assignment_Score"].between(0, 100), "Assignment_Score"] = float("nan")
    df.loc[~df["Final_Score"].between(0, 100), "Final_Score"] = float("nan")

    # Median imputation keeps this small demonstration dataset usable.
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def create_visualizations(df: pd.DataFrame) -> None:
    """Save charts used in the EDA report."""
    plt.figure()
    plt.hist(df["Study_Hours"], bins=6)
    plt.title("Distribution of Daily Study Hours")
    plt.xlabel("Study hours per day")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "study_hours_histogram.png", dpi=160)
    plt.close()

    plt.figure()
    plt.scatter(df["Study_Hours"], df["Final_Score"])
    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study hours per day")
    plt.ylabel("Final score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "study_hours_vs_score.png", dpi=160)
    plt.close()

    plt.figure()
    df.groupby("Group")["Final_Score"].mean().plot(kind="bar")
    plt.title("Average Final Score by Group")
    plt.xlabel("Group")
    plt.ylabel("Average final score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_score_by_group.png", dpi=160)
    plt.close()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE.name}. Keep it in the same folder as main.py."
        )

    raw_df = pd.read_csv(INPUT_FILE)
    print("\n--- RAW DATA PREVIEW ---")
    print(raw_df.head())
    print("\n--- MISSING VALUES BEFORE CLEANING ---")
    print(raw_df.isna().sum())
    print(f"\nRows before cleaning: {len(raw_df)}")

    cleaned_df = clean_data(raw_df)
    print(f"Rows after duplicate removal and cleaning: {len(cleaned_df)}")

    output_csv = BASE_DIR / "sample_student_performance_cleaned.csv"
    cleaned_df.to_csv(output_csv, index=False)

    numeric_cols = [
        "Study_Hours",
        "Attendance_Percent",
        "Assignment_Score",
        "Final_Score",
    ]
    stats = cleaned_df[numeric_cols].describe().T
    stats["median"] = cleaned_df[numeric_cols].median()
    stats.to_csv(OUTPUT_DIR / "summary_statistics.csv")

    print("\n--- SUMMARY STATISTICS ---")
    print(stats.round(2))
    print("\n--- VALUE COUNTS: GROUP ---")
    print(cleaned_df["Group"].value_counts())
    print("\n--- CORRELATION: STUDY HOURS AND FINAL SCORE ---")
    print(round(cleaned_df["Study_Hours"].corr(cleaned_df["Final_Score"]), 3))

    create_visualizations(cleaned_df)
    print(f"\nFinished. Cleaned data and charts are saved in: {BASE_DIR}")


if __name__ == "__main__":
    main()
