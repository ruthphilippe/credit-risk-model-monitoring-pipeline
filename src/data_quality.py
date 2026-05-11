import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/german_credit_clean.csv")

def run_data_quality_checks(df: pd.DataFrame) -> None:
    """
    Runs basic data quality checkes for the credit risk dataset.
    """

    print("Running data quality checks...")

    #Check empty dataset
    assert not df.empty, "Dataset is empty."

    #Check duplicates
    duplicate_count = df.duplicated().sum()
    assert duplicate_count == 0, f"Dataset contains {duplicate_count} duplicate rows."

    #Check target column exists
    assert "default" in df.columns, "Target column 'defaul' is missing."

    #Check target values
    valid_target_values = set(df["default"].unique())
    assert valid_target_values.issubset({0,1}), "Target column must only contain 0 and 1."

    #Check missing target
    assert df["default"].isna().sum() == 0, "Target column contains missing values."

    #Check missing values by column
    missing_values = df.isna().sum()
    columns_with_missing = missing_values[missing_values > 0]

    if not columns_with_missing.empty:
        print("Warning: Missing values found:")
        print(columns_with_missing)
    else:
        print("No missing values found.")
    
    #Check default rate
    default_rate = df["default"].mean()
    assert 0 < default_rate < 1, "Default rate should be between 0 and 1."

    print(f"Default rate: {default_rate: .2%}")
    print("All critical data quality checkes passed.")

def main():
    df = pd.read_csv(DATA_PATH)
    run_data_quality_checks(df)

if __name__ == "__main__":
    main()