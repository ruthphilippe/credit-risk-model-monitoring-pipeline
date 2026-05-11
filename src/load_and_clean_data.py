import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/german_credit_raw.csv")
PROCESSED_DATA_PATH = Path("data/processed/german_credit_clean.csv")

DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"


def load_dataset() -> pd.DataFrame:
    """
    Loads the German Credit dataset from a public CSV URL.
    """

    print("Loading German Credit dataset...")

    df = pd.read_csv(DATA_URL)

    print(f"Dataset loaded successfully. Shape: {df.shape}")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the German Credit dataset.
    The target column is converted into a binary default variable:
    good = 0
    bad = 1
    """

    df = df.copy()

    # Standardize column names
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(".", "_")
    )

    print("Columns found:")
    print(df.columns.tolist())

    # Rename target column if needed
    if "class" in df.columns:
        df = df.rename(columns={"class": "default_status"})
    elif "credit_risk" in df.columns:
        df = df.rename(columns={"credit_risk": "default_status"})
    elif "creditability" in df.columns:
        df = df.rename(columns={"creditability": "default_status"})

    if "default_status" not in df.columns:
        raise ValueError("Could not find target column. Expected 'class', 'credit_risk', or 'creditability'.")

    # Standardize target values
    df["default_status"] = df["default_status"].astype(str).str.lower().str.strip()

    print("Target values found:")
    print(df["default_status"].value_counts())

    # Create binary target
    df["default"] = df["default_status"].map({
        "good": 0,
        "bad": 1,
        "1": 0,
        "2": 1,
        "0": 1
    })

    # Validate target mapping
    if df["default"].isna().sum() > 0:
        raise ValueError("Target column contains values that could not be mapped to 0 and 1.")

    # Drop original target text column
    df = df.drop(columns=["default_status"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Basic validation
    if not set(df["default"].unique()).issubset({0, 1}):
        raise ValueError("Target column should only contain 0 and 1.")

    print("Cleaning completed.")
    print(f"Clean dataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")

    return df


def save_data(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    """
    Saves raw and processed data.
    """

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(RAW_DATA_PATH, index=False)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Raw data saved to: {RAW_DATA_PATH}")
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")


def main():
    raw_df = load_dataset()
    clean_df = clean_dataset(raw_df)
    save_data(raw_df, clean_df)


if __name__ == "__main__":
    main()