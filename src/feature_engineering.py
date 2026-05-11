import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/german_credit_clean.csv")
OUTPUT_PATH = Path("data/processed/german_credit_features.csv")

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates additional credit-risk features.
    """

    df = df.copy()

    #Loan amount per month
    df["amount_per_month"] = df["amount"] / df["duration"]

    #Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 100],
        labels=["under_25", "25-35", "36-50", "over_50"]
    )

    #Loan duration groups
    df["duration_group"] = pd.cut(
        df["duration"],
        bins=[0, 12, 24, 36, 100],
        labels=["short_term", "medium_term", "long_term", "very_long_term"]
    )

    #Loan amount groups
    df["amount_group"] = pd.cut(
        df["amount"],
        bins=[0, 2000, 5000, 10000, 100000],
        labels=["small" "medium", "large", "very_large", "too_much"]
    )

    print("Feature engineering completed.")
    print(f"New shape: {df.shape}")
    print("New columns added:")
    print(["amount_per_month", "age_group", "duration_group", "amount_group"])

    return df

def main():
    df = pd.read_csv(INPUT_PATH)
    df_features = create_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(OUTPUT_PATH, index=False)

    print(f"Feature dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()