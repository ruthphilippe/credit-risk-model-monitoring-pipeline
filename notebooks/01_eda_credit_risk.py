import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/german_credit_clean.csv"

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isna().sum())

print("\nTarget Distribution:")
print(df["default"].value_counts())

print("\nDefault rate:")
print(df["default"].mean())

#Target distribution chart
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="default")
plt.title("Default Distribution")
plt.xlabel("Default")
plt.ylabel("Number of Customers")
plt.show()

#Default rate by checking status
if "checking_status" in df.columns:
    default_by_checking = df.groupby("checking_status")["default"].mean().sort_values(ascending=False)

    print("\nDefault rate by checking status:")
    print(default_by_checking)

    plt.figure(figsize=(10, 4))
    default_by_checking.plot(kind="bar")
    plt.title("Default rate by Checking Status")
    plt.xlabel("Checking Status")
    plt.ylabel("Default Rate")
    plt.xticks(rotation=45)
    plt.show()

#Default rate by credit history
if "credit_history" in df.columns:
    default_by_history = df.groupby("credit_history")["default"].mean().sort_values(ascending=False)

    print("\nDefault rate by credit history:")
    print(default_by_history)

    plt.figure(figsize=(10, 4))
    default_by_history.plot(kind="bar")
    plt.title("Default Rate by Credit History")
    plt.xlabel("Credit History")
    plt.ylabel("Default Rate")
    plt.xticks(rotation=45)
    plt.show()


#Numeri variable summary
print("\nNumeric Summary:")
print(df.describe())