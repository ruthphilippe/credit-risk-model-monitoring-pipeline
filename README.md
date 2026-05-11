# Credit Risk Model Monitoring Pipeline

This project is an end-to-end credit risk modeling pipeline built with Python. The goal is to predict whether a loan applicant is likely to become a credit risk/default case, using the German Credit Risk dataset.

The project simulates a real-world banking risk workflow: data loading, data cleaning, data quality checks, feature engineering, model preparation, and future model monitoring.

## Project Objective

The objective of this project is to build a reproducible machine learning pipeline that can support credit risk decision-making.

The project focuses on:

- Loading and cleaning credit risk data
- Creating a binary default/risk target
- Running data quality checks
- Creating credit-risk-related features
- Preparing the dataset for machine learning
- Building a foundation for model evaluation and monitoring

## Dataset

This project uses the German Credit Risk dataset.

The dataset contains information about loan applicants, including:

- Loan amount
- Loan duration
- Credit history
- Savings status
- Employment duration
- Age
- Housing status
- Job type
- Credit risk label

The target variable is converted into a binary column called `default`:

| Value | Meaning |
|---|---|
| 0 | Good credit risk / low default risk |
| 1 | Bad credit risk / higher default risk |

## Project Structure

```text
credit-risk-model-monitoring-pipeline/
│
├── data/
│   ├── raw/
│   │   └── german_credit_raw.csv
│   ├── processed/
│   │   ├── german_credit_clean.csv
│   │   └── german_credit_features.csv
│   └── production_sample/
│
├── notebooks/
│   └── 01_eda_credit_risk.ipynb
│
├── src/
│   ├── load_and_clean_data.py
│   ├── data_quality.py
│   └── feature_engineering.py
│
├── models/
├── reports/
├── dashboard/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
