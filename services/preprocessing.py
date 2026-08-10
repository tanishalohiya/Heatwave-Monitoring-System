import pandas as pd


def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def clean_dataset(df):
    """
    Clean the dataset by removing duplicate rows
    and filling missing values.
    """
    df = df.drop_duplicates()

    # Fill missing numerical values with column mean
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return df