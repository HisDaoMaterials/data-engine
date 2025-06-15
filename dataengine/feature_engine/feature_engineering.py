"""
Feature engineering module for preprocessing and selecting features.
"""

import pandas as pd


def get_numerical_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numerical columns from the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing only numerical columns.
    """
    return df.select_dtypes(include=["number"])


def get_categorical_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract categorical columns from the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame containing only categorical columns.
    """
    return df.select_dtypes(include=["object", "category"])


def get_numerical_features(df: pd.DataFrame) -> list:
    """
    Get a list of numerical feature names from the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        list: List of numerical feature names.
    """
    return get_numerical_dataframe(df).columns.tolist()


def get_categorical_features(df: pd.DataFrame) -> list:
    """
    Get a list of categorical feature names from the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        list: List of categorical feature names.
    """
    return get_categorical_dataframe(df).columns.tolist()
