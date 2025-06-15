"""
Feature selection module for identifying important features.
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each feature in the DataFrame.
    VIF is a measure of multicollinearity among features.
    """
    df_vif = pd.DataFrame()
    df_vif["Feature"] = df.columns
    df_vif["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    df_vif = df_vif.sort_values(by="VIF", ascending=False)
    return df_vif


def drop_high_vif_features(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Drop features with VIF above a specified threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): VIF threshold for dropping features.

    Returns:
        pd.DataFrame: DataFrame with high VIF features dropped.
    """
    df_new = df.copy(deep=True)

    vif_df = compute_vif(df)

    max_vif = vif_df["VIF"].max()

    while max_vif >= threshold:
        max_vif_feature = vif_df[vif_df["VIF"] == max_vif]["Feature"].values[0]

        print(
            f"Dropping Highest VIF Feature > {threshold}: Feature '{max_vif_feature}' has VIF {max_vif}"
        )
        df_new = df_new.drop(columns=max_vif_feature)
        vif_df = compute_vif(df_new)
        max_vif = vif_df["VIF"].max()

    return df_new


# drop_high_vif_features,
#     drop_low_variance_features,
#     drop_missing_features,
#     drop_rare_labels,
#     select_features_by_correlation,
#     select_features_by_importance,
#     select_features_by_permutation_importance,
#     select_features_by_p_value,
#     select_features_by_variance_threshold,
