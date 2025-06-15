"""
Visualization module for exploring and understanding data.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
    method: str = "pearson",
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str = "coolwarm",
    square: bool = True,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """
    Create a correlation heatmap for the DataFrame.
    This function computes the correlation matrix of the DataFrame and visualizes it using a heatmap.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        title (str): Title of the heatmap.
        method (str): Correlation method to use ('pearson', 'kendall', 'spearman').
        annot (bool): Whether to annotate the heatmap with correlation values.
        fmt (str): String format for the annotation text.
        cmap (str): Colormap to use for the heatmap.
        vmin (float): Minimum value for the colormap.
        vmax (float): Maximum value for the colormap.
    """
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df.corr(method=method),
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        square=square,
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(title)
    plt.show()
