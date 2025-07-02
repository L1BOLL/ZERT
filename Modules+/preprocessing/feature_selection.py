# Modules 1.1/preprocessing/feature_selection.py

from pathlib import Path
from typing import Sequence, Union
import pandas as pd
from sklearn.model_selection import train_test_split


def select_top_transcripts(
    df: pd.DataFrame,
    n_transcripts: int = 1000,
    exclude_cols: Union[Sequence[str], None] = None
) -> pd.DataFrame:
    """
    From a DataFrame of counts, pick the n_transcripts columns
    with the fewest zeros (excluding any specified cols).

    :param df: input DataFrame, first col is ID, last is label
    :param n_transcripts: how many transcript columns to keep
    :param exclude_cols: list of column names to skip when counting zeros
    :returns: subset DataFrame with [ID] + top n + [label]
    """
    # figure out which columns to consider
    all_cols = list(df.columns)
    core_cols = all_cols[1:-1]  # skip first and last by default
    if exclude_cols:
        core_cols = [c for c in core_cols if c not in exclude_cols]

    zero_counts = (df[core_cols] == 0).sum(axis=0)
    top = zero_counts.nsmallest(n_transcripts).index.tolist()

    return df[[all_cols[0]] + top + [all_cols[-1]]]


def split_train_test(
    df: pd.DataFrame,
    label_col: str = "type",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split.

    :param df: DataFrame including the label_col
    :param label_col: name of the column to stratify on
    :param test_size: fraction to reserve for test
    :param random_state: for reproducibility
    :returns: (train_df, test_df)
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )
    return train, test
