# Modules 1.1/analysis/expression_analysis.py

from typing import List, Optional
import pandas as pd


def compute_group_means(
    df: pd.DataFrame,
    group_col: str = 'type'
) -> pd.DataFrame:
    """
    Group samples by `group_col` and compute mean expression per transcript.

    :param df: DataFrame with samples as rows, transcripts as columns, plus group_col
    :param group_col: column name to group by
    :returns: DataFrame indexed by group values, columns are transcripts
    """
    return df.groupby(group_col).mean()


def compute_differential_expression(
    means_df: pd.DataFrame
) -> pd.Series:
    """
    Compute differential expression profile between the last and previous group.

    :param means_df: DataFrame of group means (index: group, columns: transcripts)
    :returns: Series of regulation values per transcript
    """
    # difference between last and prior group
    diff = means_df.diff().iloc[-1]
    return diff


def get_top_markers(
    df: pd.DataFrame,
    diff_series: pd.Series,
    group_col: str = 'type',
    top_n: int = 20
) -> pd.DataFrame:
    """
    Select top transcripts by absolute differential expression.

    :param df: original DataFrame with samples as rows, transcripts as columns plus group_col
    :param diff_series: Series of regulation values per transcript
    :param group_col: name of grouping column to drop
    :param top_n: number of top transcripts to select
    :returns: DataFrame of top transcripts (rows) with regulation values
    """
    # transpose to have transcripts as rows
    df_t = df.drop(columns=[group_col]).transpose()
    # annotate regulation
    df_t['regulation'] = diff_series
    # sort by absolute regulation descending
    df_sorted = df_t.reindex(diff_series.abs().sort_values(ascending=False).index)
    return df_sorted.head(top_n)


def extract_marker_data(
    df: pd.DataFrame,
    markers: List[str]
) -> pd.DataFrame:
    """
    Subset original DataFrame by marker transcripts.

    :param df: original DataFrame with transcripts and group_col
    :param markers: list of transcript names to extract
    :returns: DataFrame of samples x selected markers
    """
    return df[markers]