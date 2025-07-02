# Modules 1.1/io/readers.py

from pathlib import Path
from typing import Union
import pandas as pd


def read_dataframe(
    path: Union[str, Path],
    **read_csv_kwargs
) -> pd.DataFrame:
    """
    Load a CSV into a pandas DataFrame.

    :param path: filesystem path or URI to CSV
    :param read_csv_kwargs: passed straight into pandas.read_csv
    :returns: DataFrame
    :raises FileNotFoundError: if file doesn’t exist
    :raises pandas.errors.ParserError: if CSV parsing fails
    """
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"no such file: {fp}")
    return pd.read_csv(fp, **read_csv_kwargs)
