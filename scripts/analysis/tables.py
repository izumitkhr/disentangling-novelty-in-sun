"""Reading the CSV tables written by the analyses.

pandas turns strings such as "NaN", "NA" or "None" into missing values by
default, which breaks reduced formulas: sodium nitride is "NaN". The tables
are written with ``DataFrame.to_csv``, which writes missing values as empty
fields, so only empty fields are read as missing.
"""

from __future__ import annotations

import pandas as pd


def read_table(path, **kwargs) -> pd.DataFrame:
    """``pd.read_csv`` with only empty fields as missing values."""
    return pd.read_csv(
        path, keep_default_na=False, na_values=[""], low_memory=False, **kwargs
    )
