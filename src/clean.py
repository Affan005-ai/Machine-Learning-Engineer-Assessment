from __future__ import annotations

import numpy as np
import pandas as pd


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ["distance", "weight"]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").abs()
    if "date" in result:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].replace([np.inf, -np.inf], np.nan)
        result[column] = result[column].fillna(result[column].median())
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].fillna("Unknown").astype(str).str.strip()
    if "date" in result:
        result["date"] = result["date"].fillna(pd.Timestamp("2025-01-01"))
    return result
