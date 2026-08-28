from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    shared_columns = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
    result = frame[
        [column for column in shared_columns if column in frame.columns]
    ].copy()
    date = pd.to_datetime(result.pop("date"), errors="coerce").fillna(
        pd.Timestamp("2025-01-01")
    )
    result["year"] = date.dt.year.astype(int)
    result["month"] = date.dt.month.astype(int)
    result["day_of_month"] = date.dt.day.astype(int)
    result["day_of_week"] = date.dt.dayofweek.astype(int)
    result["day_of_year"] = date.dt.dayofyear.astype(int)
    if {"pickup", "delivery"}.issubset(result.columns):
        result["lane"] = result["pickup"] + " -> " + result["delivery"]
    if "distance" in result:
        result["distance_squared"] = result["distance"] ** 2
        result["distance_log"] = np.log1p(result["distance"].clip(lower=0))
    if {"distance", "weight"}.issubset(result.columns):
        result["weight_per_mile"] = result["weight"] / result["distance"].clip(lower=1)
    return result
