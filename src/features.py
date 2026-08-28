from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
	result = frame.copy()
	date = pd.to_datetime(result.pop("date"), errors="coerce").fillna(pd.Timestamp("2025-01-01"))
	result["year"] = date.dt.year.astype(int)
	result["month"] = date.dt.month.astype(int)
	result["day_of_month"] = date.dt.day.astype(int)
	result["day_of_week"] = date.dt.dayofweek.astype(int)
	result["day_of_year"] = date.dt.dayofyear.astype(int)
	if {"pickup", "delivery"}.issubset(result.columns):
		result["lane"] = result["pickup"] + " -> " + result["delivery"]
	if {"pickup_lat", "delivery_lat"}.issubset(result.columns):
		result["latitude_delta"] = (result["delivery_lat"] - result["pickup_lat"]).abs()
	if {"pickup_lon", "delivery_lon"}.issubset(result.columns):
		result["longitude_delta"] = (result["delivery_lon"] - result["pickup_lon"]).abs()
	if "distance" in result:
		result["distance_squared"] = result["distance"] ** 2
		result["distance_log"] = np.log1p(result["distance"].clip(lower=0))
	if {"distance", "weight"}.issubset(result.columns):
		result["weight_per_mile"] = result["weight"] / result["distance"].clip(lower=1)
	if {"market_index", "quote_signal"}.issubset(result.columns):
		result["market_quote_interaction"] = result["market_index"] * result["quote_signal"]
	return result
