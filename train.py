from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.clean import clean_data
from src.features import engineer_features
from src.model import predict, save_model, train_model


def train_linear(features: pd.DataFrame, target: pd.Series) -> Pipeline:
	categorical = features.select_dtypes(include=["object", "string"]).columns.tolist()
	numeric = [column for column in features.columns if column not in categorical]
	pipeline = Pipeline([
		("preprocessor", ColumnTransformer([
			("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
			("numeric", "passthrough", numeric),
		])),
		("regressor", Ridge(alpha=10.0)),
	])
	pipeline.fit(features, np.log1p(target.clip(lower=0)))
	return pipeline


def report_metrics(name: str, actual: pd.Series, predicted: pd.Series) -> None:
	print(f"{name} MAE: ${mean_absolute_error(actual, predicted):,.2f}")
	print(f"{name} RMSE: ${mean_squared_error(actual, predicted) ** 0.5:,.2f}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Train and time-validate the freight rate model.")
	parser.add_argument("--input", type=Path, default=Path("data/train-test.csv"))
	parser.add_argument("--model", type=Path, default=Path("artifacts/rate_model.joblib"))
	args = parser.parse_args()
	raw = pd.read_csv(args.input)
	target = raw.pop("posted_rate").astype(float)
	dates = pd.to_datetime(raw["date"], errors="coerce")
	features = engineer_features(clean_data(raw))
	train_mask = dates < pd.Timestamp("2025-09-01")
	test_mask = ~train_mask
	print(f"Training rows: {train_mask.sum():,}; time holdout rows: {test_mask.sum():,}")
	linear = train_linear(features.loc[train_mask], target.loc[train_mask])
	linear_predictions = np.expm1(linear.predict(features.loc[test_mask]))
	report_metrics("Linear Regression", target.loc[test_mask], pd.Series(linear_predictions, index=target.loc[test_mask].index))
	validation_model = train_model(features.loc[train_mask], target.loc[train_mask])
	report_metrics("Random Forest", target.loc[test_mask], predict(validation_model, features.loc[test_mask]))
	final_model = train_model(features, target)
	save_model(final_model, args.model)
	print(f"Saved final model to {args.model}")


if __name__ == "__main__":
	main()
