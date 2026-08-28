from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


class RateModel:
	def __init__(self, pipeline: Pipeline):
		self.pipeline = pipeline

	def predict(self, features: pd.DataFrame) -> pd.Series:
		predictions = np.expm1(self.pipeline.predict(features))
		return pd.Series(predictions, index=features.index).clip(lower=0.01)


def train_model(features: pd.DataFrame, target: pd.Series) -> RateModel:
	categorical = features.select_dtypes(include=["object", "string"]).columns.tolist()
	numeric = [column for column in features.columns if column not in categorical]
	preprocessor = ColumnTransformer([
		("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
		("numeric", "passthrough", numeric),
	])
	pipeline = Pipeline([
		("preprocessor", preprocessor),
		("regressor", RandomForestRegressor(
			n_estimators=150,
			min_samples_leaf=2,
			max_features=0.8,
			n_jobs=-1,
			random_state=42,
		)),
	])
	pipeline.fit(features, np.log1p(target.clip(lower=0)))
	return RateModel(pipeline)


def predict(model: RateModel, features: pd.DataFrame) -> pd.Series:
	return model.predict(features)


def save_model(model: RateModel, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, path)


def load_model(path: Path) -> RateModel:
	return joblib.load(path)
