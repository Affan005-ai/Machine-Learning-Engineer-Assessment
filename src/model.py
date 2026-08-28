from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


class RateModel:
	def __init__(self, pipeline: Pipeline):
		self.pipeline = pipeline

	def predict(self, features: pd.DataFrame) -> pd.Series:
		predictions = np.expm1(self.pipeline.predict(features))
		return pd.Series(predictions, index=features.index).clip(lower=0.01)


def build_pipeline(features: pd.DataFrame, **regressor_parameters: object) -> Pipeline:
	categorical = features.select_dtypes(include=["object", "string"]).columns.tolist()
	numeric = [column for column in features.columns if column not in categorical]
	preprocessor = ColumnTransformer([
		("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
		("numeric", "passthrough", numeric),
	])
	forest_parameters = {
		"n_estimators": 150,
		"min_samples_leaf": 2,
		"max_features": 0.8,
		"n_jobs": -1,
		"random_state": 42,
	}
	forest_parameters.update(regressor_parameters)
	pipeline = Pipeline([
		("preprocessor", preprocessor),
		("regressor", RandomForestRegressor(**forest_parameters)),
	])
	return pipeline


def train_model(features: pd.DataFrame, target: pd.Series, **regressor_parameters: object) -> RateModel:
	pipeline = build_pipeline(features, **regressor_parameters)
	pipeline.fit(features, np.log1p(target.clip(lower=0)))
	return RateModel(pipeline)


def tune_model(features: pd.DataFrame, target: pd.Series, n_iter: int = 8) -> tuple[RateModel, dict[str, object]]:
	search = RandomizedSearchCV(
		build_pipeline(features),
		param_distributions={
			"regressor__max_depth": [8, 12, 16, 20, None],
			"regressor__min_samples_leaf": [1, 2, 4, 8],
			"regressor__n_estimators": [100, 150, 200],
		},
		n_iter=n_iter,
		cv=TimeSeriesSplit(n_splits=3),
		scoring="neg_mean_absolute_error",
		n_jobs=-1,
		random_state=42,
	)
	search.fit(features, np.log1p(target.clip(lower=0)))
	return RateModel(search.best_estimator_), search.best_params_


def predict(model: RateModel, features: pd.DataFrame) -> pd.Series:
	return model.predict(features)


def save_model(model: RateModel, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, path)


def load_model(path: Path) -> RateModel:
	return joblib.load(path)
