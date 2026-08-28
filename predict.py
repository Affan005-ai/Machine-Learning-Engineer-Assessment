from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.clean import clean_data
from src.features import engineer_features
from src.model import load_model, predict


def predict_frame(model, frame: pd.DataFrame) -> pd.Series:
    features = engineer_features(
        clean_data(frame.drop(columns=["load_id"], errors="ignore"))
    )
    return predict(model, features)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate validation and December predictions."
    )
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/rate_model.joblib")
    )
    parser.add_argument("--validation", type=Path, default=Path("data/validation.csv"))
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("data/validation-predictions-template.csv"),
    )
    parser.add_argument(
        "--december", type=Path, default=Path("data/december-chart-inputs.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("validation_predictions.csv")
    )
    args = parser.parse_args()
    model = load_model(args.model)
    validation = pd.read_csv(args.validation)
    template = pd.read_csv(args.template)
    predictions = pd.DataFrame(
        {
            "load_id": validation["load_id"],
            "predicted_rate": predict_frame(model, validation),
        }
    )
    template[["load_id"]].merge(predictions, on="load_id", how="left").to_csv(
        args.output, index=False
    )
    december = pd.read_csv(args.december)
    december["predicted_rate"] = predict_frame(model, december)
    december.to_csv(args.december, index=False)
    print(f"Wrote {args.output} and updated {args.december}")


if __name__ == "__main__":
    main()
