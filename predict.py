import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import RAW_PATH, load_raw, prepare_daily_data
from src.features import make_features
from src.model import FEATURE_COLUMNS, MODEL_PATH, load_model, predict_model


FORECAST_HORIZON = 7


def parse_date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).normalize()
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid date: {value}. Use YYYY-MM-DD."
        ) from error


def create_future_rows(
    restaurant_id: int,
    forecast_start: pd.Timestamp,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                start=forecast_start,
                periods=horizon,
                freq="D",
            ),
            "restaurant_id": restaurant_id,
            "guests": np.nan,
            "revenue": np.nan,
            "is_forecast": True,
        }
    )


def build_forecast_features(
    history: pd.DataFrame,
    restaurant_id: int,
    forecast_start: pd.Timestamp,
) -> pd.DataFrame:
    restaurant_history = history[
        history["restaurant_id"].eq(restaurant_id)
    ].copy()

    if restaurant_history.empty:
        raise ValueError(f"Unknown restaurant_id: {restaurant_id}")

    restaurant_history = restaurant_history[
        restaurant_history["date"].lt(forecast_start)
    ].copy()

    if len(restaurant_history) < 34:
        raise ValueError(
            "Not enough history for forecast. "
            "At least 34 daily observations are required."
        )

    restaurant_history["is_forecast"] = False

    future_rows = create_future_rows(
        restaurant_id=restaurant_id,
        forecast_start=forecast_start,
    )

    combined = pd.concat(
        [restaurant_history, future_rows],
        ignore_index=True,
    )

    features = make_features(combined)
    forecast_features = features[features["is_forecast"]].copy()

    missing_features = forecast_features.loc[:, FEATURE_COLUMNS].isna().any()
    missing_feature_names = missing_features[missing_features].index.tolist()

    if missing_feature_names:
        raise ValueError(
            f"Cannot build forecast features: {missing_feature_names}"
        )

    return forecast_features


def make_forecast(
    history: pd.DataFrame,
    model,
    restaurant_id: int,
    forecast_start: pd.Timestamp,
) -> pd.DataFrame:
    forecast_features = build_forecast_features(
        history=history,
        restaurant_id=restaurant_id,
        forecast_start=forecast_start,
    )

    predictions = predict_model(model, forecast_features)

    forecast = forecast_features.loc[:, ["date", "restaurant_id"]].copy()
    forecast["guests_prediction"] = (
        predictions.clip(lower=0).round().astype(int)
    )

    return forecast.reset_index(drop=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast restaurant guest flow for the next 7 days."
    )

    parser.add_argument(
        "--date",
        required=True,
        type=parse_date,
        help="First forecast date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--restaurant",
        required=True,
        type=int,
        help="Restaurant identifier.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_PATH,
        help="Path to the raw history CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the forecast CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    raw = load_raw(args.input)
    raw_history = raw[raw["date"].lt(args.date)].copy()

    if raw_history.empty:
        raise ValueError(
            f"No observations before forecast date: {args.date.date()}"
        )

    dataset = prepare_daily_data(raw_history)
    model = load_model(MODEL_PATH)

    forecast = make_forecast(
        history=dataset.frame,
        model=model,
        restaurant_id=args.restaurant,
        forecast_start=args.date,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(args.output, index=False)

    print(forecast.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(f"Ошибка: {error}")
        raise SystemExit(1)