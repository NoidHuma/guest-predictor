import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def time_split(
    df: pd.DataFrame,
    valid_weeks: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].max() - pd.Timedelta(weeks=valid_weeks)

    train = df[df["date"].le(cutoff)].copy()
    valid = df[df["date"].gt(cutoff)].copy()

    if train.empty or valid.empty:
        raise ValueError("Not enough data for time split")

    return train, valid


def mean_absolute_percentage_error_safe(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    non_zero_mask = y_true_array != 0

    if not non_zero_mask.any():
        return np.nan

    return np.mean(
        np.abs(
            (y_true_array[non_zero_mask] - y_pred_array[non_zero_mask])
            / y_true_array[non_zero_mask]
        )
    )


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error_safe(y_true, y_pred),
    }


def evaluate_baseline(
    valid_df: pd.DataFrame,
    target: str = "guests",
    prediction_column: str = "lag_7",
) -> dict[str, float]:
    if prediction_column not in valid_df.columns:
        raise ValueError(f"Missing baseline prediction column: {prediction_column}")

    valid = valid_df.dropna(subset=[target, prediction_column])

    if valid.empty:
        raise ValueError("No rows available for baseline evaluation")

    return calculate_metrics(
        y_true=valid[target],
        y_pred=valid[prediction_column],
    )


def metrics_to_frame(
    metrics_by_model: dict[str, dict[str, float]],
) -> pd.DataFrame:
    return (
        pd.DataFrame.from_dict(metrics_by_model, orient="index")
        .rename_axis("model")
        .reset_index()
        .sort_values("mae")
        .reset_index(drop=True)
    )