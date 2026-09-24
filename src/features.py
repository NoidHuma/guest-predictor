import pandas as pd


HOLIDAY_DATES = (
    "2024-01-01",
    "2024-01-02",
    "2024-01-03",
    "2024-01-04",
    "2024-01-05",
    "2024-01-06",
    "2024-01-07",
    "2024-01-08",
    "2024-02-14",
    "2024-03-08",
    "2024-05-01",
    "2024-05-09",
    "2024-06-12",
    "2024-11-04",
    "2024-12-31",
    "2025-01-01",
    "2025-01-02",
    "2025-01-03",
    "2025-01-04",
    "2025-01-05",
    "2025-01-06",
    "2025-01-07",
    "2025-01-08",
    "2025-02-14",
    "2025-03-08",
    "2025-05-01",
    "2025-05-09",
    "2025-06-12",
    "2025-11-04",
    "2025-12-31",
    "2026-01-01",
    "2026-01-02",
    "2026-01-03",
    "2026-01-04",
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
    "2026-01-08",
    "2026-02-14",
    "2026-03-08",
    "2026-05-01",
    "2026-05-09",
    "2026-06-12",
    "2026-11-04",
    "2026-12-31",
)

LAGS = (7, 14, 28)
ROLLING_WINDOWS = (7, 28)
ROLLING_SHIFT = 7


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    holiday_dates = pd.to_datetime(HOLIDAY_DATES)

    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["is_holiday"] = result["date"].isin(holiday_dates).astype(int)

    return result


def add_lag_features(
    df: pd.DataFrame,
    target: str = "guests",
) -> pd.DataFrame:
    result = df.copy()

    for lag in LAGS:
        result[f"lag_{lag}"] = result.groupby("restaurant_id")[target].shift(lag)

    return result


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "guests",
) -> pd.DataFrame:
    result = df.copy()
    shifted_target = result.groupby("restaurant_id")[target].shift(ROLLING_SHIFT)

    for window in ROLLING_WINDOWS:
        result[f"rolling_mean_{window}_lag_{ROLLING_SHIFT}"] = (
            shifted_target
            .groupby(result["restaurant_id"])
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return result


def make_features(
    df: pd.DataFrame,
    target: str = "guests",
) -> pd.DataFrame:
    result = df.sort_values(["restaurant_id", "date"]).reset_index(drop=True)
    result = add_calendar_features(result)
    result = add_lag_features(result, target=target)
    result = add_rolling_features(result, target=target)

    return result


def drop_rows_without_history(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in df.columns
        if column.startswith("lag_") or column.startswith("rolling_")
    ]

    return df.dropna(subset=feature_columns).reset_index(drop=True)
