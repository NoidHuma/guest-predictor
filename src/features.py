import pandas as pd


HOLIDAY_MONTH_DAY = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 14),
    (3, 8),
    (5, 1),
    (5, 9),
    (6, 12),
    (11, 4),
    (12, 31),
)

LAGS = (7, 14, 28)
ROLLING_WINDOWS = (7, 28)
ROLLING_SHIFT = 7


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month

    holiday_index = pd.MultiIndex.from_tuples(HOLIDAY_MONTH_DAY)
    month_day = pd.MultiIndex.from_arrays(
        [result["date"].dt.month, result["date"].dt.day]
    )
    result["is_holiday"] = month_day.isin(holiday_index).astype(int)

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
