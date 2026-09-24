from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/dataset.csv")
PROCESSED_PATH = Path("data/processed/dataset.csv")

DEFAULT_VALID_WEEKS = 6

RAW_SCHEMA = {
    "date": "datetime64[ns]",
    "restaurant_id": "int64",
    "guests": "float64",
    "revenue": "float64",
}

PROCESSED_SCHEMA = {
    "date": "datetime64[ns]",
    "restaurant_id": "int64",
    "guests": "int64",
    "revenue": "float64",
}


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    target: str = "guests"


@dataclass(frozen=True)
class ImputationStats:
    by_group: pd.Series
    by_restaurant: pd.Series
    global_value: float


@dataclass(frozen=True)
class CapBounds:
    upper_by_restaurant: pd.Series


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])

    missing_columns = set(RAW_SCHEMA) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    df = df.loc[:, list(RAW_SCHEMA)].copy()
    df["restaurant_id"] = df["restaurant_id"].astype(RAW_SCHEMA["restaurant_id"])
    df["guests"] = pd.to_numeric(df["guests"], errors="coerce").astype(RAW_SCHEMA["guests"])
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").astype(RAW_SCHEMA["revenue"])

    duplicate_count = df.duplicated(["restaurant_id", "date"]).sum()
    if duplicate_count:
        raise ValueError(f"Found duplicated restaurant-date rows: {duplicate_count}")

    return df.sort_values(["restaurant_id", "date"]).reset_index(drop=True)


def restore_daily_calendar(df: pd.DataFrame) -> pd.DataFrame:
    restored = []

    for restaurant_id, group in df.groupby("restaurant_id"):
        full_dates = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        calendar = pd.DataFrame(
            {
                "restaurant_id": restaurant_id,
                "date": full_dates,
            }
        )

        restored.append(
            calendar.merge(
                group,
                on=["restaurant_id", "date"],
                how="left",
            )
        )

    return (
        pd.concat(restored, ignore_index=True)
        .sort_values(["restaurant_id", "date"])
        .reset_index(drop=True)
    )


def fit_imputation_stats(df: pd.DataFrame, column: str) -> ImputationStats:
    day_of_week = df["date"].dt.dayofweek

    by_group = (
        df.assign(day_of_week=day_of_week)
        .groupby(["restaurant_id", "day_of_week"])[column]
        .median()
    )
    by_restaurant = df.groupby("restaurant_id")[column].median()

    return ImputationStats(
        by_group=by_group,
        by_restaurant=by_restaurant,
        global_value=float(df[column].median()),
    )


def apply_imputation(
    df: pd.DataFrame,
    stats: ImputationStats,
    column: str,
) -> pd.Series:
    day_of_week = df["date"].dt.dayofweek
    keys = pd.MultiIndex.from_arrays([df["restaurant_id"], day_of_week])

    group_values = pd.Series(
        stats.by_group.reindex(keys).to_numpy(),
        index=df.index,
    )
    restaurant_values = df["restaurant_id"].map(stats.by_restaurant)

    filled = df[column].copy()
    filled = filled.fillna(group_values)
    filled = filled.fillna(restaurant_values)
    filled = filled.fillna(stats.global_value)

    return filled


def fit_cap_bounds(
    df: pd.DataFrame,
    column: str,
    iqr_multiplier: float = 3.0,
) -> CapBounds:
    upper_by_restaurant = {}

    for restaurant_id, group in df.groupby("restaurant_id"):
        values = group[column].dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            upper_by_restaurant[restaurant_id] = np.inf
        else:
            upper_by_restaurant[restaurant_id] = q3 + iqr_multiplier * iqr

    return CapBounds(upper_by_restaurant=pd.Series(upper_by_restaurant))


def apply_cap(
    df: pd.DataFrame,
    bounds: CapBounds,
    column: str,
) -> pd.Series:
    upper = df["restaurant_id"].map(bounds.upper_by_restaurant)
    return df[column].clip(upper=upper)


def apply_processed_schema(df: pd.DataFrame) -> pd.DataFrame:
    result = df.loc[:, list(PROCESSED_SCHEMA)].copy()

    result["date"] = result["date"].astype(PROCESSED_SCHEMA["date"])
    result["restaurant_id"] = result["restaurant_id"].astype(PROCESSED_SCHEMA["restaurant_id"])
    result["guests"] = np.rint(result["guests"]).astype(PROCESSED_SCHEMA["guests"])
    result["revenue"] = result["revenue"].round(2).astype(PROCESSED_SCHEMA["revenue"])

    return result


def prepare_daily_data(
    df: pd.DataFrame,
    valid_weeks: int = DEFAULT_VALID_WEEKS,
) -> Dataset:
    restored = restore_daily_calendar(df)

    cutoff = restored["date"].max() - pd.Timedelta(weeks=valid_weeks)
    train_raw = restored[restored["date"].le(cutoff)]

    if train_raw.empty:
        raise ValueError("Not enough history to fit imputation statistics")

    guests_stats = fit_imputation_stats(train_raw, "guests")
    revenue_stats = fit_imputation_stats(train_raw, "revenue")
    guests_bounds = fit_cap_bounds(train_raw, "guests")
    revenue_bounds = fit_cap_bounds(train_raw, "revenue")

    prepared = restored.copy()
    prepared["guests"] = apply_imputation(prepared, guests_stats, "guests")
    prepared["revenue"] = apply_imputation(prepared, revenue_stats, "revenue")
    prepared["guests"] = apply_cap(prepared, guests_bounds, "guests")
    prepared["revenue"] = apply_cap(prepared, revenue_bounds, "revenue")
    prepared = apply_processed_schema(prepared)

    prepared = prepared.sort_values(["restaurant_id", "date"]).reset_index(drop=True)

    return Dataset(frame=prepared)


def save_processed(
    dataset: Dataset,
    path: str | Path = PROCESSED_PATH,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_csv(path, index=False)


def main() -> None:
    raw = load_raw(RAW_PATH)
    dataset = prepare_daily_data(raw)
    save_processed(dataset, PROCESSED_PATH)


if __name__ == "__main__":
    main()