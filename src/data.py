from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/dataset.csv")
PROCESSED_PATH = Path("data/processed/dataset.csv")

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


def fill_missing_by_calendar_median(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    filled = df[column].copy()
    day_of_week = df["date"].dt.dayofweek

    group_medians = (
        df.assign(day_of_week=day_of_week)
        .groupby(["restaurant_id", "day_of_week"])[column]
        .transform("median")
    )

    restaurant_medians = df.groupby("restaurant_id")[column].transform("median")
    global_median = df[column].median()

    filled = filled.fillna(group_medians)
    filled = filled.fillna(restaurant_medians)
    filled = filled.fillna(global_median)

    return filled


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["guests"] = fill_missing_by_calendar_median(result, "guests")
    result["revenue"] = fill_missing_by_calendar_median(result, "revenue")

    return result


def cap_outliers(
    df: pd.DataFrame,
    column: str,
    iqr_multiplier: float = 3.0,
) -> pd.DataFrame:
    result = df.copy()

    for _, group in result.groupby("restaurant_id"):
        values = group[column].dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        upper_bound = q3 + iqr_multiplier * iqr

        result.loc[group.index, column] = result.loc[group.index, column].clip(
            upper=upper_bound
        )

    return result


def apply_processed_schema(df: pd.DataFrame) -> pd.DataFrame:
    result = df.loc[:, list(PROCESSED_SCHEMA)].copy()

    result["date"] = result["date"].astype(PROCESSED_SCHEMA["date"])
    result["restaurant_id"] = result["restaurant_id"].astype(PROCESSED_SCHEMA["restaurant_id"])
    result["guests"] = np.rint(result["guests"]).astype(PROCESSED_SCHEMA["guests"])
    result["revenue"] = result["revenue"].round(2).astype(PROCESSED_SCHEMA["revenue"])

    return result


def prepare_daily_data(df: pd.DataFrame) -> Dataset:
    prepared = restore_daily_calendar(df)
    prepared = fill_missing_values(prepared)
    prepared = cap_outliers(prepared, "guests")
    prepared = cap_outliers(prepared, "revenue")
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