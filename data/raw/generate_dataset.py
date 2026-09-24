from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
START_DATE = "2024-01-01"
END_DATE = "2026-09-30"
OUTPUT_PATH = Path("data/raw/dataset.csv")


RESTAURANTS = [
    {
        "restaurant_id": 1,
        "base_guests": 92,
        "base_avg_check": 1450,
        "daily_trend": 0.0001,
        "noise": 0.12,
        "closed_dow": [],
        "weekly": [0.70, 0.76, 0.88, 1.00, 1.24, 1.42, 1.12],
    },
    {
        "restaurant_id": 2,
        "base_guests": 76,
        "base_avg_check": 1180,
        "daily_trend": 0.00015,
        "noise": 0.14,
        "closed_dow": [0],
        "weekly": [0.00, 0.82, 0.92, 1.03, 1.20, 1.36, 1.18],
    },
    {
        "restaurant_id": 3,
        "base_guests": 58,
        "base_avg_check": 1680,
        "daily_trend": -0.00005,
        "noise": 0.16,
        "closed_dow": [],
        "weekly": [0.72, 0.80, 0.94, 1.08, 1.34, 1.50, 0.96],
    },
]


HOLIDAY_DATES = (
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


def build_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    calendar = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    holiday_dates = pd.to_datetime(HOLIDAY_DATES)

    calendar["holiday_multiplier"] = np.where(
        calendar["date"].isin(holiday_dates),
        1.25,
        1.0,
    )

    return calendar


def generate_restaurant_frame(
    restaurant: dict,
    calendar: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    frame = calendar.copy()
    frame["restaurant_id"] = restaurant["restaurant_id"]

    day_number = np.arange(len(frame))
    dow = frame["date"].dt.dayofweek.to_numpy()
    doy = frame["date"].dt.dayofyear.to_numpy()

    weekly = np.take(np.array(restaurant["weekly"], dtype=float), dow)
    yearly = 1.0 + 0.10 * np.sin(2 * np.pi * (doy - 35) / 365.25)
    summer = 1.0 + 0.06 * np.sin(2 * np.pi * (doy - 160) / 365.25)
    trend = 1.0 + restaurant["daily_trend"] * day_number
    noise = rng.lognormal(mean=0.0, sigma=restaurant["noise"], size=len(frame))

    expected_guests = (
        restaurant["base_guests"]
        * weekly
        * yearly
        * summer
        * trend
        * frame["holiday_multiplier"].to_numpy()
        * noise
    )

    guests = rng.poisson(np.maximum(expected_guests, 1.0)).astype(int)

    closed_mask = frame["date"].dt.dayofweek.isin(restaurant["closed_dow"]).to_numpy(dtype=bool, copy=True)

    maintenance_candidates = np.flatnonzero(~closed_mask)
    maintenance_count = max(2, int(len(frame) * 0.006))
    maintenance_idx = rng.choice(maintenance_candidates, size=maintenance_count, replace=False)
    closed_mask[maintenance_idx] = True

    if restaurant["restaurant_id"] == 3:
        new_year_mask = (frame["date"].dt.month.eq(1) & frame["date"].dt.day.eq(1)).to_numpy()
        closed_mask = closed_mask | new_year_mask

    anomaly_candidates = np.flatnonzero(~closed_mask)
    anomaly_count = max(5, int(len(frame) * 0.014))
    anomaly_idx = rng.choice(anomaly_candidates, size=anomaly_count, replace=False)
    anomaly_multiplier = rng.choice([0.25, 0.35, 2.15, 2.75], size=anomaly_count)

    guests[anomaly_idx] = np.maximum(0, np.rint(guests[anomaly_idx] * anomaly_multiplier)).astype(int)
    guests[closed_mask] = 0

    avg_check_noise = rng.lognormal(mean=0.0, sigma=0.08, size=len(frame))
    weekend_uplift = np.where(np.isin(dow, [4, 5, 6]), 1.08, 1.0)

    avg_check = restaurant["base_avg_check"] * avg_check_noise * weekend_uplift
    avg_check = np.rint(avg_check).astype(int)

    revenue_noise = rng.lognormal(mean=0.0, sigma=0.06, size=len(frame))
    revenue = np.rint(guests * avg_check * revenue_noise).astype(float)

    revenue_anomaly_count = max(3, int(len(frame) * 0.006))
    revenue_anomaly_idx = rng.choice(anomaly_candidates, size=revenue_anomaly_count, replace=False)
    revenue[revenue_anomaly_idx] = revenue[revenue_anomaly_idx] * rng.choice(
        [0.45, 1.85],
        size=revenue_anomaly_count,
    )

    revenue[closed_mask] = 0.0

    frame["guests"] = guests
    frame["revenue"] = np.round(revenue, 2)

    missing_candidates = frame.index[~closed_mask].to_numpy()
    missing_count = max(8, int(len(frame) * 0.018))
    missing_idx = rng.choice(missing_candidates, size=missing_count, replace=False)

    frame = frame.drop(index=missing_idx)

    partial_missing_candidates = frame.index[
        frame["guests"].gt(0) & frame["revenue"].gt(0)
        ].to_numpy()

    guests_missing_count = max(3, int(len(frame) * 0.004))
    revenue_missing_count = max(3, int(len(frame) * 0.005))

    guests_missing_idx = rng.choice(
        partial_missing_candidates,
        size=guests_missing_count,
        replace=False,
    )

    revenue_missing_candidates = np.setdiff1d(
        partial_missing_candidates,
        guests_missing_idx,
    )

    revenue_missing_idx = rng.choice(
        revenue_missing_candidates,
        size=revenue_missing_count,
        replace=False,
    )

    frame.loc[guests_missing_idx, "guests"] = np.nan
    frame.loc[revenue_missing_idx, "revenue"] = np.nan

    return frame[
        [
            "date",
            "restaurant_id",
            "guests",
            "revenue",
        ]
    ]


def generate_raw_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    calendar = build_calendar(START_DATE, END_DATE)

    frames = [
        generate_restaurant_frame(restaurant, calendar, rng)
        for restaurant in RESTAURANTS
    ]

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["restaurant_id", "date"]).reset_index(drop=True)

    dataset["date"] = dataset["date"].dt.strftime("%Y-%m-%d")
    dataset["revenue"] = dataset["revenue"].round(2)

    return dataset


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_raw_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()