import numpy as np
import pandas as pd
import pytest

from src.features import make_features


@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01", periods=40, freq="D")
    guests = np.arange(1, 41)
    return pd.DataFrame({
        "date": dates,
        "restaurant_id": 1,
        "guests": guests,
        "revenue": guests * 100.0,
    })


@pytest.fixture
def feature_columns(sample_data: pd.DataFrame) -> list[str]:
    df = make_features(sample_data, target="guests")
    original = {"date", "restaurant_id", "guests", "revenue"}
    return [c for c in df.columns if c not in original]


def test_expected_columns_present(sample_data: pd.DataFrame):
    df = make_features(sample_data, target="guests")

    expected = {
        "day_of_week", "month", "is_holiday",
        "lag_7", "lag_14", "lag_28",
        "rolling_mean_7_lag_7", "rolling_mean_28_lag_7",
    }
    missing = expected - set(df.columns)
    assert not missing, f"Пропущены признаки: {sorted(missing)}"


def test_lag_7_matches_shift(sample_data: pd.DataFrame):
    df_feat = make_features(sample_data, target="guests")

    guests_sorted = sample_data.sort_values("date")["guests"].reset_index(drop=True)
    expected_lag_7 = guests_sorted.shift(7)

    np.testing.assert_allclose(
        df_feat["lag_7"].to_numpy(),
        expected_lag_7.to_numpy(),
        equal_nan=True,
    )


def test_warmup_nans(sample_data: pd.DataFrame):
    df = make_features(sample_data, target="guests")

    assert df["lag_7"].iloc[:7].isna().all()
    assert not df["lag_7"].iloc[7:].isna().any()


def test_rolling_features_are_lagged(sample_data: pd.DataFrame):
    df_orig = make_features(sample_data, target="guests")

    target_day = pd.Timestamp("2024-02-05")
    mutation_day = pd.Timestamp("2024-02-01")

    mutated = sample_data.copy()
    mutated.loc[mutated["date"] == mutation_day, "guests"] = 9999
    df_mut = make_features(mutated, target="guests")

    for col in ["rolling_mean_7_lag_7", "rolling_mean_28_lag_7"]:
        val_orig = df_orig.loc[df_orig["date"] == target_day, col].iloc[0]
        val_mut = df_mut.loc[df_mut["date"] == target_day, col].iloc[0]

        assert not pd.isna(val_orig), f"{col} на {target_day.date()} не должен быть NaN"
        assert val_orig == val_mut, (
            f"{col} изменился от мутации в {mutation_day.date()} — rolling не сдвинут"
        )


def test_sorting_is_handled(sample_data: pd.DataFrame):
    shuffled = sample_data.sample(frac=1, random_state=42).reset_index(drop=True)

    df = make_features(shuffled, target="guests")

    assert df["date"].is_monotonic_increasing


def test_no_target_column_as_feature(
    sample_data: pd.DataFrame,
    feature_columns: list[str],
):
    calendar_cols = {
        "day_of_week",
        "month",
        "month_sin",
        "month_cos",
        "is_holiday",
    }
    for col in feature_columns:
        is_lag = col.startswith("lag_") or col.endswith("_lag_7") or col.endswith("_lag_1")
        is_calendar = col in calendar_cols
        assert is_lag or is_calendar, f"Неожиданный признак: {col}"