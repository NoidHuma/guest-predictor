from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_PATH = Path("models/ridge_model.joblib")

TARGET_COLUMN = "guests"

NUMERIC_FEATURES = (
    "month",
    "is_weekend",
    "is_holiday",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7_lag_7",
    "rolling_mean_28_lag_7",
)

CATEGORICAL_FEATURES = (
    "restaurant_id",
    "day_of_week",
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def train_model(
    train_df: pd.DataFrame,
    target: str = TARGET_COLUMN,
) -> Pipeline:
    missing_features = set(FEATURE_COLUMNS) - set(train_df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {sorted(missing_features)}")

    X_train = train_df.loc[:, FEATURE_COLUMNS]
    y_train = train_df[target]

    model = build_model()
    model.fit(X_train, y_train)

    return model


def predict_model(
    model: Pipeline,
    df: pd.DataFrame,
) -> pd.Series:
    missing_features = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {sorted(missing_features)}")

    predictions = model.predict(df.loc[:, FEATURE_COLUMNS])

    return pd.Series(predictions, index=df.index, name="prediction")


def save_model(
    model: Pipeline,
    path: str | Path = MODEL_PATH,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(
    path: str | Path = MODEL_PATH,
) -> Pipeline:
    return joblib.load(path)