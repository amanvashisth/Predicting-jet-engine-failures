"""Feature engineering and model training for FD001 RUL predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "Processed Data" / "train_features.csv"
ROLLING_WINDOW = 15
IDENTIFIER_COLUMNS = ["unit", "cycle"]
SETTING_COLUMNS = ["set1", "set2", "set3"]
SENSOR_COLUMNS = [
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s20",
    "s21",
]
RAW_COLUMNS = IDENTIFIER_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
TARGET_COLUMN = "RUL"
ROLLING_FEATURE_COLUMNS = [
    feature
    for sensor in SENSOR_COLUMNS
    for feature in (f"{sensor}_mean_15", f"{sensor}_std_15")
]
MODEL_FEATURE_COLUMNS = [
    *(column for column in RAW_COLUMNS if column != "unit"),
    *ROLLING_FEATURE_COLUMNS,
]
MODEL_PARAMETERS = {
    "objective": "regression",
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
    "subsample": 1.0,
    "reg_lambda": 0,
    "reg_alpha": 0,
    "num_leaves": 15,
    "n_estimators": 300,
    "min_child_samples": 50,
    "max_depth": 12,
    "learning_rate": 0.03,
    "colsample_bytree": 0.7,
}


def engineer_features(records: pd.DataFrame) -> pd.DataFrame:
    """Validate sensor history and create the notebook's 15-cycle statistics."""
    if records.empty:
        raise ValueError("Upload at least 15 cycles of engine history.")

    missing_columns = [column for column in RAW_COLUMNS if column not in records.columns]
    if missing_columns:
        raise ValueError(f"Missing required input columns: {', '.join(missing_columns)}")

    data = records.loc[:, RAW_COLUMNS].copy()
    for column in RAW_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.isna().any().any():
        invalid_columns = data.columns[data.isna().any()].tolist()
        raise ValueError(
            "Input values must be numeric and non-empty. Check columns: "
            + ", ".join(invalid_columns)
        )
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Input values must all be finite numbers.")

    for column in IDENTIFIER_COLUMNS:
        if (data[column] < 1).any() or not np.equal(data[column] % 1, 0).all():
            raise ValueError(f"'{column}' must contain positive whole numbers.")
        data[column] = data[column].astype("int64")

    if data.duplicated(IDENTIFIER_COLUMNS).any():
        raise ValueError("Each engine must have only one row per cycle.")

    data = data.sort_values(IDENTIFIER_COLUMNS).reset_index(drop=True)
    cycle_steps = data.groupby("unit", sort=False)["cycle"].diff().dropna()
    if not cycle_steps.eq(1).all():
        raise ValueError("Cycle history must be consecutive for each engine.")

    cycle_counts = data.groupby("unit", sort=False).size()
    if cycle_counts.lt(ROLLING_WINDOW).any():
        raise ValueError(
            "Provide at least 15 consecutive cycles per engine so the latest "
            "rolling features include the full window."
        )

    for sensor in SENSOR_COLUMNS:
        sensor_history = data.groupby("unit", sort=False)[sensor]
        data[f"{sensor}_mean_15"] = sensor_history.transform(
            lambda values: values.rolling(ROLLING_WINDOW, min_periods=1).mean()
        )
        data[f"{sensor}_std_15"] = sensor_history.transform(
            lambda values: values.rolling(ROLLING_WINDOW, min_periods=1)
            .std()
            .fillna(0)
        )

    return data


def train_model(
    training_data_path: Path = TRAINING_DATA_PATH,
) -> tuple[LGBMRegressor, list[str], float]:
    """Fit the notebook's tuned LightGBM model on all available training engines."""
    training_data = pd.read_csv(training_data_path)
    required_columns = [*MODEL_FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in training_data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Training data is missing required columns: " + ", ".join(missing_columns)
        )

    feature_columns = MODEL_FEATURE_COLUMNS
    features = training_data.loc[:, feature_columns]
    target = pd.to_numeric(training_data[TARGET_COLUMN], errors="coerce")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Training data contains missing or non-numeric features.")
    if not np.isfinite(features.to_numpy(dtype=float)).all() or not np.isfinite(
        target.to_numpy(dtype=float)
    ).all():
        raise ValueError("Training data contains non-finite values.")

    model = LGBMRegressor(**MODEL_PARAMETERS)
    model.fit(features, target)
    return model, feature_columns, float(target.max())


def predict_latest_cycles(
    records: pd.DataFrame,
    model: LGBMRegressor,
    feature_columns: list[str],
    max_rul: float,
) -> pd.DataFrame:
    """Engineer uploaded history and predict RUL for each engine's latest cycle."""
    engineered = engineer_features(records)
    latest_cycles = (
        engineered.groupby("unit", sort=True)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )
    features = latest_cycles.loc[:, feature_columns]
    predictions = np.clip(model.predict(features), 0, max_rul)

    return pd.DataFrame(
        {
            "unit": latest_cycles["unit"],
            "cycle": latest_cycles["cycle"],
            "predicted_RUL": predictions,
        }
    )
