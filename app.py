
import os
import re
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Jet Engine RUL Prediction",
    page_icon="✈️",
    layout="wide"
)


# =========================================================
# 2. EXPECTED INPUT COLUMNS
# =========================================================

FEATURE_COLUMNS = (
    ["unit", "cycle", "set1", "set2", "set3"]
    + [f"s{i}" for i in range(1, 22)]
)

SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]


# =========================================================
# 3. CUSTOM TRANSFORMERS
# These definitions allow pickle to find classes saved
# previously under the __main__ module.
# Keep them compatible with the original definitions.
# =========================================================

class InputValidator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        normalised = {
            col: re.sub(r"[^a-z0-9]", "", str(col).lower())
            for col in X.columns
        }

        reverse_map = {
            re.sub(r"[^a-z0-9]", "", col.lower()): col
            for col in FEATURE_COLUMNS
        }

        recognised = {
            col: reverse_map[name]
            for col, name in normalised.items()
            if name in reverse_map
        }

        if len(recognised) == len(FEATURE_COLUMNS):
            X = X.rename(columns=recognised)

        elif X.shape[1] == len(FEATURE_COLUMNS):
            X.columns = FEATURE_COLUMNS

        else:
            raise ValueError(
                f"Expected {len(FEATURE_COLUMNS)} columns, "
                f"but received {X.shape[1]}."
            )

        if X.columns.duplicated().any():
            raise ValueError("Duplicate column names detected.")

        missing = set(FEATURE_COLUMNS) - set(X.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        X = X[FEATURE_COLUMNS].copy()

        for col in FEATURE_COLUMNS:
            X[col] = pd.to_numeric(X[col], errors="raise")

        if X.isna().any().any():
            raise ValueError("Missing values detected.")

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Infinite values detected.")

        if (X["cycle"] < 1).any():
            raise ValueError("Cycle values must be positive.")

        if X.duplicated(["unit", "cycle"]).any():
            raise ValueError("Duplicate engine-cycle records detected.")

        return X


class SensorFilter(BaseEstimator, TransformerMixin):

    def __init__(self, variance_threshold=0.0):
        self.variance_threshold = variance_threshold

    def fit(self, X, y=None):
        sensors = [c for c in SENSOR_COLUMNS if c in X.columns]

        variances = X[sensors].var(ddof=0)

        self.selected_sensors_ = variances[
            variances > self.variance_threshold
        ].index.tolist()

        self.dropped_sensors_ = [
            s for s in sensors
            if s not in self.selected_sensors_
        ]

        if not self.selected_sensors_:
            raise ValueError("No usable sensors remain.")

        return self

    def transform(self, X):
        non_sensor_columns = [
            c for c in X.columns
            if c not in SENSOR_COLUMNS
        ]

        return X[
            non_sensor_columns + self.selected_sensors_
        ].copy()


class RollingFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, window=15):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["_original_order"] = np.arange(len(X))

        X = X.sort_values(
            ["unit", "cycle"],
            kind="stable"
        ).copy()

        sensors = [
            c for c in SENSOR_COLUMNS if c in X.columns
        ]

        for sensor in sensors:
            grouped = X.groupby("unit", sort=False)[sensor]

            X[f"{sensor}_rolling_mean"] = grouped.transform(
                lambda s: s.rolling(
                    window=self.window,
                    min_periods=1
                ).mean()
            )

            X[f"{sensor}_rolling_std"] = grouped.transform(
                lambda s: s.rolling(
                    window=self.window,
                    min_periods=1
                ).std()
            ).fillna(0)

        X = X.sort_values("_original_order", kind="stable")
        X = X.drop(columns="_original_order")

        return X.reset_index(drop=True)


class DropEngineID(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=["unit"])


# =========================================================
# 4. MODEL PATH AND LOADING
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "rul_prediction_pipeline.pkl"
)


@st.cache_resource
def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


# =========================================================
# 5. APP HEADER
# =========================================================

st.title("✈️ Jet Engine Remaining Useful Life Prediction")

st.write(
    "Upload sensor history to calculate 15-cycle rolling features and predict remaining useful life for the latest cycle of each engine."
)

st.info(
    "Include at least 15 consecutive cycles for every engine. " 
    "For a new cycle, include its previous 14 cycles as well; this app does not retain history between uploads."
)

# =========================================================
# CSV TEMPLATE DOWNLOAD
# =========================================================

FEATURE_COLUMNS = (
    ["unit", "cycle", "set1", "set2", "set3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Create a blank template with 20 rows
template = pd.DataFrame(columns=FEATURE_COLUMNS)

template_csv = template.to_csv(index=False).encode("utf-8")

st.subheader("Download CSV Template")

st.write(
    "Download the template, enter 20 readings for one engine, "
    "and upload the completed CSV below."
)

st.download_button(
    label="Download CSV Template",
    data=template_csv,
    file_name="engine_sensor_template.csv",
    mime="text/csv"
)


# =========================================================
# 6. LOAD THE SAVED PIPELINE
# =========================================================

try:
    rul_pipeline = load_model(MODEL_PATH)
    st.success("Jet Engine RUL Prediction Model is Ready for use!")

except Exception as exc:
    st.error(
        "Could not load the Jet Engine RUL Prediction Model. Confirm that "
        "model/rul_prediction_pipeline.pkl exists and that "
        "the saved transformer classes match this app."
    )
    st.exception(exc)
    st.stop()


# =========================================================
# 7. UPLOAD CSV
# =========================================================

uploaded_file = st.file_uploader(
    "Upload engine readings (CSV)",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        user_data = pd.read_csv(uploaded_file)

        if user_data.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        # Normalize column names for common formats such as
        # "Set 1", "set_1", and "set1".
        normalised = {
            col: re.sub(r"[^a-z0-9]", "", str(col).lower())
            for col in user_data.columns
        }

        reverse_map = {
            re.sub(r"[^a-z0-9]", "", col.lower()): col
            for col in FEATURE_COLUMNS
        }

        rename_map = {
            col: reverse_map[name]
            for col, name in normalised.items()
            if name in reverse_map
        }

        if len(rename_map) == len(FEATURE_COLUMNS):
            user_data = user_data.rename(columns=rename_map)

        if user_data.columns.duplicated().any():
            st.error("Duplicate column names detected.")
            st.stop()

        missing = set(FEATURE_COLUMNS) - set(user_data.columns)
        extra = set(user_data.columns) - set(FEATURE_COLUMNS)

        if missing or extra:
            if missing:
                st.error(f"Missing columns: {sorted(missing)}")

            if extra:
                st.error(f"Unexpected columns: {sorted(extra)}")

            st.stop()

        # Enforce the same input column order expected by training.
        user_data = user_data[FEATURE_COLUMNS].copy()

        for col in FEATURE_COLUMNS:
            user_data[col] = pd.to_numeric(
                user_data[col],
                errors="raise"
            )

        if user_data.isna().any().any():
            st.error("Missing values found in the uploaded data.")
            st.stop()

        if not np.isfinite(
            user_data.to_numpy(dtype=float)
        ).all():
            st.error("Infinite values found in the uploaded data.")
            st.stop()

        if (user_data["cycle"] < 1).any():
            st.error("Cycle values must be positive.")
            st.stop()

        if user_data.duplicated(["unit", "cycle"]).any():
            st.error("Duplicate engine-cycle records found.")
            st.stop()

        if user_data["unit"].nunique() != 1:
            st.error(
                "Please upload readings for one engine at a time."
            )
            st.stop()

        # Sort so the newest cycle is the final row.
        user_data = user_data.sort_values(
            "cycle"
        ).reset_index(drop=True)

        st.subheader("Uploaded sensor history")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Engine ID",
            int(user_data["unit"].iloc[0])
        )

        col2.metric(
            "Number of readings",
            len(user_data)
        )

        col3.metric(
            "Latest cycle",
            int(user_data["cycle"].iloc[-1])
        )

        st.dataframe(
            user_data,
            use_container_width=True
        )

        if len(user_data) < 15:
            st.warning(
                "Fewer than 15 readings were provided. "
                "Rolling features will use the available history."
            )

        # =================================================
        # 8. PREDICT RUL
        # =================================================

        if st.button(
            "Predict Remaining Useful Life",
            type="primary"
        ):

            with st.spinner("Generating prediction..."):

                # The saved pipeline performs preprocessing.
                predictions = rul_pipeline.predict(user_data)

                # The last row represents the latest cycle.
                predicted_rul = float(predictions[-1])

                st.subheader("Prediction Result")

                result_col1, result_col2 = st.columns(2)

                result_col1.metric(
                    "Predicted RUL",
                    f"{predicted_rul:.2f} cycles"
                )

                result_col2.metric(
                    "Latest observed cycle",
                    int(user_data["cycle"].iloc[-1])
                )

                st.caption(
                    "This is a model estimate, not a guarantee "
                    "of when an engine will fail. It should not "
                    "replace validated engineering safety procedures."
                )

                result = pd.DataFrame({
                    "unit": [
                        int(user_data["unit"].iloc[-1])
                    ],
                    "latest_cycle": [
                        int(user_data["cycle"].iloc[-1])
                    ],
                    "predicted_RUL_cycles": [predicted_rul]
                })

                st.download_button(
                    "Download prediction result",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="rul_prediction.csv",
                    mime="text/csv"
                )

    except Exception as exc:
        st.error(
            "An error occurred while processing the CSV "
            "or generating the prediction."
        )
        st.exception(exc)


# =========================================================
# 9. FOOTER
# =========================================================

st.divider()

st.caption(
    "Predicting Jet Engine Failure | NASA C-MAPSS FD001"
)
