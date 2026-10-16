"""Streamlit interface for engine remaining-useful-life predictions."""

import pandas as pd
import streamlit as st

from pipeline import RAW_COLUMNS, predict_latest_cycles, train_model


st.set_page_config(page_title="Jet Engine RUL Prediction", page_icon="✈️")
st.title("Jet Engine Remaining Useful Life")
st.write(
    "Upload sensor history to calculate 15-cycle rolling features and predict "
    "remaining useful life for the latest cycle of each engine."
)
st.info(
    "Include at least 15 consecutive cycles for every engine. For a new cycle, "
    "include its previous 14 cycles as well; this app does not retain history "
    "between uploads."
)


@st.cache_resource
def get_model():
    return train_model()


template = pd.DataFrame(columns=RAW_COLUMNS).to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV template",
    data=template,
    file_name="engine_sensor_history_template.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader("Upload engine history (CSV)", type=["csv"])
if uploaded_file is not None:
    try:
        records = pd.read_csv(uploaded_file)
    except pd.errors.ParserError as error:
        st.error(f"Could not read the uploaded CSV: {error}")
    else:
        st.caption(f"Uploaded {len(records):,} cycle records.")
        if st.button("Predict RUL", type="primary"):
            try:
                with st.spinner("Training the model and calculating predictions..."):
                    model, feature_columns, max_rul = get_model()
                    predictions = predict_latest_cycles(
                        records,
                        model,
                        feature_columns,
                        max_rul,
                    )
            except (OSError, ValueError) as error:
                st.error(str(error))
            else:
                st.success("Predictions ready.")
                st.dataframe(predictions, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download predictions",
                    data=predictions.to_csv(index=False).encode("utf-8"),
                    file_name="engine_rul_predictions.csv",
                    mime="text/csv",
                )
