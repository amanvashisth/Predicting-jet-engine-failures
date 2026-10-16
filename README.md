# Predicting-jet-engine-failures
A predictive maintenance system that reads engine sensor measurements, estimates remaining useful life (RUL) in flight cycles, and raises an alert when an engine may be approaching failure.

## Run the prediction site

Install the project dependencies and start the Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app trains the notebook's tuned LightGBM regressor from
`data/Processed Data/train_features.csv` the first time a prediction is
requested. Upload a CSV containing engine history with these columns:
`unit`, `cycle`, `set1`, `set2`, `set3`, and `s2`, `s3`, `s4`, `s5`, `s6`,
`s7`, `s8`, `s9`, `s11`, `s12`, `s13`, `s14`, `s15`, `s16`, `s17`, `s20`,
`s21`.

Include at least 15 consecutive cycles per engine. The app computes the same
15-cycle rolling sensor means and sample standard deviations used by the
notebook, then predicts RUL for each engine's latest cycle. For a newly
observed cycle, include the previous 14 cycles in the upload so its rolling
features have the right history. The app is stateless and does not store
uploaded sensor readings between visits.
