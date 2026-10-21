# NASA Turbofan Engine RUL Prediction

🔗 **Live Demo:** [Try the App](https://amanvashisth-predicting-jet-engine-failures-app-lbkxc8.streamlit.app/)

A machine learning project that predicts the **Remaining Useful Life (RUL)** of turbofan engines using NASA's C-MAPSS FD001 dataset. It combines sensor data preprocessing, rolling statistical feature engineering, and LightGBM regression in an interactive Streamlit application.

## Project Overview

Predictive maintenance helps estimate when equipment may require maintenance before failure occurs. This project uses historical turbofan engine sensor readings to estimate the number of operating cycles remaining before an engine reaches its end of life.

The trained pipeline processes uploaded sensor history and generates an RUL prediction for the latest observed engine cycle.

## Key Features

- **RUL Prediction:** Estimates remaining engine operating cycles.
- **Data Preprocessing:** Validates and prepares raw engine sensor data.
- **Sensor Filtering:** Removes constant sensor features using training data.
- **Feature Engineering:** Generates rolling means and standard deviations over a 15-cycle window.
- **Machine Learning:** Uses LightGBM regression for RUL estimation.
- **Model Evaluation:** Evaluates predictions using RMSE and MAE.
- **Streamlit Deployment:** Provides a web interface for uploading sensor data and viewing predictions.
- **CSV Export:** Allows users to download prediction results.

## Dataset

**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation  
**Subset:** FD001

The FD001 dataset contains simulated turbofan engine run-to-failure data, including:

- 100 training engines
- 100 test engines
- 21 sensor measurements
- 3 operating-setting variables
- Engine unit identifiers and operating cycles
- One operating condition and one fault mode

The training data contains complete engine lifecycles, while the test data contains truncated engine histories. The model learns degradation patterns from the training data to estimate the remaining useful life of engines with incomplete histories.

## Tech Stack

- **Python** for model development
- **Pandas and NumPy** for data processing
- **Scikit-learn** for preprocessing and evaluation
- **LightGBM** for regression
- **Streamlit** for the web application
- **Jupyter Notebook** for exploratory data analysis and model development
- **Pickle** for saving the trained pipeline

## Project Structure

```text
Predicting Jet Engine Failure/
│
├── data/
│   ├── Processed Data/
│   │   ├── train_features.csv
│   │   └── test_features.csv
│   │
│   ├── Raw Data/
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   │
│   └── Test Data/
│       └── final_engine_20_cycles_all_21_sensors.csv
│
├── model/
│   └── rul_prediction_pipeline.pkl
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── ml_workflow.ipynb
│
├── app.py
├── README.md
└── requirements.txt
```

## Machine Learning Workflow

1. **Data Exploration:** Investigate engine operating cycles, sensor measurements, and degradation patterns.
2. **Data Preparation:** Load the raw FD001 files and organize engine readings into structured datasets.
3. **RUL Calculation:** Calculate training RUL from each engine's final cycle, with RUL capped at 125 cycles.
4. **Sensor Filtering:** Identify constant or uninformative sensors using the training data.
5. **Feature Engineering:** Generate rolling statistical features grouped by engine and ordered by cycle.
6. **Model Training:** Train regression models, including Linear Regression and LightGBM.
7. **Model Evaluation:** Compare model performance using RMSE and MAE.
8. **Pipeline Export:** Save preprocessing steps and the trained model in a single serialized pipeline.
9. **Deployment:** Load the saved pipeline in Streamlit to generate predictions from uploaded CSV files.

## Model Performance

The LightGBM model achieved the following reported results:

| Metric | Result |
|---|---:|
| Test RMSE | 17.84 cycles |
| Baseline RMSE | 41.72 cycles |
| RMSE Reduction | 57.2% |

The baseline uses a Dummy Regressor. The RMSE reduction is calculated relative to the baseline, provided both scores were obtained using the same test engines and evaluation procedure.

An RMSE of 17.84 cycles indicates the scale of prediction error under the RMSE metric. It does not guarantee the exact remaining life of an individual engine.

## How to Use the Application

1. Open the live demo.
2. Prepare a CSV file containing the engine's sensor history.
3. Include the required columns: `unit`, `cycle`, `set1`, `set2`, `set3`, and `s1` through `s21`.
4. Upload the CSV file through the application.
5. View the predicted RUL for the latest observed cycle.
6. Download the prediction result if the export option is available.

The application expects the input format used by the trained pipeline. Providing multiple recent readings helps the rolling feature engineering step capture short-term sensor patterns.

## Limitations

- The model is trained on FD001 and may not generalize to other C-MAPSS operating conditions or fault modes.
- C-MAPSS data is simulated and does not represent direct measurements from operational aircraft engines.
- RUL estimates are predictions, not guarantees of engine lifespan.
- Real-world maintenance decisions require additional validation and engineering assessment.

## Future Improvements

- Evaluate performance across additional C-MAPSS subsets.
- Compare additional regression and sequence-based models.
- Add uncertainty estimates to predictions.
- Visualize sensor trends and predicted degradation.
- Improve validation using engine-level splits and consistent final-cycle evaluation.

## Acknowledgments

This project uses NASA's C-MAPSS turbofan engine degradation dataset for educational and machine learning research purposes.
