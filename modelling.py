import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor # For a simpler baseline
import os

# # Step by step
# inital git repository, create venv, install base dependencies, create requirement.txt, simulate data read and preprocessing, versioned the data
# training model, save the run, version the model + MLFlow logging
# Assumption here is your model was saved, then load the .py, define dependencies, create serving scripts (input/output format, preprocessing/post-processing), run locally to start web server, test API to your local host, conatinerize with docker
# Infrastructure provisioning: setting up necessary compute resources, real-time inference (immediate prediction) use dockerized fastAPI application + scalability configure auto-scaling + load balancing , batch scheduler for making prediction on large dataset (like run_batch_prediction in app/main.py). THE PROCESS ARE scheduler triggers the job, job loads the model, job read the large dataset, perform batch prediction, saves result to data store, streaming inference for continous prediction such as applicatio nconnect to data stream + immediate pre-processed + predict + written to another stream and real-time database
# Monitor and logging: proper loggin that sends log to centralized logging system, deatils to log such as timestamp request + unique request id + input feature + raw model output + final prediction + any error or wanring + model version used, for data drift detection such as collect statistic such as  mean, variance, quartiles, distribution of incoming prediction data + compare statistics + use statistic test to detect significant, conceptual drift detection such as compare model actual performance and historical performance + analyze prediction confidence + actual outcome, alert CT team such integrate monitoring tools with alert system that triggers from these detection
# Automated things: singal output for retraining such as these detection + schedule retraining, orchestration by github actions/jenkins + pipeline Data Ingestion: Pull the latest, most relevant data (including new ground truth if available).
# - Data Validation: Ensure the new data conforms to expected schema and quality.
# - Data Preprocessing: Apply the same preprocessing steps as the initial training (critical for consistency).
# - Model Training: Run src/train.py with the updated data.
# - Model Evaluation: Evaluate the new model against a separate test set and compare its performance to the currently deployed model (the "champion" model).
# - Model Versioning/Registration: Register the newly trained model in the Model Registry (e.g., MLflow Model Registry) with a new version number and its evaluation metrics.
# - Model Approval (Optional): Automatic or manual review step to decide if the new model is better than the champion.
# - Automated Deployment: If the new model is approved, trigger the deployment pipeline (from Phase 3) to replace the old model in production (e.g., using Blue/Green or Canary deployments).

# Define base path
base_dir = os.path.join(os.path.dirname(__file__), 'processed_data')

# File paths
train_path = os.path.join(base_dir, 'train.csv')
test_path = os.path.join(base_dir, 'test.csv')
val_path = os.path.join(base_dir, 'val.csv')

# Read CSV files
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
val_df = pd.read_csv(val_path)

# Define sMAPE function (robust to zero)
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
    return 100 * np.mean(diff)

# Baseline: predict y[t+24] = y[t]
def naive_forecast_24h(df, target_col='y', horizon=24):
    df = df.copy()
    df['y_pred'] = df[target_col].shift(horizon)
    df = df.dropna(subset=['y', 'y_pred'])

    mae = mean_absolute_error(df[target_col], df['y_pred'])
    smape_score = smape(df[target_col], df['y_pred'])

    print(f"24-step MAE: {mae:.4f}")
    print(f"24-step sMAPE: {smape_score:.2f}%")

    return df

# Example usage
forecast_df = naive_forecast_24h(val_df)

















































































































































































































































