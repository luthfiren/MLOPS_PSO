import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor # For a simpler baseline
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, QuantileLoss
from lightning.pytorch import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pytorch_lightning.loggers import TensorBoardLogger

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


# # Define sMAPE function (robust to zero)
# def smape(y_true, y_pred):
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
#     diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
#     return 100 * np.mean(diff)

# # Baseline: predict y[t+24] = y[t]
# def naive_forecast_24h(df, target_col='y', horizon=24):
#     df = df.copy()
#     df['y_pred'] = df[target_col].shift(horizon)
#     df = df.dropna(subset=['y', 'y_pred'])

#     mae = mean_absolute_error(df[target_col], df['y_pred'])
#     smape_score = smape(df[target_col], df['y_pred'])

#     print(f"24-step MAE: {mae:.4f}")
#     print(f"24-step sMAPE: {smape_score:.2f}%")

#     return df

# Several testing or measurement on data
# ===========================
def has_trend(time_series, alpha=0.05):
    """
    Check if a time series has a significant trend using the Mann-Kendall test.
    
    Parameters:
    - time_series: List or 1D array of values.
    - alpha: Significance level (default 0.05).
    
    Returns:
    - (bool) True if trend is detected.
    """
    n = len(time_series)
    x = np.arange(n)
    tau, p_value = kendalltau(x, time_series)
    return p_value < alpha

def auto_detect_seasonality(time_series, max_period=48, alpha=0.05, min_peak_prominence=0.1):
    """
    Efficiently detect candidate seasonality periods in a time series using ACF.
    
    Parameters:
    - time_series: 1D array-like numeric data.
    - max_period: max lag to check for seasonality.
    - alpha: significance level for autocorrelation.
    - min_peak_prominence: minimum relative height to consider an ACF peak significant.
    
    Returns:
    - List of candidate seasonal periods (lags).
    """
    ts = np.asarray(time_series)
    # Detrend by first differencing
    ts_diff = np.diff(ts)
    n = len(ts_diff)
    
    # 95% confidence interval for zero autocorrelation
    conf_interval = 1.96 / np.sqrt(n)
    
    # Compute ACF up to max_period with FFT for speed on large data
    acf_vals = acf(ts_diff, nlags=max_period, fft=True)
    
    # Focus only on lags from 2 to max_period
    lags = np.arange(2, max_period+1)
    acf_subset = acf_vals[2:max_period+1]
    
    # Find local maxima in ACF values (candidate seasonality peaks)
    local_max_idx = argrelextrema(acf_subset, np.greater)[0]
    local_max_lags = lags[local_max_idx]
    local_max_vals = acf_subset[local_max_idx]
    
    # Filter peaks that exceed confidence interval and min_peak_prominence
    candidates = local_max_lags[
        (np.abs(local_max_vals) > conf_interval) & 
        (local_max_vals > min_peak_prominence)
    ]
    
    return candidates.tolist()

# MODEL SARIMA
# =============================
# FUNGSI PREPROCESSING
# =============================
def load_data(file_path):
    """Memuat data dari CSV dan mengatur timestamp sebagai index."""
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def add_time_features(df, target_col):
    """Menambahkan fitur berbasis waktu (jam, hari, musim)."""
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["season"] = df["month"] % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Autumn
    return df

def check_stationarity(series):
    """Menguji apakah data stationer menggunakan ADF Test."""
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1] < 0.05

def apply_differencing(series, d=1, D=0, seasonal_period=24):
    """Melakukan differencing non-musiman dan musiman."""
    if d > 0:
        series = series.diff(d).dropna()
    if D > 0:
        series = series.diff(seasonal_period * D).dropna()
    return series

# =============================
# FUNGSI MODELING
# =============================
def identify_parameters(series, seasonal_period=24, plot=False):
    """Mengidentifikasi parameter SARIMA berdasarkan ACF/PACF."""
    if plot:
        plot_acf(series, lags=40)
        plot_pacf(series, lags=40)
        plt.show()
    
    # Contoh parameter default (sesuaikan dengan analisis)
    return (2,1,1), (1,1,1,seasonal_period)

def train_sarima_model(train_data, order, seasonal_order):
    """Melatih model SARIMA."""
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results

def forecast_sarima(model, steps):
    """Melakukan prediksi menggunakan model SARIMA."""
    forecast = model.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int()
    return pred_mean, pred_ci

# =============================
# FUNGSI EVALUASI
# =============================
def generate_time_series_folds(df, date_col, n_splits, horizon):
    df = df.sort_values(by=date_col)
    fold_size = len(df) // (n_splits + 1)
    folds = []

    for i in range(n_splits):
        train = df.iloc[:fold_size * (i + 1)]
        test = df.iloc[fold_size * (i + 1): fold_size * (i + 1) + horizon]
        folds.append((train, test))

    return folds

def evaluate_forecast(y_true, y_pred):
    """Menghitung metrik evaluasi (MAE, RMSE)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

def plot_forecast(y_train, y_true, y_pred, pred_ci=None):
    """Memvisualisasikan hasil forecasting."""
    plt.figure(figsize=(12,6))
    plt.plot(y_train[-100:], label="Train")
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Forecast", color="red")
    if pred_ci is not None:
        plt.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='k', alpha=.2)
    plt.legend()
    plt.show()

# =============================
# FUNGSI UTAMA (PIPELINE)
# =============================
def run_sarima_pipeline(train_path, val_path, test_path, target_col="value", seasonal_period=24):
    """Pipeline lengkap SARIMA dari data hingga evaluasi."""
    # 1. Load Data
    train_df = load_data(train_path)
    val_df = load_data(val_path)
    test_df = load_data(test_path)
    
    # 2. Tambahkan fitur waktu
    train_df = add_time_features(train_df, target_col)
    val_df = add_time_features(val_df, target_col)
    test_df = add_time_features(test_df, target_col)
    
    # 3. Gabungkan data
    full_df = pd.concat([train_df, val_df, test_df])
    
    # 4. Cek stationeritas
    series = full_df[target_col]
    is_stationary = check_stationarity(series)
    if not is_stationary:
        series = apply_differencing(series, d=1, D=1, seasonal_period=seasonal_period)
    
    # 5. Split kembali ke train/val/test
    train_data = series.loc[train_df.index]
    val_data = series.loc[val_df.index]
    test_data = series.loc[test_df.index]
    
    # 6. Identifikasi parameter
    order, seasonal_order = identify_parameters(series, seasonal_period)
    
    # 7. Train model
    model = train_sarima_model(train_data, order, seasonal_order)
    
    # 8. Validasi
    val_pred, _ = forecast_sarima(model, len(val_data))
    print("Validation Metrics:")
    evaluate_forecast(val_data, val_pred)
    plot_forecast(train_data, val_data, val_pred)
    
    # 9. Retrain dengan full data
    full_data = pd.concat([train_data, val_data])
    final_model = train_sarima_model(full_data, order, seasonal_order)
    
    # 10. Test
    test_pred, test_ci = forecast_sarima(final_model, len(test_data))
    print("Test Metrics:")
    evaluate_forecast(test_data, test_pred)
    plot_forecast(full_data, test_data, test_pred, test_ci)
    
    return final_model
        
# # Model ES (Simple Exponential)
class ExponentialSmoothingSelector:
    def __init__(self, time_series, has_trend=False, seasonal_periods=None, seasonal_type='add'):
        self.time_series = np.asarray(time_series)
        self.has_trend = has_trend
        self.seasonal_periods = seasonal_periods
        self.seasonal_type = seasonal_type

        # Internal storage
        self.models = {}
        self.evaluations = {}
        self.selected_model_name = None
        self.best_model = None
        self.best_forecast = None

    def _train_model(self, model_name, model_cls, **fit_args):
        try:
            model = model_cls(self.train_series, **fit_args)
            fitted = model.fit()
            forecast = fitted.forecast(self.test_size)
            metrics = evaluate_forecast(self.test_series, forecast) 

            self.models[model_name] = fitted
            self.evaluations[model_name] = metrics
        except Exception as e:
            print(f"[WARNING] {model_name} failed: {e}")

    def train_and_select(self, test_size=12):
        self.test_size = test_size
        self.train_series = self.time_series[:-test_size]
        self.test_series = self.time_series[-test_size:]

        if not self.has_trend and not self.seasonal_periods:
            self._train_model("SES", SimpleExpSmoothing)
        elif self.has_trend and not self.seasonal_periods:
            self._train_model("Holt", Holt)
        elif self.seasonal_periods:
            self._train_model(
                "Holt-Winters",
                ExponentialSmoothing,
                trend='add' if self.has_trend else None,
                seasonal=self.seasonal_type,
                seasonal_periods=self.seasonal_periods
            )
        else:
            raise RuntimeError("Invalid combination of trend and seasonality flags.")
        
        if not self.evaluations:
            raise RuntimeError("No models were successfully trained.")

        sorted_models = sorted(
            self.evaluations.items(),
            key=lambda x: (x[1]['MAE'] + x[1]['RMSE']) / 2
        )
        
        self.selected_model_name = sorted_models[0][0]
        self.best_model = self.models[self.selected_model_name]
        self.best_forecast = self.best_model.forecast(test_size)

        return {
            'selected_model': self.selected_model_name,
            'best_forecast': self.best_forecast,
            'MAE': self.evaluations[self.selected_model_name]['MAE'],
            'RMSE': self.evaluations[self.selected_model_name]['RMSE']
        }

    def get_all_evaluations(self):
        return self.evaluations

# # Model Theta


# # MODEL TFT
def add_time_idx(df, time_col="endTime"):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))
    return df

def add_group_column(df):
    df["group"] = "series_1"
    return df

def create_datasets(train_long, val_long, test_long, 
                    time_idx="time_idx", group_col="group", target_col="value",
                    max_encoder_length=24, max_prediction_length=6):
    train_dataset = TimeSeriesDataSet(
        train_long,
        time_idx=time_idx,
        target=target_col,
        group_ids=[group_col],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=[time_idx],
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=[group_col]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_long, predict=False, stop_randomization=True)
    test_dataset_pred = TimeSeriesDataSet.from_dataset(train_dataset, test_long, predict=True, stop_randomization=True)

    return train_dataset, val_dataset, test_dataset_pred


def create_dataloaders(train_dataset, val_dataset, test_dataset_pred, batch_size=64):
    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    test_dataloader_pred = test_dataset_pred.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader_pred


def build_model(train_dataset, dataTrain_loaders, dataVal_loaders):
    """Instantiate TFT model and trainer."""
    
    # assuming you have train_dataloader and val_dataloader
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        log_interval=2,
        reduce_on_plateau_patience=4,
    )
    
    # Set up callbacks and logger
    logger = TensorBoardLogger("lightning_logs")

    trainer = Trainer(
        max_epochs=10,
        gradient_clip_val=0.1,
        logger=logger,
    )

    # use model.fit instead of passing only model
    trainer.fit(
        model,
        train_dataloaders=dataTrain_loaders,
        val_dataloaders=dataVal_loaders
    )

    return model, trainer


def train_model(trainer, model, train_dataloader, val_dataloader):
    """Train the TFT model."""
    trainer.fit(model, train_dataloader, val_dataloader)


def evaluate_model(model, test_dataloader):
    """Evaluate the TFT model and return MAE, RMSE."""    
    # Predict
    output = model.predict(test_dataloader, mode="raw", return_x=True)

    # Unpack what we need
    raw_predictions = output[0]
    x = output[1]

    # Extract true and predicted values
    y_true = x["decoder_target"].cpu().numpy().squeeze().flatten()
    
    # Extract the actual prediction tensor
    y_pred_tensor = raw_predictions.prediction  # This is the tensor with predictions

    # Now convert to numpy
    y_pred = y_pred_tensor.detach().cpu().numpy()[:, :, 0].flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

from model import ThetaModel
from sklearn.model_selection import TimeSeriesSplit
import time

def univariate_configuration(train, val, id_value='series_1'):
    univariate_col = ['endTime', 'value']
    rename_map = {'endTime': 'ds', 'value': 'y'}
    
    def prepare_df(df):
        return df.loc[:, univariate_col].rename(columns=rename_map).assign(unique_id=id_value)
    
    df_train = prepare_df(train)
    df_validation = prepare_df(val)
    
    return df_train, df_validation

# 🧪 Example Usage
if __name__ == "__main__":
    # Define base path
    base_dir = os.path.join(os.path.dirname(__file__), 'processed_data')

    # File paths
    train_path = os.path.join(base_dir, 'train.csv')
    val_path = os.path.join(base_dir, 'val.csv')
    test_path = os.path.join(base_dir, 'test.csv') # hapus kalau sudah integrasi
    prediction_path = os.path.join(base_dir, 'test.csv')

    # Read CSV files
    train_df = pd.read_csv(train_path, sep=',')
    val_df = pd.read_csv(val_path, sep=',')
    test_df = pd.read_csv(test_path, sep=',') # hapus kalau sudah integrasi
    prediction_df = pd.read_csv(test_path, sep=',')
    prediction_df = prediction_df[['endTime', 'value']]
    prediction_df['value'] = 0
    
    # Univariate Configuration
    univariate_train_df, univariate_val_df = univariate_configuration(train_df, val_df)
    print(univariate_train_df.columns, 'val', univariate_val_df.columns)
    
    # Assuming df is your DataFrame with 'ds' and 'y'
    tscv = TimeSeriesSplit(n_splits=5)

    # Split now contains 5 tuples of (train_df, test_df)
    splits = []
    for train_idx, test_idx in tscv.split(univariate_train_df):
        train_split = univariate_train_df.iloc[train_idx]
        test_split = univariate_train_df.iloc[test_idx]
        splits.append((train_split, test_split))
            
    theta_model = ThetaModel()
    theta_mae = theta_model.train_with_fold(folds=splits)
    print(theta_mae)
    
    # Check sesonality and trend
    # trend_status = has_trend(time_series=train_df['value'])
    # sesonality_list = auto_detect_seasonality(time_series=train_df['value'])
    
    # print(trend_status, auto_detect_seasonality)
        
    ######## Ngebuat model wajib yang bisa di fit, predict, dan saved ke joblib. MAKE SURE parameter bisa diganti2 contoh theta bisa diganti jadi forecast_horizon=48
    
    # # ======= TFT =========
    # train_df_tft = add_time_idx(train_df.copy())
    # val_df_tft = add_time_idx(val_df.copy())
    # test_df_tft = add_time_idx(test_df.copy())
    
    # train_df_tft = add_group_column(train_df_tft)
    # val_df_tft = add_group_column(val_df_tft)
    # test_df_tft = add_group_column(test_df_tft)

    # train_ds, val_ds, test_ds_pred = create_datasets(train_df_tft, val_df_tft, test_df_tft)
    # train_dl, val_dl, test_dl_pred = create_dataloaders(train_ds, val_ds, test_ds_pred)

    # # Train
    # model, trainer = build_model(train_ds, train_dl, val_dl)
    # train_model(trainer, model, train_dl, val_dl)

    # # Evaluate on val set (which has targets)
    # val_mae, val_rmse = evaluate_model(model, val_dl)

    # # Predict on the real test set (no targets)
    # predictions = model.predict(test_dl_pred)

    # # ======= ARIMA =========
    # # Jalankan pipeline
    # model = run_sarima_pipeline(train_path, val_path, test_path, target_col="valuex", seasonal_period=24)
    
    # # # Simpan model
    # # model.save("models/sarima_final.pkl")
    
    # ======= Exponential Semoothing =========
    # selector = ExponentialSmoothingSelector(time_series=train_df.copy(), has_trend=trend_status, seasonal_periods=sesonality_list)