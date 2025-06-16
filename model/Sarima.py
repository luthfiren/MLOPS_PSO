import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

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