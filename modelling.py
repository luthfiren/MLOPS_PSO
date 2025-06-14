# modelling.py
import numpy as np
import pandas as pd
import json
import os
import mlflow
from datetime import datetime

# --- Import model-model dari folder 'model' ---
from model.ExponentialSmoothing import ExponentialSmoothingModel
from model.Theta import ThetaModel # Pastikan Theta.py memiliki kelas ThetaModel
from model.Sarima import SarimaModel

# --- FUNGSI UTAMA PREPROCESSING DATA ---
# Ini fungsi yang akan menyiapkan data untuk training/evaluasi model
def load_and_preprocess_data(train_path, val_path, test_path, target_col="value"):
    # Memuat data dari CSV dan mengatur timestamp sebagai index
    def load_df(file_path):
        df = pd.read_csv(file_path)
        df["ds"] = pd.to_datetime(df["timestamp"]) # Gunakan 'ds' untuk statsforecast
        df = df.rename(columns={'value': 'y'}) # Gunakan 'y' untuk statsforecast
        if 'unique_id' not in df.columns: # StatsForecast bisa single series dengan 'unique_id'
             df['unique_id'] = 'series_1'
        df = df[['unique_id', 'ds', 'y']].set_index(['unique_id', 'ds']) # Set MultiIndex
        return df.reset_index() # Reset index agar 'ds' dan 'unique_id' jadi kolom biasa lagi

    train_df = load_df(train_path)
    val_df = load_df(val_path)
    test_df = load_df(test_path) # Data ini mungkin tidak memiliki 'y' jika ini data masa depan

    # Gabungkan semua data training/validasi untuk optimasi model
    full_training_df = pd.concat([train_df, val_df], ignore_index=True)
    full_training_df['ds'] = pd.to_datetime(full_training_df['ds']) # Pastikan datetime

    return train_df, val_df, test_df, full_training_df

# --- FUNGSI UTAMA UNTUK MENYIMPAN OUTPUT (SESUAI DASHBOARD) ---
def save_forecast_to_csv(forecast_df, actual_df_for_comparison, file_path="data/forecasts/latest_forecast.csv"):
    """
    Menyimpan DataFrame hasil prediksi ke file CSV, termasuk data aktual untuk perbandingan.
    forecast_df harus memiliki kolom 'ds' dan 'yhat' (prediksi).
    actual_df_for_comparison harus memiliki kolom 'ds' dan 'y' (aktual).
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Rename 'y' to 'actual_price' and 'yhat' to 'predicted_price' for dashboard
    merged_df = pd.merge(
        forecast_df.rename(columns={'yhat': 'predicted_price'}),
        actual_df_for_comparison.rename(columns={'y': 'actual_price'}),
        on='ds',
        how='left'
    )
    merged_df = merged_df[['ds', 'predicted_price', 'actual_price']].rename(columns={'ds': 'tanggal_jam'})
    merged_df.to_csv(file_path, index=False)
    print(f"Hasil prediksi dan aktual disimpan ke: {file_path}")

def save_metrics_to_json(metrics_dict, file_path="artifacts/metrics/model_metrics.json"):
    """
    Menyimpan metrik model ke file JSON. Ini akan meng-append ke file yang sudah ada.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    metrics_dict['training_date'] = datetime.now().isoformat()

    all_metrics = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                all_metrics = json.load(f)
            if not isinstance(all_metrics, list):
                all_metrics = [all_metrics]
        except json.JSONDecodeError:
            print(f"Warning: Existing {file_path} is corrupted. Starting new metrics list.")
            all_metrics = []

    all_metrics.append(metrics_dict)
    
    with open(file_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrik model disimpan ke: {file_path}")

# --- FUNGSI UTAMA ORKESTRASI (PIPELAINE UTAMA YANG DIJALANKAN GH ACTIONS) ---
def run_mlops_pipeline(
    train_path="processed_data/train.csv", 
    val_path="processed_data/val.csv", 
    test_path="processed_data/test.csv", # Data untuk prediksi masa depan (bisa tanpa kolom 'y')
    actuals_for_dashboard_path="data/historical_actuals.csv", # Data aktual historis untuk perbandingan dashboard
    forecast_horizon=24,
    es_season_list=[6, 12, 24], # Contoh daftar periode musiman untuk ES
    theta_season_list=[6, 12, 24] # Contoh daftar periode musiman untuk Theta
):
    print("Memulai MLOps Pipeline...")
    
    # --- MLflow Outer Run for the entire MLOps Pipeline execution ---
    with mlflow.start_run(run_name="Full_MLOps_Pipeline_Run"):
        mlflow.log_param("pipeline_start_time", datetime.now().isoformat())
        mlflow.log_param("forecast_horizon", forecast_horizon)
        
        # 1. Load dan Preprocess Data
        print("Memuat dan memproses data...")
        train_df, val_df, test_df, full_training_df = load_and_preprocess_data(train_path, val_path, test_path)
        
        mlflow.log_param("train_data_rows", len(train_df))
        mlflow.log_param("val_data_rows", len(val_df))
        mlflow.log_param("test_data_rows", len(test_df))
        
        # 2. Optimasi dan Pemilihan Model (Theta vs. Exponential Smoothing)
        print("Memulai optimasi model Theta...")
        theta_model_instance = ThetaModel(freq='h', forecast_horizon=forecast_horizon) # Asumsi data hourly
        best_theta_mae, best_theta_params, best_theta_model_obj = theta_model_instance.optimize(
            df=full_training_df, forecast_horizon=forecast_horizon, season_list=theta_season_list
        )
        mlflow.log_metric("overall_theta_best_mae", best_theta_mae)

        print("\nMemulai optimasi model Exponential Smoothing...")
        # Asumsikan kita mendeteksi trend/seasonality sebelumnya atau mengujinya di sini
        # Untuk contoh ini, kita set has_trend=True dan seasonal_periods=24
        es_model_instance = ExponentialSmoothingModel(
            has_trend=True, seasonal_periods=24, seasonal_type='add', forecast_horizon=forecast_horizon
        ) 
        best_es_mae, best_es_params, best_es_model_obj = es_model_instance.optimize(
            df=full_training_df.rename(columns={'y': 'y', 'ds': 'ds', 'unique_id': 'unique_id'}), # ES expects 'y'
            forecast_horizon=forecast_horizon, season_list=es_season_list
        )
        mlflow.log_metric("overall_es_best_mae", best_es_mae)

        # 3. Pilih Model Terbaik Secara Global
        print("\nMemilih model terbaik secara keseluruhan...")
        overall_best_model_name = ""
        overall_best_mae = float('inf')
        overall_best_model_obj = None
        overall_best_run_id = None
        mlflow_artifact_path = ""

        if best_theta_mae < overall_best_mae:
            overall_best_mae = best_theta_mae
            overall_best_model_name = "ThetaModel"
            overall_best_model_obj = best_theta_model_obj
            # Dapatkan run_id dari run yang melog model terbaik Theta
            # Ini memerlukan modifikasi di Theta.py untuk mengembalikan run_id
            # Untuk sementara, asumsikan kita punya cara untuk mendapatkannya dari MLflow Tracking UI.
            mlflow_artifact_path = "champion_theta_model" # Sesuai artifact_path di Theta.py
            
        if best_es_mae < overall_best_mae:
            overall_best_mae = best_es_mae
            overall_best_model_name = "ExponentialSmoothingModel"
            overall_best_model_obj = best_es_model_obj
            mlflow_artifact_path = "champion_es_model" # Sesuai artifact_path di ExponentialSmoothing.py
            
        if overall_best_model_obj is None:
            print("Peringatan: Tidak ada model yang berhasil dioptimasi atau tidak ada model terbaik ditemukan.")
            # Mungkin fallback ke model default atau keluar dari pipeline
            mlflow.log_param("status", "No_Best_Model_Found")
            return

        print(f"Model terbaik secara keseluruhan: {overall_best_model_name} dengan MAE Validasi: {overall_best_mae:.4f}")
        mlflow.log_param("overall_best_model_name", overall_best_model_name)
        mlflow.log_metric("overall_best_validation_mae", overall_best_mae)

        # 4. Registrasi Model Terbaik ke MLflow Model Registry
        print(f"Mendaftarkan model '{overall_best_model_name}' ke MLflow Model Registry...")
        
        # Karena kita melog model di dalam metode optimize(), 
        # kita perlu memastikan model tersebut terdaftar dari run yang benar.
        # Cara termudah adalah dengan melognya lagi di sini dan mereferensikan run_id asalnya.
        
        # Untuk contoh ini, kita akan mereferensikan artefak model dari run terbaik
        # dan mendaftarkannya ke registry.
        # Anda perlu mendapatkan run_id dari run MLflow yang menghasilkan model terbaik
        # Misalnya, Anda bisa menyimpannya sebagai bagian dari return best_model_obj
        # Atau, setelah optimize(), query MLflow Tracking untuk menemukan run_id dengan best_validation_mae tertinggi.
        
        # Dummy run_id untuk demonstrasi:
        # best_model_run_id = "your_best_model_mlflow_run_id" 
        
        # Paling baik: log model ini lagi di overall run ini, lalu register
        # Karena Theta dan ES pakai joblib/StatsForecast/statsmodels, 
        # kita perlu wrapper pyfunc jika tidak ada native flavor
        
        # --- Wrapper sederhana untuk model yang dimuat joblib ---
        class GenericPyfuncModel(mlflow.pyfunc.PythonModel):
            def __init__(self, model_path):
                self.model_path = model_path
                self.model = None

            def load_context(self, context):
                # Memuat model asli dari file joblib
                self.model = joblib.load(self.model_path)

            def predict(self, context, model_input: pd.DataFrame):
                # Sesuaikan input DataFrame dengan format yang diharapkan model Anda
                # Misalnya, untuk Theta/ES, butuh kolom 'ds', 'unique_id'
                # Pastikan 'ds' bertipe datetime dan naive
                model_input['ds'] = pd.to_datetime(model_input['ds']).dt.tz_localize(None)

                # Panggil metode predict dari model yang dimuat
                if hasattr(self.model, 'predict'):
                    if overall_best_model_name == "ThetaModel":
                        # predict() dari StatsForecast butuh input df, kasih df future dates
                        forecast_output = self.model.predict(model_input)
                        # hasil forecast_output StatsForecast punya kolom 'ds', 'unique_id', 'yhat'
                        return forecast_output.rename(columns={'yhat': 'predicted_value'})
                    elif overall_best_model_name == "ExponentialSmoothingModel":
                        # predict() dari ExponentialSmoothingModel butuh df, tapi isinya future dates
                        # Hasilnya harus di-DataFrame-kan dan diberi index yang benar
                        forecast_values = self.model.predict(len(model_input)) # Atau self.model.predict(model_input)
                        # Perlu buat df dengan index yang sesuai
                        forecast_df = pd.DataFrame({'ds': model_input['ds'], 'predicted_value': forecast_values})
                        return forecast_df
                    else:
                        raise ValueError("Model predict method not implemented for selected model type.")
                else:
                    raise AttributeError("Loaded model does not have a 'predict' method.")
        
        # Simpan model terbaik secara lokal sebelum meregistrasi sebagai pyfunc
        overall_best_champion_path = os.path.join("artifacts/models", f"{overall_best_model_name.lower()}_champion.joblib")
        # Asumsikan best_model_obj.model adalah objek model yang sudah fit (e.g., StatsForecast object atau ExponentialSmoothing fitted object)
        joblib.dump(overall_best_model_obj.model, overall_best_champion_path)

        # Log dan Registrasi model menggunakan Pyfunc
        with mlflow.start_run(run_name="Register Overall Champion", nested=True):
            mlflow.pyfunc.log_model(
                artifact_path="overall_best_forecaster",
                python_model=GenericPyfuncModel(overall_best_champion_path), # Wrapper kita
                registered_model_name="ElectricityForecaster" # Nama di MLflow Model Registry
            )
            print(f"Model '{overall_best_model_name}' didaftarkan sebagai 'ElectricityForecaster' di MLflow Model Registry.")

        # 5. Lakukan Forecasting dan Simpan Hasil
        print("Melakukan forecasting dan menyimpan hasil untuk dashboard...")
        
        # Pastikan test_df adalah data untuk periode masa depan yang ingin diprediksi
        # Jika test_df tidak punya kolom 'y', ini adalah prediksi murni masa depan
        
        # Buat DataFrame untuk future dates
        last_train_timestamp = full_training_df['ds'].max()
        future_dates = pd.date_range(
            start=last_train_timestamp + pd.Timedelta(hours=1), # Mulai dari 1 jam setelah data training terakhir
            periods=forecast_horizon, 
            freq='H' # Sesuaikan frekuensi data Anda (H=Hourly, D=Daily)
        )
        
        # Buat DataFrame input untuk prediksi
        future_input_df = pd.DataFrame({'ds': future_dates})
        if 'unique_id' in full_training_df.columns:
            future_input_df['unique_id'] = full_training_df['unique_id'].iloc[0] # Asumsi single series
        future_input_df['y'] = np.nan # Tambahkan kolom 'y' dummy jika diperlukan oleh predict method

        # Muat model terbaik yang baru saja didaftarkan untuk melakukan forecasting
        # Ini akan memastikan kita menggunakan model dari registry
        loaded_forecaster = mlflow.pyfunc.load_model("models:/ElectricityForecaster/Production")
        
        # Lakukan prediksi
        forecast_result_df = loaded_forecaster.predict(future_input_df) # Ini akan mengembalikan DataFrame dengan predicted_value

        # --- Simpan Hasil Prediksi dan Metrik untuk Dashboard ---
        # Data aktual historis untuk perbandingan di dashboard (historical_actuals.csv)
        # Ini adalah data aktual dari masa lalu yang akan digunakan dashboard
        actual_data_for_dashboard = pd.read_csv(actuals_for_dashboard_path, parse_dates=['ds'])
        actual_data_for_dashboard = actual_data_for_dashboard.rename(columns={'value': 'y', 'timestamp': 'ds'}) # Sesuaikan kolom

        save_forecast_to_csv(forecast_result_df, actual_data_for_dashboard, "data/forecasts/latest_forecast.csv")

        # Save metrik untuk dashboard (dari evaluasi test set)
        final_metrics = {
            "model_name": overall_best_model_name,
            "MAE": overall_best_mae, # MAE dari validasi (cross-validation)
            "forecast_horizon": forecast_horizon,
            "best_params": overall_best_params_found # Simpan juga parameter terbaiknya
        }
        save_metrics_to_json(final_metrics, "artifacts/metrics/model_metrics.json")
        
        mlflow.log_metric("pipeline_end_time", datetime.now().isoformat())
        print("\nMLOps Pipeline selesai. Data untuk dashboard telah diperbarui.")

# ====================================================================================================
# BLOK EKSEKUSI UTAMA
# ====================================================================================================

if __name__ == "__main__":
    # Konfigurasi MLflow Tracking Server (jika tidak lokal)
    # Ini harus diset sebelum run MLflow pertama
    # Jika Anda menjalankan server MLflow terpisah (misalnya di Azure), ubah URI ini
    # Contoh: os.environ["MLFLOW_TRACKING_URI"] = "https://your-mlflow-server.azureml.net/mlflow/v1.0/subscriptions/..."
    # Atau jika Anda menjalankan secara lokal dengan database:
    # os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db" 
    
    # Untuk local testing tanpa server MLflow eksternal, cukup biarkan default (akan membuat folder 'mlruns')
    # mlflow.set_tracking_uri("http://127.0.0.1:5000") # Jika Anda menjalankan 'mlflow ui' terpisah

    # Path ke data Anda
    train_data_path = 'processed_data/train.csv'
    val_data_path = 'processed_data/val.csv'
    test_data_path = 'processed_data/test.csv' # Ini adalah data yang akan Anda prediksi (future data)
    historical_actuals_for_dashboard = 'data/historical_actuals.csv' # Untuk perbandingan di dashboard

    # Pastikan file dummy ada untuk pengujian lokal jika Anda belum menjalankan ingestion
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(train_data_path):
        pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'), 'value': np.random.rand(100)*100}).to_csv(train_data_path, index=False)
    if not os.path.exists(val_data_path):
        pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'), 'value': np.random.rand(24)*100}).to_csv(val_data_path, index=False)
    if not os.path.exists(test_data_path): # Ini adalah data masa depan yang akan diprediksi
        pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'), 'value': np.random.rand(24)*100}).to_csv(test_data_path, index=False)
    if not os.path.exists(historical_actuals_for_dashboard): # Data aktual untuk visualisasi dashboard
        pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=200, freq='H'), 'y': np.random.rand(200)*100}).to_csv(historical_actuals_for_dashboard, index=False)

    run_mlops_pipeline(
        train_path=train_data_path,
        val_path=val_data_path,
        test_path=test_data_path,
        actuals_for_dashboard_path=historical_actuals_for_dashboard,
        forecast_horizon=24 # Contoh: prediksi 24 jam ke depan
    )