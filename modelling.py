# modelling.py
import numpy as np
import pandas as pd
import json
import os
import mlflow
import joblib # Untuk menyimpan model terbaik secara lokal
from datetime import datetime

# --- Import model-model dari folder 'model' ---
from model.ExponentialSmoothing import ExponentialSmoothingModel
from model.Theta import ThetaModel 

# --- FUNGSI UTAMA PREPROCESSING DATA ---
# Ini fungsi yang akan menyiapkan data untuk training/evaluasi model
def load_and_preprocess_data(master_data_path="data/master_electricity_prices.csv", target_col="value"):
    df = pd.read_csv(master_data_path)
    df["ds"] = pd.to_datetime(df["timestamp"]) # Sesuaikan dengan nama kolom timestamp Anda
    df = df.rename(columns={target_col: 'y'}) # Gunakan 'y' untuk statsforecast/ES/Theta
    
    # Pastikan unique_id ada. StatsForecast/ExponentialSmoothing bisa single series dengan 'unique_id'
    if 'unique_id' not in df.columns: 
         df['unique_id'] = 'series_1'
    
    # Pastikan data terurut berdasarkan waktu
    df = df.sort_values(by='ds').reset_index(drop=True)

    # Lakukan split train, val, test dari master data ini
    # Ini harus disesuaikan dengan strategi split time-series Anda
    # Contoh sederhana:
    total_len = len(df)
    forecast_horizon = 24 # Contoh horizon yang akan diprediksi
    val_len = forecast_horizon * 2 # Gunakan 2x horizon sebagai validasi

    # test_df akan menjadi data terakhir yang akan kita gunakan untuk prediksi 'masa depan'
    # val_df akan menjadi data validasi untuk cross-validation
    # train_df adalah sisanya untuk pelatihan
    
    test_df_for_prediction = df.iloc[-forecast_horizon:] # Data paling akhir untuk prediksi
    val_df_for_evaluation = df.iloc[-(forecast_horizon + val_len):-forecast_horizon] # Data untuk validasi
    train_df_for_training = df.iloc[:-(forecast_horizon + val_len)] # Data untuk pelatihan

    # full_training_df adalah data yang akan digunakan untuk melatih model terakhir sebelum prediksi
    # Ini adalah train + val data
    full_training_df = df.iloc[:-forecast_horizon]

    return train_df_for_training, val_df_for_evaluation, test_df_for_prediction, full_training_df, df # Kembalikan juga df lengkap (master data)

# --- FUNGSI UTAMA UNTUK MENYIMPAN OUTPUT (SESUAI DASHBOARD) ---
def save_forecast_to_csv(forecast_df: pd.DataFrame, master_actuals_df: pd.DataFrame, file_path="data/forecasts/latest_forecast.csv"):
    """
    Menyimpan DataFrame hasil prediksi ke file CSV, menggabungkannya dengan data aktual terbaru
    dari master_actuals_df untuk periode yang sudah memiliki aktual.
    forecast_df harus memiliki kolom 'ds' dan 'yhat' (prediksi).
    master_actuals_df harus memiliki kolom 'ds' dan 'y' (aktual).
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Gabungkan prediksi dengan data aktual yang paling mutakhir
    # forecast_df memiliki 'ds', 'yhat'
    # master_actuals_df memiliki 'ds', 'y'
    
    # Rename kolom untuk dashboard
    forecast_df_renamed = forecast_df.rename(columns={'yhat': 'predicted_price', 'ds': 'tanggal_jam'})
    master_actuals_df_renamed = master_actuals_df.rename(columns={'y': 'actual_price', 'ds': 'tanggal_jam'})

    # Pastikan 'tanggal_jam' bertipe datetime untuk penggabungan
    forecast_df_renamed['tanggal_jam'] = pd.to_datetime(forecast_df_renamed['tanggal_jam']).dt.tz_localize(None)
    master_actuals_df_renamed['tanggal_jam'] = pd.to_datetime(master_actuals_df_renamed['tanggal_jam']).dt.tz_localize(None)

    # Gabungkan berdasarkan kolom 'tanggal_jam'
    # Ini akan memastikan untuk setiap 'tanggal_jam' yang ada di prediksi, jika ada aktualnya di master_actuals_df, akan digabungkan
    merged_df = pd.merge(
        forecast_df_renamed,
        master_actuals_df_renamed[['tanggal_jam', 'actual_price']], # Hanya ambil ds dan actual_price
        on='tanggal_jam',
        how='left' # Gunakan left join agar semua prediksi tetap ada
    )
    
    # Pilih dan rename kolom untuk output akhir
    final_output_df = merged_df[['tanggal_jam', 'predicted_price', 'actual_price']]
    final_output_df.to_csv(file_path, index=False)
    print(f"Hasil prediksi (termasuk aktual yang up-to-date) disimpan ke: {file_path}")

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
    master_data_path="data/master_electricity_prices.csv",
    forecast_horizon=24,
    es_season_list=[6, 12, 24], # Contoh daftar periode musiman untuk ES
    theta_season_list=[6, 12, 24] # Contoh daftar periode musiman untuk Theta
):
    print("Memulai MLOps Pipeline...")
    
    # --- MLflow Outer Run for the entire MLOps Pipeline execution ---
    # Ini akan menjadi run utama yang mencakup seluruh proses MLOps
    with mlflow.start_run(run_name="Full_MLOps_Pipeline_Run") as pipeline_run:
        mlflow.log_param("pipeline_start_time", datetime.now().isoformat())
        mlflow.log_param("forecast_horizon", forecast_horizon)
        
        # 1. Load dan Preprocess Data
        print("Memuat dan memproses data...")
        train_df, val_df, test_df_for_prediction, full_training_df, master_df_full = \
            load_and_preprocess_data(master_data_path)
        
        mlflow.log_param("master_data_rows", len(master_df_full))
        mlflow.log_param("train_data_rows", len(train_df))
        mlflow.log_param("val_data_rows", len(val_df))
        mlflow.log_param("test_data_for_prediction_rows", len(test_df_for_prediction))
        
        # 2. Optimasi dan Pemilihan Model (Theta vs. Exponential Smoothing)
        print("Memulai optimasi model Theta...")
        theta_model_instance = ThetaModel(freq='h', forecast_horizon=forecast_horizon) 
        best_theta_mae, best_theta_params, best_theta_model_obj, theta_run_id = theta_model_instance.optimize(
            df=full_training_df, forecast_horizon=forecast_horizon, season_list=theta_season_list
        )
        mlflow.log_metric("overall_theta_best_mae", best_theta_mae)
        mlflow.log_param("theta_optimization_run_id", theta_run_id)


        print("\nMemulai optimasi model Exponential Smoothing...")
        es_model_instance = ExponentialSmoothingModel(
            has_trend=True, seasonal_periods=24, seasonal_type='add', forecast_horizon=forecast_horizon
        ) 
        best_es_mae, best_es_params, best_es_model_obj, es_run_id = es_model_instance.optimize(
            df=full_training_df.rename(columns={'y': 'y', 'ds': 'ds', 'unique_id': 'unique_id'}), # ES expects 'y'
            forecast_horizon=forecast_horizon, season_list=es_season_list
        )
        mlflow.log_metric("overall_es_best_mae", best_es_mae)
        mlflow.log_param("es_optimization_run_id", es_run_id)

        # 3. Pilih Model Terbaik Secara Global
        print("\nMemilih model terbaik secara keseluruhan...")
        overall_best_model_name = ""
        overall_best_mae = float('inf')
        overall_best_model_obj = None
        overall_best_artifact_path = ""
        overall_best_run_id = "" # Run ID dari run MLflow yang melog model terbaik

        if best_theta_model_obj and best_theta_mae < overall_best_mae:
            overall_best_mae = best_theta_mae
            overall_best_model_name = "ThetaModel"
            overall_best_model_obj = best_theta_model_obj
            overall_best_run_id = theta_run_id
            overall_best_artifact_path = "champion_theta_model" # Sesuai artifact_path di Theta.py
            
        if best_es_model_obj and best_es_mae < overall_best_mae:
            overall_best_mae = best_es_mae
            overall_best_model_name = "ExponentialSmoothingModel"
            overall_best_model_obj = best_es_model_obj
            overall_best_run_id = es_run_id
            overall_best_artifact_path = "champion_es_model" # Sesuai artifact_path di ExponentialSmoothing.py
            
        if overall_best_model_obj is None:
            print("Peringatan: Tidak ada model yang berhasil dioptimasi atau tidak ada model terbaik ditemukan.")
            mlflow.log_param("status", "No_Best_Model_Found")
            return

        print(f"Model terbaik secara keseluruhan: {overall_best_model_name} dengan MAE Validasi: {overall_best_mae:.4f}")
        mlflow.log_param("overall_best_model_name", overall_best_model_name)
        mlflow.log_metric("overall_best_validation_mae", overall_best_mae)
        mlflow.log_param("overall_best_model_run_id", overall_best_run_id)

        # 4. Registrasi Model Terbaik ke MLflow Model Registry
        # Kita akan meregistrasikan model yang sudah di-log di nested run
        print(f"Mendaftarkan model '{overall_best_model_name}' ke MLflow Model Registry...")
        
        # URI model di MLflow tracking dari run terbaik
        model_uri_to_register = f"runs:/{overall_best_run_id}/{overall_best_artifact_path}"

        mlflow.register_model(
            model_uri=model_uri_to_register,
            name="ElectricityPriceForecaster", # Nama model di registry Anda
            tags={"project": "MLOps_Finland_Electricity", "source_pipeline_run": pipeline_run.info.run_id},
            description=f"Overall best model ({overall_best_model_name}) from pipeline run {pipeline_run.info.run_id}."
        )
        print(f"Model '{overall_best_model_name}' versi terbaru didaftarkan sebagai 'ElectricityForecaster' di MLflow Model Registry.")

        # 5. Lakukan Forecasting dan Simpan Hasil
        print("Melakukan forecasting dan menyimpan hasil untuk dashboard...")
        
        # Buat DataFrame untuk future dates
        last_train_timestamp = full_training_df['ds'].max()
        future_dates = pd.date_range(
            start=last_train_timestamp + pd.Timedelta(hours=1), 
            periods=forecast_horizon, 
            freq='H' # Sesuaikan frekuensi data Anda (H=Hourly, D=Daily)
        )
        
        future_input_df = pd.DataFrame({'ds': future_dates})
        if 'unique_id' in full_training_df.columns:
            future_input_df['unique_id'] = full_training_df['unique_id'].iloc[0]
        future_input_df['y'] = np.nan # Tambahkan kolom 'y' dummy jika diperlukan oleh predict method

        # Muat model terbaik yang baru saja didaftarkan untuk melakukan forecasting
        # Ini akan memastikan kita menggunakan model dari registry
        # Muat model dengan stage 'Production' jika sudah dipromosikan (biasanya secara manual atau terpisah)
        # Untuk pertama kali deploy, kita bisa muat versi terbaru langsung
        try:
            loaded_forecaster = mlflow.pyfunc.load_model("models:/ElectricityForecaster/Production")
            print("Memuat model 'ElectricityForecaster' dari MLflow Model Registry (Production Stage).")
        except Exception as e:
            print(f"Gagal memuat model Production: {e}. Mencoba memuat versi terbaru yang terdaftar.")
            # Fallback: load versi terbaru jika Production belum ada/gagal
            loaded_forecaster = mlflow.pyfunc.load_model("models:/ElectricityForecaster/latest")
            print("Memuat model 'ElectricityForecaster' dari MLflow Model Registry (Versi Terbaru).")
        
        # Lakukan prediksi
        forecast_result_df = loaded_forecaster.predict(future_input_df) 

        # --- Simpan Hasil Prediksi dan Metrik untuk Dashboard ---
        # save_forecast_to_csv membutuhkan master_df_full sebagai sumber aktual
        save_forecast_to_csv(forecast_result_df, master_df_full, "data/forecasts/latest_forecast.csv")

        # Save metrik untuk dashboard (dari evaluasi test set)
        final_metrics_for_dashboard = {
            "model_name": overall_best_model_name,
            "MAE": overall_best_mae, # MAE dari validasi (cross-validation)
            "forecast_horizon": forecast_horizon,
            "best_params": overall_best_params_found
        }
        save_metrics_to_json(final_metrics_for_dashboard, "artifacts/metrics/model_metrics.json")
        
        mlflow.log_param("pipeline_end_time", datetime.now().isoformat())
        print("\nMLOps Pipeline selesai. Data untuk dashboard telah diperbarui.")

# ====================================================================================================
# BLOK EKSEKUSI UTAMA (untuk menjalankan pipeline)
# ====================================================================================================

if __name__ == "__main__":
    # --- Konfigurasi MLflow Tracking Server ---
    # Jika Anda menggunakan server MLflow terpisah (misalnya di Azure ML Workspace),
    # Anda perlu mengatur MLFLOW_TRACKING_URI dan kredensial.
    # Contoh:
    # os.environ["MLFLOW_TRACKING_URI"] = "azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/YOUR_SUBS_ID/resourceGroups/YOUR_RG/providers/Microsoft.MachineLearningServices/workspaces/YOUR_WORKSPACE_NAME"
    # os.environ["AZURE_CLIENT_ID"] = "..."
    # os.environ["AZURE_TENANT_ID"] = "..."
    # os.environ["AZURE_CLIENT_SECRET"] = "..."
    
    # Untuk local testing tanpa server MLflow eksternal, cukup biarkan default (akan membuat folder 'mlruns')
    # Anda juga bisa mengarahkan ke database lokal untuk persistensi run
    # mlflow.set_tracking_uri("sqlite:///mlruns.db") # Contoh dengan database SQLite

    # Path ke data Anda
    master_data_path = 'data/master_electricity_prices.csv'
    
    # --- Pastikan file dummy ada untuk pengujian lokal jika Anda belum menjalankan ingestion ---
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/forecasts', exist_ok=True)
    os.makedirs('artifacts/metrics', exist_ok=True)
    os.makedirs('artifacts/models', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True) # Untuk pipeline_timings.json

    if not os.path.exists(master_data_path):
        print("Membuat data dummy master_electricity_prices.csv untuk pengujian lokal...")
        pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='H'),
            'value': np.random.rand(200)*100,
            'unique_id': 'series_1'
        }).to_csv(master_data_path, index=False)

    # Note: latest_forecast.csv dan model_metrics.json akan dibuat oleh run_mlops_pipeline itu sendiri
    # Jika ada, mereka akan ditimpa/di-append

    run_mlops_pipeline(
        master_data_path=master_data_path,
        forecast_horizon=24 
    )
