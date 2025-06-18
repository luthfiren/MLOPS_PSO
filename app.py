import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import base64
from io import BytesIO
from flask import Flask, render_template, redirect, url_for, jsonify
from datetime import datetime, timezone
import requests
import time
import importingDataFinGrid
import eskretsec

# --- IMpor modul-modul dari modelling.py yang relevan jika app.py juga melakukan prediksi real-time ---
# CATATAN: Untuk dashboard monitoring, app.py biasanya HANYA MEMBACA hasil dari modelling.py.
# Maka, mengimpor seluruh modul modelling.py dan dependensinya (statsmodels, pytorch_forecasting, dll.)
# mungkin tidak diperlukan di sini jika app.py tidak melakukan inferensi langsung.
# Namun, jika Anda di masa depan ingin menambahkan fitur prediksi real-time, Anda mungkin perlu mengimpornya.
# Untuk saat ini, kita fokus pada visualisasi data yang sudah dihasilkan.
# import joblib # Untuk memuat model jika ingin melakukan inferensi real-time
# from statsmodels.tsa.statespace.sarimax import SARIMAX # Contoh model
# from pytorch_forecasting import TemporalFusionTransformer # Contoh model
# from your_modelling_script import load_data, preprocess_data, forecast_sarima # Jika ingin impor parsial


# ====================================================================================================
# FUNGSI VISUALISASI (SEBELUMNYA DI app.py, DIULANG DI SINI)
# ====================================================================================================    
    

def plot_forecast_only(forecast_file):
    """
    Menghasilkan plot hanya untuk data prediksi dari file CSV.
    """
    try:
        df_forecast = pd.read_csv(forecast_file, parse_dates=['tanggal_jam'])
        df_forecast = df_forecast.sort_values('tanggal_jam')

        if 'predicted_price' not in df_forecast.columns or df_forecast.empty:
            print("No forecast data available in the file.")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_forecast['tanggal_jam'], df_forecast['predicted_price'], label='Harga Prediksi', color='#2ecc71')  # Hijau

        ax.set_xlabel('Tanggal dan Jam', fontsize=12)
        ax.set_ylabel('Harga (€/MWh)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    except FileNotFoundError as e:
        print(f"File forecast tidak ditemukan: {e}")
        return None
    except Exception as e:
        print(f"Terjadi error saat membuat plot forecast: {e}")
        return None

def plot_mae_trend(metrics_file='artifacts/metrics/model_metrics.json'):
    """Menghasilkan plot tren MAE dari file JSON."""
    try:
        if not os.path.exists(metrics_file):
            print(f"File metrik model tidak ditemukan: {metrics_file}")
            return None

        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        if not isinstance(metrics_data, list):
            metrics_data = [metrics_data] # Pastikan selalu list of dicts

        if not metrics_data:
            print("No model metrics data found for MAE trend plot.")
            return None

        df_metrics = pd.DataFrame(metrics_data)
        df_metrics['training_date'] = pd.to_datetime(df_metrics['training_date'])
        df_metrics = df_metrics.sort_values('training_date')
        
        latest_model_name = df_metrics.iloc[-1]['model_name'].replace('Model', '')
        print(latest_model_name)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_metrics['training_date'], df_metrics['MAE'], marker='o', linestyle='-', color='#8e44ad') # Ungu
        ax.set_xlabel('Tanggal Pelatihan', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"Terjadi error saat membuat plot tren MAE: {e}")
        return None

def plot_pipeline_timings(timings_file='artifacts/pipeline_timings.json', daily_deadline_minutes=30):
    """Menghasilkan plot waktu penyelesaian pipeline dari file JSON."""
    try:
        print("CWD:", os.getcwd())
        print("Looking for timings file at:", os.path.abspath(timings_file))
        if not os.path.exists(timings_file):
            print(f"File waktu pipeline tidak ditemukan: {timings_file}")
            return None

        with open(timings_file, 'r') as f:
            content = f.read().strip()
            if not content:
                print("Pipeline timings file is empty.")
                return None
            try:
                timings_data = json.loads(content)
            except Exception as e:
                print("JSON decode error:", e)
                print("Content was:", content)
                return None

        if not timings_data:
            print("No pipeline timing data found.")
            return None

        df_timings = pd.DataFrame(timings_data)
        df_timings['run_date'] = pd.to_datetime(df_timings['run_date'], format='mixed')
        df_timings = df_timings.sort_values('run_date')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df_timings['run_date'], df_timings['duration_minutes'], color='#3498db') # Biru
        ax.axhline(y=daily_deadline_minutes, color='#c0392b', linestyle='--', linewidth=2, label=f'Tenggat Waktu ({daily_deadline_minutes} menit)') # Merah gelap

        ax.set_xlabel('Tanggal Menjalankan Pipeline', fontsize=12)
        ax.set_ylabel('Durasi (Menit)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"Terjadi error saat membuat plot waktu pipeline: {e}")
        return None
        
def trigger_github_action(mode="true"):
    GITHUB_TOKEN = eskretsec.retces['ipa_eky'] # 🔐 Should be stored in environment (.env or CI/CD Secrets)
    if not GITHUB_TOKEN:
        raise ValueError("API_KEY environment variable is not set")
    REPO = "luthfiren/MLOPS_PSO"
    WORKFLOW_ID = "ml-pipeline.yml"  # or file name of your workflow in .github/workflows/
    API_URL = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_ID}/dispatches"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}"
    }

    data = {
        "ref": "production",  # or your deployment branch
        "inputs": {
            "mode": mode
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 204:
        print(f"✅ Workflow '{WORKFLOW_ID}' triggered successfully on branch '{data['ref']}'.")
        return True
    else:
        print(f"❌ Failed to trigger workflow: {response.status_code}")
        print("Response content:\n", response.text)
        # Try to parse as JSON, but only if content-type is JSON and not empty
        if response.text and 'application/json' in response.headers.get('content-type', ''):
            try:
                error_json = response.json()
                print("DEBUG: GitHub error response (parsed as JSON):", error_json)
            except Exception as e:
                import traceback
                print(f"❌ Error during prediction: {e}")
                print(traceback.format_exc())
                return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500        
        response.raise_for_status()
               
# ====================================================================================================
# APLIKASI FLASK
# ====================================================================================================

app = Flask(__name__)

@app.route('/')
def dashboard():
    forecast_file = 'data/forecasts/latest_forecast.csv'
    metrics_file = 'artifacts/metrics/model_metrics.json'
    timings_file = 'artifacts/pipeline_timings.json' 

    # --- Panggil fungsi visualisasi dan dapatkan gambar Base64 ---
    plot_forecast_only_img = plot_forecast_only(forecast_file)
    plot_mae_trend_img = plot_mae_trend(metrics_file)
    plot_pipeline_timings_img = plot_pipeline_timings(timings_file)
    with open(metrics_file, 'r') as f:
        data = json.load(f)
        latest_model_name = data[-1]['model_name'].replace('Model', '')  # Get the latest (last) entry
        print(latest_model_name)
    
    # --- Render template HTML dengan data gambar ---
    return render_template(
        'index.html',
        plot_forecast_only=plot_forecast_only_img,
        plot_mae_trend=plot_mae_trend_img,
        plot_pipeline_timings=plot_pipeline_timings_img,
        model_name=latest_model_name
    )

@app.route('/predict', methods=['POST'])
def predict():
    """
    Triggers the full prediction pipeline via GitHub Actions (batch prediction).
    Waits for fresh forecast file, returns success/failure message.
    """
    try:
        last_ingestion_time = importingDataFinGrid.load_last_success_time()
        now = datetime.now(timezone.utc)
        if (now - last_ingestion_time).total_seconds() >= 3600 :
            return jsonify({"message": "Prediction already recent. Skipping."}), 200

        print("✅ Triggering GitHub Action...")
        start_trigger_time = datetime.now(timezone.utc)
        trigger_github_action()

        forecast_path = "data/forecasts/latest_forecast.csv"
        timeout = 1500  # seconds
        polling_interval = 60  # seconds
        elapsed = 0

        while elapsed < timeout:
            now = datetime.now(timezone.utc)
            if os.path.exists(forecast_path):
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(forecast_path), tz=timezone.utc)
                print(f"[Polling] Found file: {forecast_path} | Modified: {file_mod_time} | Trigger Time: {start_trigger_time}")

                if file_mod_time > start_trigger_time:
                    print("✅ New forecast file detected.")
                    return jsonify({"message": "Prediction completed successfully"}), 200

            time.sleep(polling_interval)
            elapsed += polling_interval

        raise TimeoutError("Timeout waiting for new forecast file.")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    
# ====================================================================================================
# BLOK EKSEKUSI UTAMA (untuk menjalankan aplikasi Flask)
# ====================================================================================================

if __name__ == '__main__':
    # Buat direktori yang diperlukan jika belum ada
    os.makedirs('data/forecasts', exist_ok=True)
    os.makedirs('artifacts/metrics', exist_ok=True)
    os.makedirs('artifacts/models', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True) # Untuk pipeline_timings.json

    # Buat file dummy jika belum ada
    if not os.path.exists('data/forecasts/latest_forecast.csv'):
        print("Membuat data dummy latest_forecast.csv untuk pengujian lokal...")
        dummy_data = pd.DataFrame({
            'tanggal_jam': pd.date_range(start='2025-06-12', periods=24, freq='H'),
            'predicted_price': np.random.rand(24) * 100 + 50,
            'actual_price': np.random.rand(24) * 100 + 50 # Untuk simulasi aktual
        })
        dummy_data.to_csv('data/forecasts/latest_forecast.csv', index=False)
    
    if not os.path.exists('data/historical_actuals.csv'):
        print("Membuat data dummy historical_actuals.csv untuk pengujian lokal...")
        dummy_actuals = pd.DataFrame({
            'tanggal_jam': pd.date_range(start='2025-06-12', periods=48, freq='H'),
            'actual_price': np.random.rand(48) * 100 + 50
        })
        dummy_actuals.to_csv('data/historical_actuals.csv', index=False)

    if not os.path.exists('artifacts/metrics/sarima_metrics.json'):
        print("Membuat data dummy sarima_metrics.json untuk pengujian lokal...")
        dummy_metrics = [
            {"model_name": "SARIMA", "MAE": 5.0, "RMSE": 7.0, "training_date": "2025-05-20T10:00:00"},
            {"model_name": "SARIMA", "MAE": 4.8, "RMSE": 6.8, "training_date": "2025-05-25T14:30:00"},
            {"model_name": "SARIMA", "MAE": 4.5, "RMSE": 6.5, "training_date": "2025-06-01T08:00:00"}
        ]
        with open('artifacts/metrics/sarima_metrics.json', 'w') as f:
            json.dump(dummy_metrics, f, indent=4)
            
    if not os.path.exists('artifacts/pipeline_timings.json'):
        print("Membuat data dummy pipeline_timings.json untuk pengujian lokal...")
        dummy_timings = [
            {"run_date": "2025-06-01T10:05:00", "duration_minutes": 25},
            {"run_date": "2025-06-05T14:35:00", "duration_minutes": 28},
            {"run_date": "2025-06-10T08:10:00", "duration_minutes": 32}
        ]
        with open('artifacts/pipeline_timings.json', 'w') as f:
            json.dump(dummy_timings, f, indent=4)

    # --- Jalankan Aplikasi Flask ---
    # Gunakan '0.0.0.0' agar dapat diakses dari luar localhost, penting untuk Docker/Deployment.
    # Gunakan PORT dari environment variable jika tersedia (digunakan oleh Azure App Service).
    # debug=True harus DIMATIKAN di produksi.
    print("Menjalankan aplikasi Flask dashboard...")
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8000))