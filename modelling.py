import numpy as np
import pandas as pd
import json
import os
import subprocess
import mlflow
import joblib
import tempfile
import argparse
from datetime import datetime

# --- parse action ---
def parse_args():
    parser = argparse.ArgumentParser(description="Run or retrain MLOps pipeline.")
    parser.add_argument("--mode", choices=["run", "retrain"], default="run", help="Pipeline mode: run or retrain")
    parser.add_argument("--model-uri", type=str, default="models:/ElectricityPriceForecaster/Production")
    parser.add_argument("--train-data", type=str, default="data/master_electricity_prices.csv")
    return parser.parse_args()

# --- Dynamic Model Discovery ---
from model import discover_model_classes

# ------- Tambahan untuk pyfunc wrapper -------
import mlflow.pyfunc


class JoblibModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        return joblib.load(context.artifacts["model_path"])
    def predict(self, context, model_input):
        model = self.load_context(context)
        # NOTE: Pastikan signature .predict sesuai model kamu!
        # Untuk statsmodels/forecasting, sesuaikan jika perlu
        if hasattr(model, "predict"):
            # statsforecast/prophet: prediksi biasanya butuh DataFrame, bisa saja perlu tweak di sini
            return model.predict(model_input)
        elif hasattr(model, "forecast"):
            return model.forecast(model_input)
        else:
            raise RuntimeError("Model does not support predict/forecast")

def load_and_preprocess_data(master_data_path="processed_data/merged_data.csv", target_col="value"):
    df = pd.read_csv(master_data_path)
    df["ds"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={target_col: 'y'})
    if 'unique_id' not in df.columns: 
         df['unique_id'] = 'series_1'
    # Pastikan tipe kolom benar untuk model time series
    df["y"] = pd.to_numeric(df["y"], errors='coerce')
    df["unique_id"] = df["unique_id"].astype(str)
    df = df.sort_values(by='ds').reset_index(drop=True)
    total_len = len(df)
    forecast_horizon = 24
    val_len = forecast_horizon * 2
    test_df_for_prediction = df.iloc[-forecast_horizon:]
    val_df_for_evaluation = df.iloc[-(forecast_horizon + val_len):-forecast_horizon]
    train_df_for_training = df.iloc[:-(forecast_horizon + val_len)]
    full_training_df = df.iloc[:-forecast_horizon]
    return train_df_for_training, val_df_for_evaluation, test_df_for_prediction, full_training_df, df

def save_forecast_to_csv(forecast_df: pd.DataFrame, master_actuals_df: pd.DataFrame, file_path="data/forecasts/latest_forecast.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    forecast_df_renamed = forecast_df.rename(columns={'yhat': 'predicted_price', 'ds': 'tanggal_jam'})
    master_actuals_df_renamed = master_actuals_df.rename(columns={'y': 'actual_price', 'ds': 'tanggal_jam'})
    forecast_df_renamed['tanggal_jam'] = pd.to_datetime(forecast_df_renamed['tanggal_jam']).dt.tz_localize(None)
    master_actuals_df_renamed['tanggal_jam'] = pd.to_datetime(master_actuals_df_renamed['tanggal_jam']).dt.tz_localize(None)
    merged_df = pd.merge(
        forecast_df_renamed,
        master_actuals_df_renamed[['tanggal_jam', 'actual_price']],
        on='tanggal_jam',
        how='left'
    )
    final_output_df = merged_df[['tanggal_jam', 'predicted_price', 'actual_price']]
    final_output_df.to_csv(file_path, index=False)
    print(f"Hasil prediksi (termasuk aktual yang up-to-date) disimpan ke: {file_path}")

def save_metrics_to_json(metrics_dict, file_path="artifacts/metrics/model_metrics.json"):
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

def retrain_model(model_uri, train_data_path, target_column='value', experiment_name='RetrainExperiment'):
    print(f"🚀 Starting retrain with model: {model_uri} and data: {train_data_path}")

    if model_uri is None:
        print("⚙️ No existing model URI provided. Running full MLOps pipeline...")
        run_mlops_pipeline()
        return

    # Load model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    model = loaded_model._model_impl.load_context(loaded_model._model_impl._context)

    # ⚠️ PROPER SPLIT: Prepare train/val/test properly FIRST
    train_df, val_df, test_df, _, full_df = load_and_preprocess_data(master_data_path=train_data_path, target_col=target_column)

    # ✅ Train ONLY on train_df
    X_train = train_df.drop(columns=['ds', 'unique_id', 'y'])
    y_train = train_df['y']
    model.fit(X_train, y_train)

    # ✅ Validation
    X_val = val_df.drop(columns=['ds', 'unique_id', 'y'])
    y_val_actual = val_df['y']
    y_val_pred = model.predict(X_val)
    mae_val = np.mean(np.abs(y_val_actual - y_val_pred))

    print(f"📊 MAE on validation set: {mae_val:.4f}")

    # Load the best previous model MAE from metrics
    metrics_file = 'artifacts/metrics/model_metrics.json'
    best_prev_mae = None
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
            if metrics_data:
                metrics_data_sorted = sorted(metrics_data, key=lambda x: x['MAE'])
                best_prev_mae = metrics_data_sorted[0]['MAE']
                print(f"📁 Best previous MAE: {best_prev_mae:.4f}")

    # Compare and decide promotion
    if best_prev_mae is None or mae_val < best_prev_mae:
        print("✅ New retrained model is better. Running full MLOps pipeline to promote...")
        run_mlops_pipeline()
    else:
        print("⚠️ Retrained model is not better. Generating forecast anyway for monitoring...")
        forecast_result_df = model.predict(test_df.drop(columns=['ds', 'unique_id', 'y']))
        forecast_df = pd.DataFrame({'ds': test_df['ds'], 'yhat': forecast_result_df})
        save_forecast_to_csv(forecast_df, full_df, "data/forecasts/latest_forecast.csv")

    # 📦 Always log retrained model + metrics to MLflow (under 'LatestRetrainAttempt')
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"LatestRetrain-{datetime.now().isoformat()}") as run:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.pkl"
            joblib.dump(model, model_path)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=JoblibModelWrapper(),
                artifacts={"model_path": model_path},
            )

            # Log metrics to MLflow
            mlflow.log_metric("MAE", mae_val)

            # Log metadata
            mlflow.set_tag("model_name", "RetrainedModel")
            mlflow.set_tag("retrain_date", datetime.now().isoformat())
            mlflow.set_tag("promotion_decision", "promoted" if best_prev_mae is None or mae_val < best_prev_mae else "rejected")

    print("📁 Latest retrained model and metrics logged to MLflow for audit.")
                                    
def run_mlops_pipeline(
    master_data_path="processed_data/merged_data.csv",
    forecast_horizon=24,
    season_list=[6, 12, 24]
):
    print("Memulai MLOps Pipeline (otomatis model discovery)...")
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

        # 2. Dynamic Model Discovery & Optimasi
        print("Mencari dan mengoptimasi seluruh model di folder /model...")
        all_model_classes = discover_model_classes()
        print(f"Model ditemukan: {list(all_model_classes.keys())}")

        best_models_info = []
        for model_name, ModelCls in all_model_classes.items():
            print(f"\nOptimasi: {model_name}")
            try:
                # Buat instance model dengan parameter default (bisa dikembangkan jadi configurable)
                model_instance = ModelCls(forecast_horizon=forecast_horizon)
                # Pastikan signature optimize konsisten antar model!
                best_mae, best_params, best_model_obj, run_id = model_instance.optimize(
                    df=full_training_df, 
                    forecast_horizon=forecast_horizon,
                    season_list=season_list
                )
                best_models_info.append({
                    "name": model_name,
                    "mae": best_mae,
                    "params": best_params,
                    "model": best_model_obj,
                    "run_id": run_id,
                    "artifact_path": getattr(model_instance, "mlflow_artifact_path", f"champion_{model_name.lower()}"),
                })
                mlflow.log_metric(f"{model_name}_best_mae", best_mae)
                mlflow.log_param(f"{model_name}_run_id", run_id)
            except Exception as e:
                print(f"Model {model_name} gagal dioptimasi: {e}")

        # 3. Pilih Model Terbaik
        overall_best = None
        for info in best_models_info:
            if overall_best is None or (info["mae"] is not None and info["mae"] < overall_best["mae"]):
                overall_best = info

        if overall_best is None:
            print("Peringatan: Tidak ada model yang berhasil dioptimasi atau tidak ada model terbaik ditemukan.")
            mlflow.log_param("status", "No_Best_Model_Found")
            return

        print(f"Model terbaik secara keseluruhan: {overall_best['name']} dengan MAE Validasi: {overall_best['mae']:.4f}")
        mlflow.log_param("overall_best_model_name", overall_best['name'])
        mlflow.log_metric("overall_best_validation_mae", overall_best['mae'])
        mlflow.log_param("overall_best_model_run_id", overall_best['run_id'])

        # 4. Log model ke MLflow sebagai pyfunc, lalu register ke Model Registry
        print(f"Mendaftarkan model '{overall_best['name']}' ke MLflow Model Registry...")

        # ----------- MODIFIKASI DI SINI -----------
        # Log model dengan pyfunc log_model
        model_file_path = str(overall_best['model'].model_file)

        mlflow.pyfunc.log_model(
            artifact_path=overall_best['artifact_path'],
            python_model=JoblibModelWrapper(),
            artifacts={"model_path": model_file_path}
        )

        run_id = mlflow.active_run().info.run_id
        model_uri_to_register = f"runs:/{run_id}/{overall_best['artifact_path']}"
        mlflow.register_model(
            model_uri=model_uri_to_register,
            name="ElectricityPriceForecaster",
            tags={"project": "MLOps_Finland_Electricity", "source_pipeline_run": pipeline_run.info.run_id}
        )
        print(f"Model '{overall_best['name']}' versi terbaru didaftarkan sebagai 'ElectricityPriceForecaster' di MLflow Model Registry.")

        # 5. Forecasting & Simpan Hasil
        print("Melakukan forecasting dan menyimpan hasil untuk dashboard...")
        last_train_timestamp = full_training_df['ds'].max()
        future_dates = pd.date_range(
            start=last_train_timestamp + pd.Timedelta(hours=1), 
            periods=forecast_horizon, 
            freq='H'
        )
        future_input_df = pd.DataFrame({'ds': future_dates})
        if 'unique_id' in full_training_df.columns:
            future_input_df['unique_id'] = full_training_df['unique_id'].iloc[0]
        future_input_df['y'] = np.nan

        try:
            loaded_forecaster = mlflow.pyfunc.load_model("models:/ElectricityPriceForecaster/Production")
            print("Memuat model 'ElectricityPriceForecaster' dari MLflow Model Registry (Production Stage).")
        except Exception as e:
            print(f"Gagal memuat model Production: {e}. Mencoba memuat versi terbaru yang terdaftar.")
            loaded_forecaster = mlflow.pyfunc.load_model("models:/ElectricityPriceForecaster/latest")
            print("Memuat model 'ElectricityPriceForecaster' dari MLflow Model Registry (Versi Terbaru).")

        forecast_result_df = loaded_forecaster.predict(future_input_df)
        save_forecast_to_csv(forecast_result_df, master_df_full, "data/forecasts/latest_forecast.csv")

        final_metrics_for_dashboard = {
            "model_name": overall_best["name"],
            "MAE": overall_best["mae"],
            "forecast_horizon": forecast_horizon,
            "best_params": overall_best["params"]
        }
        save_metrics_to_json(final_metrics_for_dashboard, "artifacts/metrics/model_metrics.json")
        
        mlflow.log_param("pipeline_end_time", datetime.now().isoformat())
        print("\nMLOps Pipeline selesai. Data untuk dashboard telah diperbarui.")

def start_mlflow_server(
    port=5000,
    postgres_user="mlflow_user",
    postgres_password="mlflow_pass",
    postgres_host="localhost",
    postgres_port=5432,
    postgres_db="mlflow_db",
    artifact_root="./mlruns"
):
    backend_uri = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

    cmd = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--backend-store-uri", backend_uri,
        "--default-artifact-root", artifact_root,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    print(f"MLflow server started at http://0.0.0.0:{port} with PID {proc.pid}")
    print(f"Backend URI: {backend_uri}")
    return proc

if __name__ == "__main__":
    args = parse_args()
    proc = start_mlflow_server()
    mlflow.set_tracking_uri("http://localhost:5000")

    os.makedirs('data', exist_ok=True)
    os.makedirs('data/forecasts', exist_ok=True)
    os.makedirs('artifacts/metrics', exist_ok=True)
    os.makedirs('artifacts/models', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    master_data_path = 'processed_data/merged_data.csv'
    
    if not os.path.exists(master_data_path):
        print("Creating dummy master_electricity_prices.csv...")
        pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='H'),
            'value': np.random.rand(200) * 100,
            'unique_id': 'series_1'
        }).to_csv(master_data_path, index=False)

    if args.mode == "retrain":
        retrain_model(model_uri=args.model_uri, train_data_path=args.train_data)
    else:
        run_mlops_pipeline(master_data_path=args.train_data, forecast_horizon=24)