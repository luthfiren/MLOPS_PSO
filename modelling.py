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
    parser.add_argument("--model-uri", type=str, default="models:/ElectricityForecaster/latest", help="MLflow Model URI (default: latest)")
    parser.add_argument("--train-data", type=str, default="processed_data/merged_data.csv")
    return parser.parse_args()

# --- Dynamic Model Discovery ---
from model import discover_model_classes

import mlflow.pyfunc

class JoblibModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        return joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        import inspect
        model = self.load_context(context)
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        if hasattr(model, "predict"):
            sig = inspect.signature(model.predict)
            params = sig.parameters
            if "h" in params and "X" in params:
                h = len(model_input)
                X = model_input.drop(columns=["ds", "unique_id", "y"], errors="ignore") if not model_input.empty else None
                return model.predict(h=h, X=X)
            elif "pred_df" in params:
                return model.predict(pred_df=model_input)
            elif "h" in params:
                h = len(model_input)
                return model.predict(h=h)
            elif "X" in params:
                return model.predict(X=model_input)
            else:
                return model.predict(model_input)
        elif hasattr(model, "forecast"):
            return model.forecast(model_input)
        else:
            raise RuntimeError("Model does not support predict/forecast")   
            
def load_and_preprocess_data(master_data_path="processed_data/merged_data.csv", target_col="value"):
    if not os.path.exists(master_data_path):
        print(f"Warning: '{master_data_path}' not found. Creating dummy data.")
        os.makedirs(os.path.dirname(master_data_path), exist_ok=True)
        pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="H"),
            "value": np.random.rand(200) * 100,
            "unique_id": "series_1"
        }).to_csv(master_data_path, index=False)
    df = pd.read_csv(master_data_path)
    df["ds"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={target_col: 'y'})
    if 'unique_id' not in df.columns:
        df['unique_id'] = 'series_1'
    df["y"] = pd.to_numeric(df["y"], errors='coerce')
    df["unique_id"] = df["unique_id"].astype(str)
    df = df.sort_values(by='ds').reset_index(drop=True)
    forecast_horizon = 24
    val_len = forecast_horizon * 2
    test_df_for_prediction = df.iloc[-forecast_horizon:]
    val_df_for_evaluation = df.iloc[-(forecast_horizon + val_len):-forecast_horizon]
    train_df_for_training = df.iloc[:-(forecast_horizon + val_len)]
    full_training_df = df.iloc[:-forecast_horizon]
    return train_df_for_training, val_df_for_evaluation, test_df_for_prediction, full_training_df, df

def save_forecast_to_csv(forecast_df: pd.DataFrame, master_actuals_df: pd.DataFrame, file_path="data/forecasts/latest_forecast.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if forecast_df.empty:
        raise ValueError("Forecast dataframe is empty after prediction.")
    # Support statsforecast: rename 'Theta' to 'yhat' if necessary
    if 'yhat' not in forecast_df.columns and 'Theta' in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={'Theta': 'yhat'})
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
    for col in ['tanggal_jam', 'predicted_price', 'actual_price']:
        if col not in merged_df.columns:
            raise KeyError(f"Column '{col}' not in merged_df.columns: {merged_df.columns}")
    final_output_df = merged_df[['tanggal_jam', 'predicted_price', 'actual_price']]
    final_output_df.to_csv(file_path, index=False)
    print(f"Hasil prediksi (termasuk aktual yang up-to-date) disimpan ke: {file_path}")

def save_metrics_to_json(metrics_dict, file_path="artifacts/metrics/model_metrics.json"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    metrics_dict = metrics_dict.copy()
    metrics_dict['training_date'] = datetime.now().isoformat()
    all_metrics = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                all_metrics = json.load(f)
                if not isinstance(all_metrics, list):
                    all_metrics = [all_metrics]
        except (json.JSONDecodeError, IOError):
            print(f"[WARNING] Existing file {file_path} is invalid. Starting new metrics list.")
            all_metrics = []
    all_metrics.append(metrics_dict)
    with open(file_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"[INFO] Model metrics saved to: {file_path}")

def save_pipeline_timing(timing_info, file_path="artifacts/pipeline_timings.json"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    entry = {**timing_info}
    all_timings = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                all_timings = json.load(f)
            if not isinstance(all_timings, list):
                all_timings = [all_timings]
        except json.JSONDecodeError:
            print(f"Warning: Existing {file_path} is corrupted. Starting new timing list.")
            all_timings = []
    all_timings.append(entry)
    with open(file_path, "w") as f:
        json.dump(all_timings, f, indent=4)
    print(f"Pipeline timing saved to: {file_path}")
                                    
def run_mlops_pipeline(
    master_data_path="processed_data/merged_data.csv",
    forecast_horizon=24,
    season_list=[6, 12, 24]
):
    start = datetime.now()
    print("Memulai MLOps Pipeline (otomatis model discovery)...")
    mlflow.set_tracking_uri("http://localhost:5001")

    with mlflow.start_run(run_name="Full_MLOps_Pipeline_Run") as pipeline_run:
        mlflow.log_param("pipeline_start_time", datetime.now().isoformat())
        mlflow.log_param("forecast_horizon", forecast_horizon)
        train_df, val_df, test_df_for_prediction, full_training_df, master_df_full = \
            load_and_preprocess_data(master_data_path)
        mlflow.log_param("master_data_rows", len(master_df_full))
        mlflow.log_param("train_data_rows", len(train_df))
        mlflow.log_param("val_data_rows", len(val_df))
        mlflow.log_param("test_data_for_prediction_rows", len(test_df_for_prediction))
        all_model_classes = discover_model_classes()
        print(f"Model ditemukan: {list(all_model_classes.keys())}")
        best_models_info = []
        for model_name, ModelCls in all_model_classes.items():
            print(f"\nOptimasi: {model_name}")
            try:
                model_instance = ModelCls(forecast_horizon=forecast_horizon)
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

        print(f"Mendaftarkan model '{overall_best['name']}' ke MLflow Model Registry...")

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
            name="ElectricityForecaster",
            tags={"project": "MLOps_Finland_Electricity", "source_pipeline_run": pipeline_run.info.run_id}
        )
        print(f"Model '{overall_best['name']}' versi terbaru didaftarkan sebagai 'ElectricityForecaster' di MLflow Model Registry.")

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

        # ================================
        # ALWAYS LOAD LATEST MODEL (not Production)
        # ================================
        try:
            loaded_forecaster = mlflow.pyfunc.load_model(model_uri="models:/ElectricityForecaster/latest")
            print("Memuat model 'ElectricityForecaster' dari MLflow Model Registry (latest version).")
        except Exception as e:
            print(f"Error loading model from registry: {e}")
            raise e

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
        
        end = datetime.now()
        duration_minutes = int((end - start).total_seconds() // 60)
        
        timing_info = {
            "run_date": start.isoformat(),
            "duration_minutes": duration_minutes,
        }
        save_pipeline_timing(timing_info=timing_info)

def retrain_best_model():
    models_uri = "artifacts/models/best_model.pkl"
    train_data_path = "processed_data/merged_data.csv"
    experiment_name = "RetrainExperiment"

    print(f"🚀 Starting retrain using model: {models_uri} and data: {train_data_path}")
    mlflow.set_tracking_uri("http://localhost:5001")

    if not os.path.exists(models_uri):
        raise FileNotFoundError(f"❌ Model file not found at: {models_uri}")

    # Load model from local pickle file
    model = joblib.load(models_uri)

    # Load and preprocess data
    train_df, val_df, test_df, _, full_df = load_and_preprocess_data(master_data_path=train_data_path)

    # Detect if using statsforecast-like model or sklearn
    is_statsforecast = hasattr(model, "fit") and hasattr(model, "predict") and hasattr(model, "models")

    if is_statsforecast:
        model.fit(train_df[["unique_id", "ds", "y"]])
        forecast_val = model.predict(h=len(val_df)).rename(columns={'Theta': 'yhat'})
        val_df['ds'] = pd.to_datetime(val_df['ds'])
        forecast_val['ds'] = pd.to_datetime(forecast_val['ds'])
        actual = pd.merge(
            val_df[['ds', 'unique_id', 'y']],
            forecast_val[['ds', 'unique_id', 'yhat']],
            on=['ds', 'unique_id'],
            how='inner'
        )
        mae_val = np.mean(np.abs(actual['y'] - actual['yhat']))
    else:
        X_train = train_df.drop(columns=['ds', 'unique_id', 'y'])
        y_train = train_df['y']
        model.fit(X_train, y_train)

        X_val = val_df.drop(columns=['ds', 'unique_id', 'y'])
        y_val_actual = val_df['y']
        y_val_pred = model.predict(X_val)
        mae_val = np.mean(np.abs(y_val_actual - y_val_pred))

    print(f"📊 MAE on validation set: {mae_val:.4f}")

    # Load previous best MAE from metrics file
    metrics_file = 'artifacts/metrics/model_metrics.json'
    best_prev_mae = None
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
            if metrics_data:
                best_prev_mae = sorted(metrics_data, key=lambda x: x['MAE'])[0]['MAE']
                print(f"📁 Best previous MAE: {best_prev_mae:.4f}")

    # Decision: Promote or trigger full pipeline
    if best_prev_mae is None or mae_val < best_prev_mae:
        print("✅ New retrained model is better. Running full MLOps pipeline to promote...")
        run_mlops_pipeline()
    else:
        print("⚠️ Retrained model is not better. Generating forecast anyway for monitoring...")
        if is_statsforecast:
            forecast_result_df = model.predict(h=len(test_df)).rename(columns={'Theta': 'yhat'})
            forecast_df = pd.DataFrame({'ds': forecast_result_df['ds'], 'yhat': forecast_result_df['yhat']})
        else:
            forecast_result_df = model.predict(test_df.drop(columns=['ds', 'unique_id', 'y']))
            forecast_df = pd.DataFrame({'ds': test_df['ds'], 'yhat': forecast_result_df})
        save_forecast_to_csv(forecast_df, full_df, "data/forecasts/latest_forecast.csv")

    # Log retrain to MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"LatestRetrain-{datetime.now().isoformat()}"):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = f"{tmpdir}/model.pkl"
            joblib.dump(model, model_file)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=JoblibModelWrapper(),
                artifacts={"model_path": model_file},
            )
            mlflow.log_metric("MAE", mae_val)
            mlflow.set_tag("model_name", "RetrainedModel")
            mlflow.set_tag("retrain_date", datetime.now().isoformat())
            mlflow.set_tag("promotion_decision", "promoted" if best_prev_mae is None or mae_val < best_prev_mae else "rejected")

    print("📁 Latest retrained model and metrics logged to MLflow for audit.")


# def retrain_model(models_uri, train_data_path, target_column='value', experiment_name='RetrainExperiment'):
#     print(f"🚀 Starting retrain with model: {models_uri} and data: {train_data_path}")
#     mlflow.set_tracking_uri("http://localhost:5001")
#     if models_uri is None:
#         print("⚙️ No existing model URI provided. Running full MLOps pipeline...")
#         run_mlops_pipeline()
#         return
#     loaded_model = mlflow.pyfunc.load_model(model_uri=models_uri)
#     model = loaded_model._model_impl.load_context(loaded_model._model_impl._context)
#     train_df, val_df, test_df, _, full_df = load_and_preprocess_data(master_data_path=train_data_path, target_col=target_column)
#     is_statsforecast = hasattr(model, "fit") and hasattr(model, "predict") and hasattr(model, "models")
#     if is_statsforecast:
#         model.fit(train_df[["unique_id", "ds", "y"]])
#         forecast_val = model.predict(h=len(val_df)).rename(columns={'Theta': 'yhat'})
#         val_df['ds'] = pd.to_datetime(val_df['ds'])
#         forecast_val['ds'] = pd.to_datetime(forecast_val['ds'])
#         actual = pd.merge(
#             val_df[['ds', 'unique_id', 'y']],
#             forecast_val[['ds', 'unique_id', 'yhat']],
#             on=['ds', 'unique_id'],
#             how='inner'
#         )
#         mae_val = np.mean(np.abs(actual['y'] - actual['yhat']))
#     else:
#         X_train = train_df.drop(columns=['ds', 'unique_id', 'y'])
#         y_train = train_df['y']
#         model.fit(X_train, y_train)
#         X_val = val_df.drop(columns=['ds', 'unique_id', 'y'])
#         y_val_actual = val_df['y']
#         y_val_pred = model.predict(X_val)
#         mae_val = np.mean(np.abs(y_val_actual - y_val_pred))
#     print(f"📊 MAE on validation set: {mae_val:.4f}")
#     metrics_file = 'artifacts/metrics/model_metrics.json'
#     best_prev_mae = None
#     if os.path.exists(metrics_file):
#         with open(metrics_file, 'r') as f:
#             metrics_data = json.load(f)
#             if metrics_data:
#                 metrics_data_sorted = sorted(metrics_data, key=lambda x: x['MAE'])
#                 best_prev_mae = metrics_data_sorted[0]['MAE']
#                 print(f"📁 Best previous MAE: {best_prev_mae:.4f}")
#     if best_prev_mae is None or mae_val < best_prev_mae:
#         print("✅ New retrained model is better. Running full MLOps pipeline to promote...")
#         run_mlops_pipeline()
#     else:
#         print("⚠️ Retrained model is not better. Generating forecast anyway for monitoring...")
#         if is_statsforecast:
#             forecast_result_df = model.predict(h=len(test_df)).rename(columns={'Theta': 'yhat'})
#             forecast_df = pd.DataFrame({'ds': forecast_result_df['ds'], 'yhat': forecast_result_df['yhat']})
#         else:
#             forecast_result_df = model.predict(test_df.drop(columns=['ds', 'unique_id', 'y']))
#             forecast_df = pd.DataFrame({'ds': test_df['ds'], 'yhat': forecast_result_df})
#         save_forecast_to_csv(forecast_df, full_df, "data/forecasts/latest_forecast.csv")
#     mlflow.set_experiment(experiment_name)
#     with mlflow.start_run(run_name=f"LatestRetrain-{datetime.now().isoformat()}") as run:
#         with tempfile.TemporaryDirectory() as tmpdir:
#             model_path = f"{tmpdir}/model.pkl"
#             joblib.dump(model, model_path)
#             mlflow.pyfunc.log_model(
#                 artifact_path="model",
#                 python_model=JoblibModelWrapper(),
#                 artifacts={"model_path": model_path},
#             )
#             mlflow.log_metric("MAE", mae_val)
#             mlflow.set_tag("model_name", "RetrainedModel")
#             mlflow.set_tag("retrain_date", datetime.now().isoformat())
#             mlflow.set_tag("promotion_decision", "promoted" if best_prev_mae is None or mae_val < best_prev_mae else "rejected")
#     print("📁 Latest retrained model and metrics logged to MLflow for audit.")

if __name__ == "__main__":
    args = parse_args()
    for d in ['data', 'data/forecasts', 'artifacts/metrics', 'artifacts/models', 'artifacts', 'processed_data', 'mlruns']:
        os.makedirs(d, exist_ok=True)
    master_data_path = 'processed_data/merged_data.csv'
    if not os.path.exists(master_data_path):
        print("Creating dummy master_electricity_prices.csv...")
        pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='H'),
            'value': np.random.rand(200) * 100,
            'unique_id': 'series_1'
        }).to_csv(master_data_path, index=False)
    if args.mode != "retrain":
        run_mlops_pipeline(master_data_path=args.train_data, forecast_horizon=24)
    else:
        retrain_best_model(models_uri=args.model_uri, train_data_path=args.train_data)
