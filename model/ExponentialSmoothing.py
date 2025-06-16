# model/ExponentialSmoothing.py
import numpy as np
import logging
import joblib
import pandas as pd
import json
import time
import os
import mlflow
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# -- MLflow Specific Imports --
import mlflow.statsmodels # Untuk melog model statsmodels secara native
import mlflow.pyfunc # Untuk wrapper universal jika diperlukan

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(MODEL_DIR, 'training.log')
CHAMPION_SCORE = os.path.join(MODEL_DIR, 'champion_score.txt')
CHAMPION_CONFIG = os.path.join(MODEL_DIR, 'champion_config.json')
CHAMPION_MODEL = os.path.join(MODEL_DIR, 'champion_model.joblib')

class ExponentialSmoothingModel:
    def __init__(self, has_trend=False, seasonal_periods=12, seasonal_type='add', forecast_horizon=24, damped_trend=False, trend_type=None):
        self.has_trend = has_trend
        self.seasonal_periods = seasonal_periods
        self.seasonal_type = seasonal_type
        self.forecast_horizon = forecast_horizon
        self.damped_trend = damped_trend
        self.trend_type = trend_type
        self.model = None
        self.model_name = 'Exponential Smoothing Model'
        self.logger = self._setup_logger()
        self.params = { # Simpan parameter aktual yang digunakan untuk instansiasi
            "has_trend": has_trend,
            "seasonal_periods": seasonal_periods,
            "seasonal_type": seasonal_type,
            "forecast_horizon": forecast_horizon,
            "damped_trend": damped_trend,
            "trend_type": trend_type
        }

    def _setup_logger(self):
        log_path = Path(__file__).parent / "training.log"
        logger = logging.getLogger(self.model_name)
        # Ensure only one handler is added to avoid duplicate logs
        if not logger.handlers:
            handler = logging.FileHandler(log_path, mode='a')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent propagation to root logger to avoid duplicate console output if root logger is configured
        logger.propagate = False
        return logger
 
    def train_with_fold(self, folds, optimizing=False):
        start_time = time.time()
        scores = []
        model_smoothing_name = None

        for i, (train_df, test_df) in enumerate(folds):
            train_series = train_df['y'].values
            test_series = test_df['y'].values

            # Select model type
            if not self.has_trend and (self.seasonal_periods is None or self.seasonal_periods == 0): # SES
                model = SimpleExpSmoothing(train_series).fit(optimized=True)
                model_smoothing_name = 'SimpleExpSmoothing'
            elif self.has_trend and (self.seasonal_periods is None or self.seasonal_periods == 0): # Holt
                # Holt can have 'add' or 'mul' trend, but statsmodels Holt class directly uses 'exponential' param
                # if trend_type is 'mul', set exponential=True, else False
                model = Holt(train_series, damped_trend=self.damped_trend, exponential=self.trend_type == 'mul').fit(optimized=True)
                model_smoothing_name = "Holt"
            elif self.seasonal_periods: # ExponentialSmoothing (Holt-Winters)
                model = ExponentialSmoothing(
                    train_series, 
                    trend=self.trend_type, # Use self.trend_type directly ('add' or None)
                    seasonal=self.seasonal_type, 
                    seasonal_periods=self.seasonal_periods,
                    damped_trend=self.damped_trend
                ).fit(optimized=True)
                model_smoothing_name = "ExponentialSmoothing"
            else:
                raise RuntimeError("Invalid configuration for Exponential Smoothing model.")

            self.model = model # Simpan model yang dilatih di instance

            # Prediksi
            forecast = model.forecast(len(test_series))

            # Evaluasi
            actual = test_series # test_series sudah numpy array dari test_df['y']
            score = mean_absolute_error(actual, forecast)
            scores.append(score)

        avg_score = np.mean(scores)
        elapsed_time = time.time() - start_time
        
        if optimizing == False:
            self.model_name = model_smoothing_name
            self.logger.info(f"{self.model_name} Average MAE across {len(folds)} folds = {avg_score:.4f}, Training Time = {elapsed_time:.2f} seconds")
        
        return avg_score # Return the average MAE

    def predict(self, pred_df: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        
        # Ensure 'ds' column is present and is datetime-like
        if 'ds' not in pred_df.columns:
            raise ValueError("Input DataFrame for predict must contain a 'ds' column.")
        
        # Sort by 'ds' to ensure correct time order for forecasting
        pred_df = pred_df.sort_values(by='ds').reset_index(drop=True)
        
        h = len(pred_df) # Forecast horizon is determined by the length of pred_df
        forecast_values = self.model.forecast(steps=h)

        # Create DataFrame for results, using 'ds' from pred_df
        forecast_df = pd.DataFrame({
            'ds': pred_df['ds'],
            'yhat': forecast_values
        })
        
        # Ensure unique_id if needed, assuming single series for ES for simplicity
        if 'unique_id' in pred_df.columns and 'unique_id' not in forecast_df.columns:
            forecast_df['unique_id'] = pred_df['unique_id'].iloc[0]
            
        return forecast_df


    def evaluate(self, actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> float:
        # Menyatukan dataframe berdasarkan 'ds' (timestamp)
        # Ensure 'ds' is datetime for both and timezone-naive for consistent merging
        actual_df['ds'] = pd.to_datetime(actual_df['ds']).dt.tz_localize(None)
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.tz_localize(None)

        merged = pd.merge(actual_df[['ds', 'y']], forecast_df[['ds', 'yhat']], on="ds", how="inner")
        
        if merged.empty:
            raise ValueError("No matching timestamps between actual and forecast data during evaluation.")

        score = mean_absolute_error(merged["y"], merged["yhat"])
        self.logger.info(f"{self.model_name} MAE={score:.4f} with params={self.params}, horizon={self.forecast_horizon}")
        return score
    
    def save(self, path: str = None):
        """Save the fitted model."""
        if path is None:
            path = Path(__file__).parent / "champion_model.joblib"
        joblib.dump(self.model, path)
        self.logger.info(f"Saved champion model to {path}")

    def create_folds(self, df, n_splits=3, test_size=24):
        """
        Time series split into n_splits folds with fixed test_size,
        growing training data, and sequential test sets (non-overlapping).
        Returns list of (train_df, test_df) tuples.
        """
        folds = []
        total_points = len(df)
        step = (total_points - test_size * n_splits) // n_splits

        for i in range(n_splits):
            train_end = step * (i + 1) + test_size * i
            test_start = train_end
            test_end = test_start + test_size

            if test_end > total_points:
                break
            train_df = df.iloc[:train_end]
            test_df = df.iloc[test_start:test_end]
            folds.append((train_df, test_df))
        return folds

    def optimize(self, df: pd.DataFrame, forecast_horizon=24, season_list=None):
        self.folds_to_train = self.create_folds(df, n_splits=3, test_size=forecast_horizon)
        self.forecast_horizon = forecast_horizon # Update forecast_horizon

        # Define parameter grid based on model configuration (trend, seasonal)
        param_grid_dict = {
            "forecast_horizon": [forecast_horizon] # Always include horizon
        }
        
        # Ensure df has 'y' and 'ds' columns for folds creation
        if 'y' not in df.columns or 'ds' not in df.columns:
            raise ValueError("Input DataFrame for optimize must contain 'y' and 'ds' columns.")

        if self.has_trend and self.seasonal_periods: # Holt-Winters (with trend & seasonality)
            param_grid_dict["seasonal_periods"] = season_list if season_list else [self.seasonal_periods]
            param_grid_dict["seasonal_type"] = [None, "add", "mul"]
            param_grid_dict["damped_trend"] = [True, False]
            param_grid_dict["trend_type"] = ['add', 'mul'] # Both add and mul are valid for ExponentialSmoothing
        elif self.has_trend and not self.seasonal_periods: # Holt (with trend, no seasonality)
            param_grid_dict["damped_trend"] = [True, False]
            param_grid_dict["trend_type"] = ['add', 'mul'] # Holt can have 'add' or 'mul' trend
        elif not self.has_trend and self.seasonal_periods: # Pure Seasonal ES (no trend)
            param_grid_dict["seasonal_periods"] = season_list if season_list else [self.seasonal_periods]
            param_grid_dict["seasonal_type"] = [None, "add", "mul"]
        # If no trend and no seasonality, it's SES, no specific params to optimize beyond forecast_horizon

        param_grid = ParameterGrid(param_grid_dict)
        
        best_score = float('inf')
        best_params_found = None
        best_model_obj = None # To store the ExponentialSmoothingModel instance

        # --- MLflow Outer Run for this optimization process ---
        # This run tracks the overall optimization attempt for ES model
        with mlflow.start_run(run_name=f"ES_Optimization_Horizon_{forecast_horizon}") as outer_run:
            mlflow.log_param("optimizer_method", "GridSearch")
            mlflow.log_param("optimization_folds", len(self.folds_to_train))
            
            for i, params in enumerate(param_grid):
                # --- MLflow Nested Run for each parameter combination ---
                # This run tracks a single ES model training with specific params
                with mlflow.start_run(nested=True, run_name=f"ES_Param_Combo_{i}") as nested_run:
                    current_model_params = {
                        "has_trend": self.has_trend, # This is a fixed param from the instance init
                        "seasonal_periods": params.get("seasonal_periods", self.seasonal_periods),
                        "seasonal_type": params.get("seasonal_type", self.seasonal_type),
                        "forecast_horizon": params.get("forecast_horizon", forecast_horizon),
                        "damped_trend": params.get("damped_trend", False),
                        "trend_type": params.get("trend_type", None)
                    }
                    
                    # Log parameters for the current run
                    for k, v in current_model_params.items():
                        mlflow.log_param(k, v)

                    try:
                        # Instantiate a new model with current parameters for training
                        candidate_model = ExponentialSmoothingModel(**current_model_params)
                        # Pass the folds directly to train_with_fold
                        score = candidate_model.train_with_fold(self.folds_to_train, optimizing=True)
                        
                        mlflow.log_metric("validation_mae", score)
                        self.logger.info(f"ES Params: {current_model_params} | MAE: {score:.4f}")

                        if score < best_score:
                            best_score = score
                            best_params_found = current_model_params # Simpan parameter lengkap
                            best_model_obj = candidate_model # Simpan instance model terbaik
                            # Capture the run_id of this nested run, which holds the best model artifact
                            best_model_run_id = nested_run.info.run_id
                            
                    except Exception as e:
                        self.logger.warning(f"Skipping ES params {current_model_params} due to error: {e}")
                        mlflow.log_param("status", "failed")
                        mlflow.log_param("error", str(e))
                        continue

            # --- Setelah loop param_grid selesai (kembali ke outer run) ---
            if best_model_obj:
                # Log the best overall parameters for this optimization run
                mlflow.log_param("best_params_found", json.dumps(best_params_found))
                mlflow.log_metric("best_validation_mae", best_score)
                mlflow.set_tag("best_model_type", "ExponentialSmoothing") # Add tag for traceability
                
                # Simpan model terbaik ke dalam MLflow Artifacts
                # Ini akan disimpan di bawah run_id optimasi saat ini
                # Pastikan candidate_model.model adalah objek statsmodels yang sudah di-fit
                mlflow.statsmodels.log_model(
                    statsmodels_model=best_model_obj.model, 
                    artifact_path="champion_es_model",
                    # Refer to the original run where it was trained for better traceability
                    # If you want to link to the run where it was *trained* (nested_run.info.run_id), 
                    # you need to pass it here. For simplicity, we log it under the current (outer) run.
                    # This model will later be registered to the MLflow Model Registry from modelling.py
                )
                self.logger.info(f"ES Champion model logged to MLflow artifacts (run_id: {outer_run.info.run_id}).")

                # Simpan champion ke joblib dan config lokal (opsional, tapi berguna untuk fallback)
                best_model_obj.save(CHAMPION_MODEL)
                with open(CHAMPION_CONFIG, 'w') as f:
                    json.dump(best_params_found, f, indent=4)
                self.logger.info(f"ES Champion model and config saved locally.")

                return best_score, best_params_found, best_model_obj, outer_run.info.run_id # Return the run_id
            else:
                self.logger.warning("No improved ES model found during optimization.")
                return float('inf'), None, None, outer_run.info.run_id # Return sentinel values
