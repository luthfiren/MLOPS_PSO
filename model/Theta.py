from model.base_model import BaseForecastModel
import logging
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import json
import math
import time
import multiprocessing
from logging.handlers import RotatingFileHandler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from statsforecast.models import Theta
from statsforecast import StatsForecast

class ThetaModel(BaseForecastModel):
    def __init__(self, season_length=12, freq='h', forecast_horizon=24):
        supported_freq = ['h', 'd', 'w', 'm']
        if freq not in supported_freq:
            raise ValueError(f"Unsupported frequency '{freq}'. Supported: {supported_freq}")

        self.season_length = season_length
        self.freq = freq
        self.forecast_horizon = forecast_horizon
        self.model_name = 'ThetaModel'
        
        self.n_jobs = math.ceil(multiprocessing.cpu_count() / 4)
        
        self.logger = self._setup_logger()
        
        self.model_dir = Path(__file__).parent
        self.log_file = self.model_dir / "training.log"
        self.champion_score_file = self.model_dir / "champion_score.txt"
        self.champion_config_file = self.model_dir / "champion_config.json"
        self.model_file = self.model_dir / f"{self.model_name}_champion.joblib"

        self.model = StatsForecast( # model StatsForecast
            models=[Theta(season_length=self.season_length)],
            freq=self.freq,
            n_jobs=self.n_jobs
        )
        
        self.params = { # Simpan parameter aktual yang digunakan untuk instansiasi
            "season_length": self.season_length,
            "frequency": self.freq,
            "forecast_horizon": self.forecast_horizon,
            "n_jobs": self.n_jobs
        }

    def _setup_logger(self):
        log_path = Path(__file__).parent / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(self.model_name)
        logger.setLevel(logging.INFO)
        # Ensure only one handler is added to avoid duplicate logs
        if not logger.handlers:
            handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, mode='a')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False

        return logger

    def train_with_fold(self, folds, optimization=False):
        start_time = time.time()
        scores = []

        for i, (train_df, test_df) in enumerate(folds):
            test_df = test_df.copy() 

            self.model.fit(train_df) # Fit model StatsForecast
            forecast = self.model.predict(h=len(test_df)).rename(columns={'Theta': 'yhat'})

            # Pastikan 'ds' bertipe datetime dengan timezone aware/naive yang sama
            test_df['ds'] = pd.to_datetime(test_df['ds']).dt.tz_localize(None) # Make naive
            forecast['ds'] = pd.to_datetime(forecast['ds']).dt.tz_localize(None) # Make naive

            actual = test_df[test_df['ds'].isin(forecast['ds'])]
            score = mean_absolute_error(actual['y'], forecast['yhat'])
            scores.append(score)

        avg_score = np.mean(scores)
        elapsed_time = time.time() - start_time

        if not optimization:
            self.logger.info(f"{self.model_name} Average MAE across {len(folds)} folds = {avg_score:.4f}, Training Time = {elapsed_time:.2f} seconds")

        return avg_score # Return the average MAE

    def predict(self, pred_df: pd.DataFrame):
        if "ds" not in pred_df.columns:
            raise ValueError("Input dataframe must contain a 'ds' column.")
        
        # Ensure 'ds' column is datetime and naive for both
        pred_df['ds'] = pd.to_datetime(pred_df['ds']).dt.tz_localize(None)

        forecast = self.model.predict(h=len(pred_df))
        forecast["ds"] = pd.to_datetime(forecast["ds"]).dt.tz_localize(None) # Make naive

        # ThetaModel output 'yhat'
        # Ensure that the output DataFrame has 'unique_id', 'ds', and 'yhat'
        if 'unique_id' not in forecast.columns and 'unique_id' in pred_df.columns:
            forecast['unique_id'] = pred_df['unique_id'].iloc[0] # Assume single series for now

        # Align based on 'ds' or return directly if lengths match and order is preserved
        if len(forecast) != len(pred_df):
            # If lengths don't match, you might need more complex alignment or raise error
            raise ValueError(f"Forecast length ({len(forecast)}) mismatch with input data length ({len(pred_df)}).")

        return forecast # Returns a DataFrame with 'ds', 'unique_id', 'yhat'

    def evaluate(self, actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> float:
        actual_df = actual_df.copy()
        forecast_df = forecast_df.copy()
        
        # Ensure 'ds' is datetime and naive for both
        actual_df.loc[:, "ds"] = pd.to_datetime(actual_df["ds"]).dt.tz_localize(None)
        forecast_df.loc[:, "ds"] = pd.to_datetime(forecast_df["ds"]).dt.tz_localize(None)
                
        merged = pd.merge(actual_df, forecast_df, on=["unique_id", "ds"], how="inner") # Merge on unique_id and ds
        
        if merged.empty:
            raise ValueError("No matching timestamps between actual and forecast data during evaluation.")

        score = mean_absolute_error(merged["y"], merged["yhat"])
        return score

    def save(self, model_path=None, config_path=None, score_path=None, score=None):
        model_path = model_path or self.model_file
        config_path = config_path or self.champion_config_file
        config_path = config_path or self.champion_config_file
        score_path = score_path or self.champion_score_file

        joblib.dump(self.model, model_path)
        with open(config_path, "w") as f:
            json.dump(self.params, f, indent=4)
        if score is not None:
            with open(score_path, "w") as f:
                f.write(str(score))

        self.logger.info(f"Saved champion model to {model_path}")
        self.logger.info(f"Saved champion config to {config_path}")
        if score is not None:
            self.logger.info(f"Champion model score ({score}) saved to {score_path}")

    def create_folds(self, df, n_splits, test_size):
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
                break  # not enough data for this fold

            train_df = df.iloc[:train_end]
            test_df = df.iloc[test_start:test_end]
            folds.append((train_df, test_df))
        return folds

    def _generate_future_dates(self, last_train_df: pd.DataFrame, h: int) -> pd.DataFrame:
        unique_id = last_train_df["unique_id"].iloc[0] if "unique_id" in last_train_df.columns else "series_1" # Default if no unique_id
        last_date = pd.to_datetime(last_train_df["ds"].iloc[-1]).dt.tz_localize(None)
        freq = pd.infer_freq(last_train_df["ds"]) or 'H' # Default to hourly if unknown
        future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), periods=h, freq=freq)
        return pd.DataFrame({"unique_id": unique_id, "ds": future_dates, "y": np.nan}) # Add y column for consistency


    def optimize(self, df: pd.DataFrame, forecast_horizon=24, season_list=None):
        self.forecast_horizon = forecast_horizon # Update forecast_horizon

        season_list = season_list or [3, 4, 6, 12, 24] # Add 24 for daily seasonality if hourly data
        if not isinstance(season_list, list) or not all(isinstance(x, int) for x in season_list):
            raise ValueError("season_list must be a list of integers")

        param_grid = ParameterGrid({
            "season_length": season_list,
            "forecast_horizon": [self.forecast_horizon]
        })

        folds = self.create_folds(df, n_splits=3, test_size=self.forecast_horizon)
        if not folds:
            self.logger.warning("Not enough data to create folds for optimization.")
            return float('inf'), None, None, None # Return error values

        best_score = float('inf')
        best_params_found = None
        best_model_obj = None # To store the ThetaModel instance
        best_model_run_id = None # To store the run_id where the best model was logged

        # --- MLflow Outer Run for this optimization process ---
        with mlflow.start_run(run_name=f"Theta_Optimization_Horizon_{forecast_horizon}") as outer_run:
            mlflow.log_param("optimizer_method", "GridSearch")
            mlflow.log_param("optimization_folds", len(folds))

            for i, params in enumerate(param_grid):
                # --- MLflow Nested Run for each parameter combination ---
                with mlflow.start_run(nested=True, run_name=f"Theta_Param_Combo_{i}") as nested_run:
                    current_model_params = {
                        "season_length": params["season_length"],
                        "freq": self.freq, # Use self.freq for consistency
                        "forecast_horizon": params["forecast_horizon"]
                    }
                    
                    # Log parameters for the current run
                    for k, v in current_model_params.items():
                        mlflow.log_param(k, v)

                    try:
                        candidate_model = ThetaModel(**current_model_params)
                        score = candidate_model.train_with_fold(folds, optimization=True) # Score from cross-validation
                        
                        mlflow.log_metric("validation_mae", score)
                        self.logger.info(f"Theta Params: {current_model_params} | MAE: {score:.4f}")

            if score < best_score:
                best_score = score
                best_params = params
                best_model = candidate_model

        if best_model:
            best_model.save(
                model_path=self.model_file,
                config_path=self.champion_config_file,
                score_path=self.champion_score_file,
                score=best_score
            )
            self.logger.info(f"New champion saved: {best_params} | Score: {best_score}")
        else:
            self.logger.warning("No improved model found. Retraining current model as fallback.")
            fallback_model = self.__class__(**self.params)
            fallback_score = fallback_model.train_with_fold(folds)
            fallback_model.save(
                model_path=self.model_file,
                config_path=self.champion_config_file,
                score_path=self.champion_score_file,
                score=fallback_score
            )
            self.logger.info(f"Fallback model retrained and saved with score: {fallback_score}")