from model.base_model import BaseForecastModel
import logging
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import json
import math
import time
import mlflow
import multiprocessing
from logging.handlers import RotatingFileHandler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from statsforecast.models import Theta
from statsforecast import StatsForecast

class ThetaModel(BaseForecastModel):
    def __init__(self, season_length=12, freq='h', forecast_horizon=24, season_list=None, **kwargs):
        supported_freq = ['h', 'd', 'w', 'm']
        if freq not in supported_freq:
            raise ValueError(f"Unsupported frequency '{freq}'. Supported: {supported_freq}")

        self.season_length = season_length
        self.freq = freq
        self.forecast_horizon = forecast_horizon
        self.season_list = season_list
        self.model_name = 'ThetaModel'
        self.n_jobs = math.ceil(multiprocessing.cpu_count() / 4)
        self.model_dir = Path(__file__).parent
        self.log_file = self.model_dir / "training.log"
        self.champion_score_file = self.model_dir / "champion_score.txt"
        self.champion_config_file = self.model_dir / "champion_config.json"
        self.model_file = self.model_dir / f"{self.model_name}_champion.joblib"
        self.mlflow_artifact_path = "champion_thetamodel"
        self.logger = self._setup_logger()

        self.model = StatsForecast(
            models=[Theta(season_length=self.season_length)],
            freq=self.freq,
            n_jobs=self.n_jobs
        )

        self.params = {
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
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df["ds"] = pd.to_datetime(train_df["ds"])
            test_df["ds"] = pd.to_datetime(test_df["ds"])
            train_df["y"] = pd.to_numeric(train_df["y"], errors='coerce')
            test_df["y"] = pd.to_numeric(test_df["y"], errors='coerce')
            if "unique_id" not in train_df.columns:
                train_df["unique_id"] = "series_1"
            if "unique_id" not in test_df.columns:
                test_df["unique_id"] = "series_1"
            train_df["unique_id"] = train_df["unique_id"].astype(str)
            test_df["unique_id"] = test_df["unique_id"].astype(str)
            # Hanya kolom yang diperlukan!
            train_df = train_df[["unique_id", "ds", "y"]]
            test_df = test_df[["unique_id", "ds", "y"]]

            self.model = StatsForecast(
                models=[Theta(season_length=self.season_length)],
                freq=self.freq,
                n_jobs=self.n_jobs
            )
            self.model.fit(train_df)
            forecast = self.model.predict(h=len(test_df)).rename(columns={'Theta': 'yhat'})
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            test_df['ds'] = pd.to_datetime(test_df['ds'])
            actual = pd.merge(
                test_df[['ds', 'unique_id', 'y']],
                forecast[['ds', 'unique_id', 'yhat']],
                on=['ds', 'unique_id'],
                how='inner'
            )
            score = mean_absolute_error(actual['y'], actual['yhat'])
            scores.append(score)
        avg_score = np.mean(scores)
        elapsed_time = time.time() - start_time
        if not optimization:
            self.logger.info(f"{self.model_name} Average MAE across {len(folds)} folds = {avg_score:.4f}, Training Time = {elapsed_time:.2f} seconds")
        return avg_score

    def predict(self, pred_df: pd.DataFrame, h=None):
        pred_df = pred_df.copy()
        if "ds" not in pred_df.columns:
            raise ValueError("Input dataframe must contain a 'ds' column.")
        if "unique_id" not in pred_df.columns:
            pred_df["unique_id"] = "series_1"
        pred_df["ds"] = pd.to_datetime(pred_df["ds"])
        pred_df["unique_id"] = pred_df["unique_id"].astype(str)
        pred_df = pred_df[["unique_id", "ds", "y"]] if "y" in pred_df.columns else pred_df[["unique_id", "ds"]]
        if h is None:
            h = len(pred_df)
        forecast = self.model.predict(h=h)
        forecast["ds"] = pd.to_datetime(forecast["ds"])
        if 'unique_id' not in forecast.columns and 'unique_id' in pred_df.columns:
            forecast['unique_id'] = pred_df['unique_id'].iloc[0]
        if len(forecast) != h:
            raise ValueError(f"Forecast length ({len(forecast)}) mismatch with requested horizon ({h}).")
        return forecast

    def evaluate(self, actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> float:
        actual_df = actual_df.copy()
        forecast_df = forecast_df.copy()
        actual_df["ds"] = pd.to_datetime(actual_df["ds"])
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
        actual_df["unique_id"] = actual_df["unique_id"].astype(str)
        forecast_df["unique_id"] = forecast_df["unique_id"].astype(str)
        merged = pd.merge(actual_df, forecast_df, on=["unique_id", "ds"], how="inner")
        if merged.empty:
            raise ValueError("No matching timestamps between actual and forecast data during evaluation.")
        score = mean_absolute_error(merged["y"], merged["yhat"])
        return score

    def save(self, model_path=None, config_path=None, score_path=None, score=None):
        model_path = model_path or self.model_file
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

    def _generate_future_dates(self, last_train_df: pd.DataFrame, h: int) -> pd.DataFrame:
        unique_id = last_train_df["unique_id"].iloc[0] if "unique_id" in last_train_df.columns else "series_1"
        last_date = pd.to_datetime(last_train_df["ds"].iloc[-1])
        freq = pd.infer_freq(last_train_df["ds"]) or 'H'
        future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), periods=h, freq=freq)
        return pd.DataFrame({"unique_id": unique_id, "ds": future_dates, "y": np.nan})

    def optimize(self, df: pd.DataFrame, forecast_horizon=24, season_list=None, **kwargs):
        self.forecast_horizon = forecast_horizon
        season_list = season_list or [3, 4, 6, 12, 24]
        if not isinstance(season_list, list) or not all(isinstance(x, int) for x in season_list):
            raise ValueError("season_list must be a list of integers")
        
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
        df["y"] = pd.to_numeric(df["y"], errors='coerce')
        if "unique_id" not in df.columns:
            df["unique_id"] = "series_1"
        df["unique_id"] = df["unique_id"].astype(str)
        # Hanya kolom yang diperlukan!
        df = df[["unique_id", "ds", "y"]]

        param_grid = ParameterGrid({
            "season_length": season_list,
            "forecast_horizon": [self.forecast_horizon]
        })
        folds = self.create_folds(df, n_splits=3, test_size=self.forecast_horizon)
        if not folds:
            self.logger.warning("Not enough data to create folds for optimization.")
            return float('inf'), None, None, None
        best_score = float('inf')
        best_params_found = None
        best_model_obj = None
        for params in param_grid:
            try:
                with mlflow.start_run(nested=True, run_name=f"Theta_Season_{params['season_length']}"):
                    candidate_model = ThetaModel(
                        season_length=params["season_length"],
                        freq=self.freq,
                        forecast_horizon=params["forecast_horizon"]
                    )
                    score = candidate_model.train_with_fold(folds, optimization=True)
                    mlflow.log_param("model", "ThetaModel")
                    mlflow.log_param("season_length", params["season_length"])
                    mlflow.log_param("forecast_horizon", self.forecast_horizon)
                    mlflow.log_metric("validation_mae", score)
                    self.logger.info(f"Theta Params: {params} | MAE: {score:.4f}")
                    if score < best_score:
                        best_score = score
                        best_params_found = params
                        best_model_obj = candidate_model
            except Exception as e:
                self.logger.warning(f"Skipping params {params} due to error: {e}")
                continue
        if best_model_obj:
            best_model_obj.save(
                model_path=self.model_file,
                config_path=self.champion_config_file,
                score_path=self.champion_score_file,
                score=best_score
            )
            self.logger.info(f"New champion saved: {best_params_found} | Score: {best_score}")
            return best_score, best_params_found, best_model_obj, "theta_run"
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
            return fallback_score, self.params, fallback_model, "theta_fallback"