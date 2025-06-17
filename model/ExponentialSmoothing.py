import numpy as np
import logging
import joblib
import pandas as pd
import json
import time
import os
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from sklearn.metrics import mean_absolute_error
from pathlib import Path

from model.base_model import BaseForecastModel

class ExponentialSmoothingModel(BaseForecastModel):
    def __init__(self, has_trend=False, seasonal_periods=12, seasonal_type='add', forecast_horizon=24, damped_trend=False, trend_type=None):
        self.has_trend = has_trend
        self.seasonal_periods = seasonal_periods
        self.seasonal_type = seasonal_type
        self.forecast_horizon = forecast_horizon
        self.damped_trend = damped_trend
        self.trend_type = trend_type
        self.model = None
        self.model_name = 'ExponentialSmoothingModel'

        self.model_dir = Path(__file__).parent
        self.log_file = self.model_dir / "training.log"
        self.champion_score_file = self.model_dir / "champion_score.txt"
        self.champion_config_file = self.model_dir / "champion_config.json"
        self.model_file = self.model_dir / f"{self.model_name}_champion.joblib"

        self.logger = self._setup_logger()
        self.params = {
            "has_trend": self.has_trend,
            "seasonal_periods": self.seasonal_periods,
            "seasonal_type": self.seasonal_type,
            "forecast_horizon": self.forecast_horizon,
            "damped_trend": self.damped_trend,
            "trend_type": self.trend_type
        }

    def _setup_logger(self):
        log_path = self.log_file
        logger = logging.getLogger(self.model_name)
        handler = logging.FileHandler(log_path, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def train_with_fold(self, folds, optimizing=False):
        self.folds_to_train = folds
        start_time = time.time()
        scores = []
        model_smoothing_name = None

        for i, (train_df, test_df) in enumerate(folds):
            train_series = train_df['y'].values
            test_series = test_df['y'].values

            # Select model type
            if not self.has_trend and not self.seasonal_periods:
                model = SimpleExpSmoothing(train_series).fit(optimized=True)
                model_smoothing_name = 'SimpleExpSmoothing'
            elif self.has_trend and not self.seasonal_periods:
                model = Holt(train_series).fit(optimized=True)
                model_smoothing_name = "Holt"
            elif self.seasonal_periods:
                model = ExponentialSmoothing(
                    train_series, 
                    trend='add' if self.has_trend else None, 
                    seasonal=self.seasonal_type, 
                    seasonal_periods=self.seasonal_periods
                ).fit(optimized=True)
                model_smoothing_name = "ExponentialSmoothing"
            else:
                raise RuntimeError("Invalid configuration")

            self.model = model
            forecast = model.forecast(len(test_series))

            test_df.loc[:, 'ds'] = pd.to_datetime(test_df['ds']).dt.tz_localize(None)
            actual = test_df[test_df['ds'].isin(test_df['ds'])]['y'].values

            score = mean_absolute_error(actual, forecast)
            scores.append(score)

        avg_score = np.mean(scores)
        elapsed_time = time.time() - start_time

        if not optimizing:
            self.model_name = model_smoothing_name
            self.logger.info(f"{self.model_name} Average MAE across {len(folds)} folds = {avg_score:.4f}, Training Time = {elapsed_time:.2f} seconds")

        return avg_score

    def predict(self, pred_df, h=None):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        if h is None:
            h = pred_df.shape[0]

        forecast_values = self.model.forecast(steps=h)

        preds_df = pred_df.copy()
        preds_df = preds_df.iloc[:h].copy()
        preds_df["yhat"] = forecast_values.astype(float)
        return preds_df

    def evaluate(self, actual_df: pd.DataFrame, forecast_df: pd.DataFrame):
        merged = actual_df.merge(forecast_df, on=["unique_id", "ds"])
        score = mean_absolute_error(merged["y"], merged["yhat"])
        self.logger.info(f"{self.model_name} MAE={score:.4f} with horizon={self.forecast_horizon}")
        return score

    def save(self, model_path=None, config_path=None, score_path=None, score=None):
        model_path = model_path or self.model_file
        config_path = config_path or self.champion_config_file
        score_path = score_path or self.champion_score_file

        joblib.dump(self.model, model_path)
        with open(config_path, 'w') as f:
            json.dump(self.params, f, indent=4)
        if score is not None:
            with open(score_path, 'w') as f:
                f.write(str(score))
        self.logger.info(f"Saved champion model with params {self.params} to {model_path}")

    def retrain(self, data: pd.DataFrame):
        self.ds = pd.to_datetime(data['ds'])
        self.time_series = np.asarray(data['y'])

        with open(self.champion_config_file, 'r') as f:
            best_config = json.load(f)

        if best_config:
            last_key = list(best_config.keys())[-1]
            last_value = best_config[last_key]

        champion_model = self.train(self.time_series, **best_config)
        joblib.dump(champion_model, self.model_file)
        self.logger.info(f"RETRAINED CHAMPION | CONFIG: {best_config}")
        return champion_model

    def create_folds(self, df, n_splits=3, test_size=24):
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

    def optimize(self, df, forecast_horizon=24, season_list=None):
        if not self.has_trend and (season_list is None or len(season_list) == 0):
            param_grid = ParameterGrid({
                "forecast_horizon": [forecast_horizon],
            })
        elif self.has_trend and (season_list is None or len(season_list) == 0):
            param_grid = ParameterGrid({
                "damped_trend": [True, False],
                "forecast_horizon": [forecast_horizon],
                "trend_type": ['add', None],
            })
        else:
            param_grid = ParameterGrid({
                "seasonal_periods": season_list,
                "seasonal_type": [None, "add", "mul"],
                "trend_type": [None, "add"],
                "damped_trend": [True, False],
                "forecast_horizon": [forecast_horizon],
            })

        self.params = param_grid

        best_score = float('inf')
        best_params = None
        best_model = None

        folds = self.create_folds(df, n_splits=3, test_size=forecast_horizon)

        for params in param_grid:
            try:
                model = self.__class__(
                    has_trend=self.has_trend,
                    seasonal_periods=params.get("seasonal_periods", self.seasonal_periods),
                    seasonal_type=params.get("seasonal_type", self.seasonal_type),
                    forecast_horizon=params.get("forecast_horizon", forecast_horizon),
                    damped_trend=params.get("damped_trend", False),
                    trend_type=params.get("trend_type", None),
                )
                score = model.train_with_fold(folds, optimizing=True)

                if score < best_score:
                    best_score = score
                    best_params = params
                    best_model = model

            except Exception as e:
                # self.logger.warning(f"Skipping params {params} due to error: {e}")
                continue

        if best_model:
            best_model.save(
                model_path=self.model_file,
                config_path=self.champion_config_file,
                score_path=self.champion_score_file,
                score=best_score,
            )
            self.logger.info(f"Champion model and config saved to: {self.model_file}")
        else:
            self.logger.warning("No model improved the score")