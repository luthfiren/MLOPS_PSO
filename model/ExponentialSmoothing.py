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

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(MODEL_DIR, 'training.log')
CHAMPION_SCORE = os.path.join(MODEL_DIR, 'champion_score.txt')
CHAMPION_CONFIG = os.path.join(MODEL_DIR, 'champion_config.json')
CHAMPION_MODEL = os.path.join(MODEL_DIR, 'champion_model.joblib')

# Model ES (Simple Exponential)
class ExponentialSmoothingModel:
    def __init__(self, has_trend=False, seasonal_periods=12, seasonal_type='add', forecast_horizon=24, damped_trend=False, trend_type=None):
        self.has_trend = has_trend
        self.seasonal_periods = seasonal_periods  # or some default
        self.seasonal_type = seasonal_type
        self.forecast_horizon = forecast_horizon
        self.damped_trend = damped_trend
        self.trend_type = trend_type
        self.model = None
        self.model_name = 'Exponential Smoothing Model'
        self.logger = self._setup_logger()
        self.params = 'baseline'

    def _setup_logger(self):
        log_path = Path(__file__).parent / "training.log"
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

            # Ensure index/DS alignment if needed
            test_df.loc[:, 'ds'] = pd.to_datetime(test_df['ds']).dt.tz_localize(None)
            actual = test_df[test_df['ds'].isin(test_df['ds'])]['y'].values

            score = mean_absolute_error(actual, forecast)
            scores.append(score)

        avg_score = np.mean(scores)
        elapsed_time = time.time() - start_time
        
        if optimizing == False:
            self.model_name = model_smoothing_name
            self.logger.info(f"{self.model_name} Average MAE across {len(folds)} folds = {avg_score:.4f}, Training Time = {elapsed_time:.2f} seconds")
        
        return avg_score        

    def predict(self, pred_df, h=None):
        if h is None:
            h = pred_df.shape[0]

        # Generate forecast values
        forecast_values = self.model.forecast(steps=h)

        # Assign 'yhat' as a new column in val_df
        preds_df = pred_df.copy()  # to avoid modifying the original df outside
        preds_df.loc[:h-1, "y"] = forecast_values.astype(preds_df["y"].dtype)

        return preds_df

    def evaluate(self, actual_df: pd.DataFrame, forecast_df: pd.DataFrame):
        merged = actual_df.merge(forecast_df, on=["unique_id", "ds"])
        score = mean_absolute_error(merged["y"], merged["yhat"])
        self.logger.info(f"{self.model_name} MAE={score:.4f} with season_length={self.season_length}, horizon={self.forecast_horizon}")
        return score
    
    def save(self, path: str = None):
        if path is None:
            path = Path(__file__).parent / "model_champion.joblib"
        joblib.dump(self.model, path)
        self.logger.info(f"Saved champion model with params {self.params} to {path}")

    def retrain(self, data: pd.DataFrame):
        self.ds = pd.to_datetime(data['ds'])
        self.time_series = np.asarray(data['y'])
                
        with open(CHAMPION_CONFIG, 'r') as f:
            best_config = json.load(f)
        
        # Get last key-value pair
        if best_config:
            last_key = list(best_config.keys())[-1]
            last_value = best_config[last_key]
            
        champion_model = self.train(self.time_series, **best_config)
        joblib.dump(champion_model, CHAMPION_MODEL)
        
        self.logger.info(f"RETRAINED CHAMPION | CONFIG: {best_config}")
        return champion_model
    
    def create_folds(self, df, n_splits=3, test_size=24):
        """
        Time series split into n_splits folds with fixed test_size,
        growing training data, and sequential test sets (non-overlapping).
        Returns list of (train_df, test_df) tuples.
        """

        folds = []
        total_points = len(df)
        step = (total_points - test_size * n_splits) // n_splits  # training expansion step

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

    def optimize(self, df, forecast_horizon=24, season_list=None):
        # Fix param names: use 'seasonal_periods' to match constructor
        if not self.has_trend and (season_list is None or len(season_list) == 0):
            param_grid = ParameterGrid({
                "forecast_horizon": [forecast_horizon],
            })
        elif self.has_trend and (season_list is None or len(season_list) == 0):
            param_grid = ParameterGrid({
                "damped_trend": [True, False],
                "forecast_horizon": [forecast_horizon],
                "trend_type": ['add', None],  # Ensure 'trend_type' aligns
            })
        else:
            param_grid = ParameterGrid({
                "seasonal_periods": season_list,      # <- match constructor
                "seasonal_type": [None, "add", "mul"],  # <- match constructor
                "trend_type": [None, "add"],            # <- match constructor
                "damped_trend": [True, False],
                "forecast_horizon": [forecast_horizon],
            })

        self.params = param_grid
        
        best_score = float('inf')
        best_params = None
        best_model = None

        for params in param_grid:
            try:
                model = self.__class__(
                    has_trend=self.has_trend,
                    seasonal_periods=params.get("seasonal_periods"),
                    seasonal_type=params.get("seasonal_type", self.seasonal_type),
                    forecast_horizon=params.get("forecast_horizon", forecast_horizon),
                    damped_trend=params.get("damped_trend", False),
                    trend_type=params.get("trend_type", None),
                )
                score = model.train_with_fold(self.folds_to_train, optimizing=True)

                if score < best_score:
                    best_score = score
                    best_params = params
                    best_model = model

            except Exception as e:
            #     self.logger.warning(f"Skipping params {params} due to error: {e}")
                continue

        if best_model:
            # Save model
            joblib.dump(best_model, CHAMPION_MODEL)

            # Save best hyperparameters (config) as JSON
            with open(CHAMPION_CONFIG, 'w') as f:
                json.dump(best_params, f, indent=4)

            self.logger.info(f"Champion model and config saved to: {CHAMPION_MODEL}")

        else:
            self.logger.warning("No model improved the score")