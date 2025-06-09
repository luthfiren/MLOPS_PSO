# model/theta_model.py
import logging
from pathlib import Path
import pandas as pd
import joblib
import json
import math
import multiprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from statsforecast.models import Theta
from statsforecast import StatsForecast

class ThetaModel:
    def __init__(self, season_length=12, freq='H', forecast_horizon=24):
        self.season_length = season_length
        self.freq = freq
        self.forecast_horizon = forecast_horizon
        self.model_name = 'ThetaModel'
        self.logger = self._setup_logger()
        self.model = StatsForecast(
            models=[Theta(season_length=self.season_length)],
            freq=self.freq,
            n_jobs=math.ceil(multiprocessing.cpu_count()/4)
        )

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

    def train_with_fold(self, folds):
        scores = []

        for i, (train_df, test_df) in enumerate(folds):
            model = StatsForecast(
                models=[Theta(season_length=self.season_length)],
                freq=self.freq,
                n_jobs=math.ceil(multiprocessing.cpu_count() / 4)
            )

            model.fit(train_df)
            forecast = model.predict(h=len(test_df))
            forecast = forecast.rename(columns={'Theta': 'yhat'})
            
            test_df['ds'] = pd.to_datetime(test_df['ds']).dt.tz_convert('UTC')
            forecast['ds'] = pd.to_datetime(forecast['ds']).dt.tz_convert('UTC')
            
            actual = test_df[test_df["ds"].isin(forecast["ds"])]
            score = mean_absolute_error(actual["y"], forecast["yhat"])
            scores.append(score)
            
        avg_score = sum(scores) / len(scores)
        self.logger.info(f"{self.model_name} Average MAE across {len(folds)} folds = {avg_score:.4f}")
        return avg_score

    def predict(self, h=None):
        if h is None:
            h = self.forecast_horizon
        return self.model.predict(h=h)

    def evaluate(self, actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> float:
        merged = actual_df.merge(forecast_df, on=["unique_id", "ds"])
        score = mean_absolute_error(merged["y"], merged["yhat"])
        self.logger.info(f"{self.model_name} MAE={score:.4f} with season_length={self.season_length}, horizon={self.forecast_horizon}")
        return score

    def save(self, path: str = None):
        if path is None:
            path = Path(__file__).parent / f"{self.model_name}.joblib"
        joblib.dump(self.model, path)
        self.logger.info(f"Saved champion model {self.model_name} to {path}")

    @classmethod
    def optimize(cls, df: pd.DataFrame, season_list=None):
        
        # Validate input type
        if not isinstance(season_list, list):
            raise TypeError("season_list must be a list of integers")

        # Optionally validate elements are integers
        if not all(isinstance(x, int) for x in season_list):
            raise ValueError("All elements in season_list must be integers")
        
        if season_list is None:
            season_list = [3, 6, 12]

        param_grid = ParameterGrid({
            "season_length": season_list,  # example tuning range for hourly data
            "forecast_horizon": [24]
        })

        CHAMPION_SCORE_PATH = Path(__file__).parent / "champion_score.txt"
        CHAMPION_CONFIG_PATH = Path(__file__).parent / "champion_config.json"

        best_score = float("inf")
        best_params = None
        best_model = None

        for params in param_grid:
            model = cls(**params)
            model.train(df)
            forecast = model.predict()
            actual = df[df["ds"].isin(forecast["ds"])]
            score = model.evaluate(actual, forecast)

            if score < best_score:
                best_score = score
                best_params = params
                best_model = model

        if best_model:
            best_model.save()
            with open(CHAMPION_SCORE_PATH, "w") as f:
                f.write(str(best_score))
            with open(CHAMPION_CONFIG_PATH, "w") as f:
                json.dump(best_params, f)
            print(f"New champion saved: {best_params}, MAE: {best_score}")
        else:
            print("No model improved the score.")