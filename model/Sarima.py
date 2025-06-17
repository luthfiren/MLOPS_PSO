import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
from logging.handlers import RotatingFileHandler

from model.base_model import BaseForecastModel

class SarimaModel(BaseForecastModel):
    def __init__(
        self,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        freq='h',
        forecast_horizon=24,
        **kwargs  # untuk kompatibilitas dengan pipeline otomatis
    ):
        supported_freq = ['h', 'd', 'w', 'm']
        if freq not in supported_freq:
            raise ValueError(f"Unsupported frequency '{freq}'. Supported: {supported_freq}")

        self.order = order
        self.seasonal_order = seasonal_order
        self.freq = freq
        self.forecast_horizon = forecast_horizon
        self.model_name = 'SarimaModel'
        self.n_jobs = 1  # SARIMAX tidak mendukung parallel processing

        self.model_dir = Path(__file__).parent
        self.log_file = self.model_dir / "training.log"
        self.champion_score_file = self.model_dir / "champion_score.txt"
        self.champion_config_file = self.model_dir / "champion_config.json"
        self.model_file = self.model_dir / f"{self.model_name}_champion.joblib"
        self.mlflow_artifact_path = "champion_sarimamodel"

        self.logger = self._setup_logger()
        self.model = None

        self.params = {
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "frequency": self.freq,
            "forecast_horizon": self.forecast_horizon
        }

    def _setup_logger(self):
        log_path = self.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(self.model_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def train_with_fold(self, folds, optimization=False):
        start_time = time.time()
        scores = []

        for i, (train_df, test_df) in enumerate(folds):
            train_data = train_df.reset_index(drop=True)
            test_data = test_df.reset_index(drop=True)

            if 'y' not in train_data.columns:
                raise ValueError("DataFrame harus memiliki kolom 'y'")

            model = SARIMAX(
                train_data['y'],
                order=self.order,
                seasonal_order=self.seasonal_order
            )
            results = model.fit(disp=False)
            self.model = results

            forecast = results.get_forecast(steps=len(test_data))
            pred_mean = forecast.predicted_mean

            score = mean_absolute_error(test_data['y'], pred_mean)
            scores.append(score)

        avg_score = np.mean(scores)
        elapsed_time = time.time() - start_time

        if not optimization:
            self.logger.info(
                f"{self.model_name} Average MAE across folds = {avg_score:.4f}, Training Time = {elapsed_time:.2f}s"
            )

        return avg_score

    def predict(self, pred_df, h=None):
        if self.model is None:
            raise ValueError("Model belum dilatih. Panggil metode fit() terlebih dahulu.")

        if 'ds' not in pred_df.columns:
            raise ValueError("Input dataframe harus memiliki kolom 'ds'")

        pred_data = pred_df.reset_index(drop=True)
        if 'y' not in pred_data.columns:
            pred_data['y'] = np.nan

        steps = h if (h is not None) else len(pred_data)
        forecast = self.model.get_forecast(steps=steps)
        pred_mean = forecast.predicted_mean
        pred_ci = forecast.conf_int()

        result = pred_data[['ds']].copy()
        result = result.iloc[:steps].copy()
        result['yhat'] = pred_mean.values[:steps]
        result['lower_bound'] = pred_ci.iloc[:steps, 0]
        result['upper_bound'] = pred_ci.iloc[:steps, 1]

        return result
    
    def evaluate(self, actual_df, forecast_df):
        merged = pd.merge(actual_df, forecast_df[['ds', 'yhat']], on='ds', how='inner')
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
        return mae 

    def save(self, model_path=None, config_path=None, score_path=None, score=None):
        model_path = model_path or self.model_file
        config_path = config_path or self.champion_config_file

        joblib.dump(self, model_path)

        with open(config_path, 'w') as f:
            json.dump(self.params, f, indent=4)

        if score is not None:
            with open(score_path or self.champion_score_file, 'w') as f:
                f.write(str(score))

        self.logger.info(f"Model disimpan di {model_path}")

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

    def optimize(self, df, forecast_horizon=24, season_list=None, order_grid=None, seasonal_order_grid=None, n_splits=3, **kwargs):
        """
        Grid search untuk mencari kombinasi order dan seasonal_order terbaik berdasarkan MAE validasi.
        season_list diterima agar interface konsisten, walau tidak digunakan.
        """
        order_grid = order_grid or [
            (1, 1, 1), (2, 1, 1), (1, 0, 1)
        ]
        seasonal_order_grid = seasonal_order_grid or [
            (1, 1, 1, 24), (0, 1, 1, 24), (1, 0, 1, 24)
        ]

        folds = self.create_folds(df, n_splits=n_splits, test_size=forecast_horizon)
        best_score = float('inf')
        best_params = None
        best_model_obj = None

        grid = ParameterGrid({
            "order": order_grid,
            "seasonal_order": seasonal_order_grid
        })

        for params in grid:
            try:
                candidate = SarimaModel(
                    order=params["order"],
                    seasonal_order=params["seasonal_order"],
                    freq=self.freq,
                    forecast_horizon=forecast_horizon
                )
                score = candidate.train_with_fold(folds, optimization=True)
                self.logger.info(f"SARIMA Params: {params} | MAE: {score:.4f}")

                if score < best_score:
                    best_score = score
                    best_params = params
                    best_model_obj = candidate
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
            self.logger.info(f"SARIMA Champion: Params={best_params} | Score={best_score:.4f}")
            # Penting: return 4 value agar kompatibel dengan pipeline otomatis
            return best_score, best_params, best_model_obj, "sarima_run"
        else:
            self.logger.warning("No improved SARIMA model found during optimization.")
            return None, None, None, None