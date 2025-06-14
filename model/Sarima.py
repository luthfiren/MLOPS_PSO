# model/Sarima.py
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from logging.handlers import RotatingFileHandler

class SarimaModel:
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,24), freq='h', forecast_horizon=24):
        # Validasi frekuensi
        supported_freq = ['h', 'd', 'w', 'm']
        if freq not in supported_freq:
            raise ValueError(f"Unsupported frequency '{freq}'. Supported: {supported_freq}")
        
        # Parameter model
        self.order = order
        self.seasonal_order = seasonal_order
        self.freq = freq
        self.forecast_horizon = forecast_horizon
        self.model_name = 'SarimaModel'
        self.n_jobs = 1  # SARIMAX tidak mendukung parallel processing
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Path file
        self.model_dir = Path(__file__).parent
        self.log_file = self.model_dir / "training.log"
        self.champion_score_file = self.model_dir / "champion_score.txt"
        self.champion_config_file = self.model_dir / "champion_config.json"
        self.model_file = self.model_dir / f"{self.model_name}_champion.joblib"
        
        # Inisialisasi model
        self.model = None
        
        # Simpan parameter
        self.params = {
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "frequency": self.freq,
            "forecast_horizon": self.forecast_horizon
        }

    def _setup_logger(self):
        """Setup rotating logger sesuai Theta.py"""
        log_path = self.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(self.model_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def train_with_fold(self, folds, optimization=False):
        """Latih model dengan cross-validation"""
        start_time = time.time()
        scores = []
        
        for i, (train_df, test_df) in enumerate(folds):
            # Reset index untuk memastikan datetime sebagai kolom
            train_data = train_df.reset_index(drop=True)
            test_data = test_df.reset_index(drop=True)
            
            # Validasi kolom target
            if 'y' not in train_data.columns:
                raise ValueError("DataFrame harus memiliki kolom 'y'")
            
            # Latih model
            self.model = SARIMAX(train_data['y'], 
                                 order=self.order, 
                                 seasonal_order=self.seasonal_order)
            results = self.model.fit(disp=False)
            
            # Prediksi
            forecast = results.get_forecast(steps=len(test_data))
            pred_mean = forecast.predicted_mean
            
            # Evaluasi
            score = mean_absolute_error(test_data['y'], pred_mean)
            scores.append(score)
        
        avg_score = np.mean(scores)
        elapsed_time = time.time() - start_time
        
        if not optimization:
            self.logger.info(f"{self.model_name} Average MAE across folds = {avg_score:.4f}, Training Time = {elapsed_time:.2f}s")
        
        return avg_score

    def predict(self, pred_df):
        """Hasilkan prediksi"""
        if self.model is None:
            raise ValueError("Model belum dilatih. Panggil metode fit() terlebih dahulu.")
        
        # Validasi kolom 'ds'
        if 'ds' not in pred_df.columns:
            raise ValueError("Input dataframe harus memiliki kolom 'ds'")
        
        # Reset index dan validasi target
        pred_data = pred_df.reset_index(drop=True)
        if 'y' not in pred_data.columns:
            pred_data['y'] = np.nan  # Kolom dummy untuk kompatibilitas
        
        # Prediksi
        forecast = self.model.get_forecast(steps=len(pred_data))
        pred_mean = forecast.predicted_mean
        pred_ci = forecast.conf_int()
        
        # Format hasil
        result = pred_data[['ds']].copy()
        result['yhat'] = pred_mean
        result['lower_bound'] = pred_ci.iloc[:, 0]
        result['upper_bound'] = pred_ci.iloc[:, 1]
        
        return result

    def evaluate(self, actual_df, forecast_df):
        """Evaluasi model dengan MAE dan RMSE"""
        merged = pd.merge(actual_df, forecast_df[['ds', 'yhat']], on='ds', how='inner')
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
        return mae, rmse

    def save(self, model_path=None, config_path=None, score_path=None, score=None):
        """Simpan model dan konfigurasi"""
        model_path = model_path or self.model_file
        config_path = config_path or self.champion_config_file
        
        # Simpan model
        joblib.dump(self, model_path)
        
        # Simpan konfigurasi
        with open(config_path, 'w') as f:
            json.dump(self.params, f, indent=4)
        
        # Simpan skor
        if score is not None:
            with open(score_path or self.champion_score_file, 'w') as f:
                f.write(str(score))
        
        self.logger.info(f"Model disimpan di {model_path}")

    def create_folds(self, df, n_splits, test_size):
        """Buat time series folds"""
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