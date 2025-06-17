import mlflow.pyfunc
import pandas as pd
import joblib

class ExponentialSmoothingForecastModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["es_model"])

    def predict(self, context, model_input):
        # Ambil horizon prediksi dari panjang input
        horizon = len(model_input)

        # Forecast dengan model statsmodels
        forecast_values = self.model.forecast(steps=horizon)

        # Ambil waktu mulai prediksi dari input
        start_time = pd.to_datetime(model_input['start'].iloc[0])
        future_dates = pd.date_range(start=start_time, periods=horizon, freq='H')

        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values
        })
