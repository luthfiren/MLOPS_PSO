import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

# Asumsi: Anda memiliki file 'forecasts.csv' (output modelling.py)
# dan 'historical_actuals.csv' (subset data historis Anda untuk perbandingan)

def plot_forecast_vs_actual(forecast_file, actuals_file, date_range=None):
    try:
        df_forecast = pd.read_csv(forecast_file, parse_dates=['tanggal_jam'])
        df_actuals = pd.read_csv(actuals_file, parse_dates=['tanggal_jam'])

        # Pastikan kolom yang relevan ada
        df_forecast = df_forecast[['tanggal_jam', 'predicted_price']]
        df_actuals = df_actuals[['tanggal_jam', 'actual_price']]

        # Gabungkan data untuk perbandingan
        df_merged = pd.merge(df_forecast, df_actuals, on='tanggal_jam', how='inner')

        if date_range:
            df_merged = df_merged[(df_merged['tanggal_jam'] >= date_range[0]) & (df_merged['tanggal_jam'] <= date_range[1])]

        if df_merged.empty:
            return "No data for the selected range."

        # Hitung MAE untuk periode yang divisualisasikan
        mae = (df_merged['predicted_price'] - df_merged['actual_price']).abs().mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_merged['tanggal_jam'], df_merged['predicted_price'], label='Harga Prediksi', color='blue')
        ax.plot(df_merged['tanggal_jam'], df_merged['actual_price'], label='Harga Aktual (Historis)', color='red', linestyle='--')

        ax.set_title(f'Prediksi vs. Aktual Harga Listrik (MAE: {mae:.2f})')
        ax.set_xlabel('Tanggal dan Jam')
        ax.set_ylabel('Harga (€/MWh)')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Konversi plot ke base64 string untuk ditampilkan di HTML
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig) # Penting untuk menutup figur agar tidak memakan memori

        return img_base64

    except FileNotFoundError:
        return "Data files not found. Please ensure 'forecasts.csv' and 'historical_actuals.csv' exist."
    except Exception as e:
        return f"An error occurred: {e}"

# Di dalam route Flask Anda:
# @app.route('/')
# def index():
#     plot_data = plot_forecast_vs_actual('path/to/data/forecasts/latest_forecast.csv', 'path/to/data/historical_actuals.csv')
#     return render_template('index.html', plot_data=plot_data)