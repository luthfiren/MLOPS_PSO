# app.py
import joblib
import os
from flask import Flask, render_template, send_from_directory # Tambahkan render_template dan send_from_directory

app = Flask(__name__, 
            static_folder='static', # Mengatur folder statis untuk Flask
            template_folder='templates') # Mengatur folder template untuk Flask

# Path ke model yang sudah dideploy
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'ThetaModel_champion.joblib') 

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    # Render template HTML dan berikan path gambar plot
    # Pastikan 'forecast_plot.png' ada di folder 'static' yang dideploy
    return render_template('index.html', forecast_image_url='/static/forecast_plot.png')

# Anda juga bisa membuat endpoint untuk menyajikan file statis secara langsung
# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
def predict():
    # ... (Logika prediksi seperti sebelumnya) ...
    return jsonify({"forecast": "Ini adalah prediksi contoh"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))