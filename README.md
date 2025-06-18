<h1>[G1 FP] MLOps : Forecasting Electricity Market</h1>
Proyek ini merupakan bagian dari Mata Kuliah PSO dan dibuat oleh :

1. Alka Sidik Prawira
2. Luthfi Rihadatul Fajri
3. Razi Alvaro Arman
4. Yeremia Maydinata Narana

<h2>Introduction</h2>
Selamat datang di proyek MLOps end-to-end ini! Proyek ini mendemonstrasikan alur lengkap Machine Learning Operations (MLOps) mulai dari data ingestion, modeling, hingga deployment web menggunakan tumpukan teknologi Python dan diotomatisasi dengan GitHub Actions untuk Continuous Integration/Continuous Deployment (CI/CD). Hasil akhirnya adalah aplikasi web berbasis Python yang siap di-deploy di Azure.

<h2>🗄️# RUN LOCALLY</h2>
<h2>Getting Started</h2>

1. Kloning Repositori
   ```
   git clone https://github.com/luthfiren/MLOPS_PSO.git
   ```

<h2>Depedency Instalation</h2>

1. Buat venv dalam kasus saya namanya "mlenv"
   ```
   python -m venv mlenv
   ```
2. Aktifkan venv
   ```
   mlenv/Scripts/activate
   ```
3. Install Depedensi
   ```
   pip install -r requirement.txt
   ```
<h2>Data Ingetion</h2>

1. Import Data FinGrid
   Khusus untuk Data FinGrid, lakukan ini bila API dari pusat data masih tersedia (karena terbatas)
   ```
   python importingDataFinGrid.py
   ```
2. Import Data FMI
   ```
   python importingDataFMI.py
   ```

<h2>Data PreProcessing</h2>

1. Jalankan file processing data
   ```
   python replacement_of_preprocessingipynb.py
   ```
   Pre-processing data terbilang berhasil ketika file ```merged_data.csv```, ```merged_count.json```, ```preprocessing.log``` terbentuk dalam folder ```processed_data```

<h2>Run Model Locally</h2>

1. Jalankan Host mlflow
   ```
   mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
   ```
     Output yang diharapkan adalah muncul host dan port dari mlflow.
     ```
     INFO:waitress:Serving on http://0.0.0.0:5001
     ```
2. Jalankan modelling dan mencari model terbaik
   File ```modelling.py``` berisi orkestrasi file yang menyambung model-model di folder model dan dicari model mana yang terbaik dengan parameternya masing-masing.
   ```
   python modelling.py
   ```
   Setelah itu, anda harus menunggu beberapa waktu hingga muncul:
   ```
   MLOps Pipeline selesai. Data untuk dashboard telah diperbarui.
   Pipeline timing saved to: artifacts/pipeline_timings.json
   🏃 View run Full_MLOps_Pipeline_Run at: http://localhost:5001/#/experiments/0/runs/cb7cdc9c4add4bef8b7d1554c60bc74c
   🧪 View experiment at: http://localhost:5001/#/experiments/0
   ```
   yang menandakan bahwa seleksi model telah selesai dan anda bisa memonitor hasil train melalui ```http://localhost:5001/#/experiments/0/runs/cb7cdc9c4add4bef8b7d1554c60bc74c```. Selain itu, output dari             ```modelling.py``` selain daripada ui dari mlflow adalah sheets ```latest_forecast.csv``` yang berisi hasil forecasting dari data praprocessing.

<h2>Run Web Locally</h2>

1. Jalankan server flask
   ```
   python app.py
   ```
3. Akses Dashboard

<h2>🤖# RUN WITH PIPELINE</h2>

1. Input Data
   Data diinput bisa melalui data ingestion bisa dengan meng-input secara mandiri ke file ```test.csv```, ```train.csv```, dan ```val.csv```. Lalu setelah data terdeteksi data baru (berdasarkan data terakhir) maka pipeline akan berjalan dengan sendirinya mulai dari pre-processing, modelling, dan deployment.
   
3. Monitor Proses Pipeline
   Anda bisa melihat proses pipeline di ```https://github.com/luthfiren/MLOPS_PSO.git``` lalu tab ```Actions```.
   
5. Akses Dashboard
   Masih di tab ```Actions```, jika proses pipeline sudah selesai maka akan muncul link yang nantinya anda akan diarahkan ke dashboard berbasis web.

<h2>Note</h2>

1. Error: Jika Anda menemui error, pastikan semua dependensi terinstal dengan benar ```pip install -r requirements.txt```. Periksa juga log di konsol Anda untuk pesan error spesifik dari Python atau Flask

2. Perubahan Kode: Setiap kali Anda melakukan perubahan pada modelling.py (untuk melatih model baru) atau app.py (untuk     mengubah tampilan dashboard), Anda perlu menghentikan server Flask (Ctrl+C) dan menjalankannya kembali ```python app.py``` untuk melihat perubahan tersebut.

