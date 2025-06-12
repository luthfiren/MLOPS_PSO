<h1>[G1 FP] MLOps : Forecasting Electricity Market</h1>
Proyek ini merupakan bagian dari Mata Kuliah PSO dan dibuat oleh :

1. Alka Sidik Prawira
2. Luthfi Rihadatul Fajri
3. Razi Alvaro Arman
4. Yeremia Maydinata Narana

<h2>Introduction</h2>
Selamat datang di proyek MLOps end-to-end ini! Proyek ini mendemonstrasikan alur lengkap Machine Learning Operations (MLOps) mulai dari data ingestion, modeling, hingga deployment web menggunakan tumpukan teknologi Python dan diotomatisasi dengan GitHub Actions untuk Continuous Integration/Continuous Deployment (CI/CD). Hasil akhirnya adalah aplikasi web berbasis Python yang siap di-deploy di Azure.

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

<h2>Run Web Locally</h2>

1. Jalankan Server Flask
   ```
   python app.py
   ```
     Output yang diharapkan adalah muncul server pengembangan lokal.
     ```
     Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
     ```
2. Akses Dashboard Web
   Klik link yang muncul dan arahkan ke browser, maka nanti akan melihat dashboard MLOps.

<h2>Note</h2>

1. Error: Jika Anda menemui error, pastikan semua dependensi terinstal dengan benar ```pip install -r requirements.txt```. Periksa juga log di konsol Anda untuk pesan error spesifik dari Python atau Flask

2. Perubahan Kode: Setiap kali Anda melakukan perubahan pada modelling.py (untuk melatih model baru) atau app.py (untuk     mengubah tampilan dashboard), Anda perlu menghentikan server Flask (Ctrl+C) dan menjalankannya kembali ```python app.py``` untuk melihat perubahan tersebut.

