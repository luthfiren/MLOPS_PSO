# Use an official Python runtime as the base image.
FROM python:3.11-slim-buster

# Set environment variables for non-interactive operations and Python unbuffered output.
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements.txt file and install dependencies.
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# --- Perubahan di sini: Gunakan satu COPY . . untuk menyalin semua file proyek ---
# Salin seluruh isi direktori proyek Anda ke dalam direktori /app di kontainer.
# Ini termasuk semua script ML, app.py, dan folder seperti 'data/', 'model/', 'processed_data/', dll.
COPY . .

# Jalankan script data ingestion, preprocessing, dan training model SELAMA PROSES DOCKER BUILD.
# Pastikan semua output yang dibutuhkan oleh app.py (misalnya model terlatih)
# disimpan ke lokasi yang dapat diakses oleh app.py di dalam kontainer (misal: ./model/).
RUN echo "Running data ingestion during Docker build..." && \
    python importingDataFinGrid.py && \
    python importingDataFMI.py && \
    echo "Data ingestion complete." && \
    \
    echo "Running data preprocessing using preprocessing.ipynb during Docker build..." && \
    # Pastikan jupyter dan nbconvert terinstal dari requirement.txt
    jupyter nbconvert --to notebook --execute --inplace preprocessing.ipynb && \
    echo "Data preprocessing complete." && \
    \
    echo "Running ML model training and evaluation during Docker build..." && \
    python modelling.py && \
    echo "ML model training and evaluation complete."

# Expose the port that your application will listen on.
EXPOSE 8000

# Define the command that will be executed when the container starts.
# Ini akan menjalankan app.py, yang diharapkan akan memuat model
# yang sudah terlatih dari lokasi yang sudah ditetapkan (misal: ./model/trained_model.pkl).
CMD python ./app.py