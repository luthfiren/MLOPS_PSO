import pandas as pd
import os
import logging
import json
import hashlib
from sklearn.preprocessing import StandardScaler

log_folder = 'processed_data'
os.makedirs(log_folder, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_folder, 'preprocessing.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_csv_files(folder):
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
        return files
    except Exception as e:
        logging.error(f"Failed to list files in {folder}: {e}")
        return []

def detect_and_load_csvs(folder):
    csv_files = get_csv_files(folder)
    datasets = {}

    for file in csv_files:
        try:
            df = pd.read_csv(file, delimiter=";")
            logging.info(f"Loaded {file} → shape: {df.shape}")

            if "endTime" in df.columns:
                datasets["fingrid"] = df
            elif "Time (UTC)" in df.columns:
                datasets["fmi"] = df
            else:
                logging.warning(f"File {file} tidak dikenali strukturnya.")
        except Exception as e:
            logging.error(f"Failed to load {file}: {e}")

    if "fingrid" not in datasets or "fmi" not in datasets:
        raise ValueError("Data Fingrid atau FMI tidak ditemukan secara lengkap.")

    return datasets["fingrid"], datasets["fmi"]

def load_and_merge_data(folder):
    try:
        df_fingrid, df_fmi = detect_and_load_csvs(folder)

        df_fingrid["timestamp"] = pd.to_datetime(df_fingrid["endTime"], errors="coerce").dt.tz_localize(None)
        df_fmi["timestamp"] = pd.to_datetime(df_fmi["Time (UTC)"], errors="coerce").dt.tz_localize(None)

        df = pd.merge(df_fingrid, df_fmi, on="timestamp", how="inner")
        df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
        df.ffill(inplace=True)

        logging.info(f"Merged data → {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except Exception as e:
        logging.error(f"Error in load_and_merge_data: {e}")
        raise

def calculate_file_hash(file_path, algo="sha256"):
    if not os.path.exists(file_path):
        logging.warning(f"File not found for hashing: {file_path}")
        return None

    try:
        hash_func = hashlib.new(algo)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logging.error(f"Error calculating hash for {file_path}: {e}")
        return None

def get_json_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]

def read_json_metadata_from_files(folder, keys=("merged_count", "data_hash")):
    try:
        json_files = get_json_files(folder)
        if not json_files:
            logging.warning(f"No JSON files found in folder: {folder}")
            return None, {}

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    return json_file, {key: data.get(key) for key in keys}
            except Exception as e:
                logging.warning(f"Failed to read {json_file}: {e}")

        logging.warning(f"No valid JSON file found in '{folder}' with keys {keys}")
        return None, {}
    except Exception as e:
        logging.error(f"Error reading JSON metadata from {folder}: {e}")
        return None, {}

def update_json_metadata(json_file, updates: dict):
    try:
        data = {}
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)

        data.update(updates)

        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

        logging.info(f"Updated metadata in {json_file}: {updates}")
    except Exception as e:
        logging.error(f"Error updating JSON metadata {json_file}: {e}")

def check_and_update_metadata(folder, current_count, merged_data, key_count="merged_count", key_hash="data_hash"):
    try:
        json_file, metadata = read_json_metadata_from_files(folder, keys=(key_count, key_hash))

        merged_csv_file_path = os.path.join(folder, "merged_data.csv")
        merged_data.to_csv(merged_csv_file_path, index=False)

        last_file_hash = calculate_file_hash(merged_csv_file_path)

        previous_count = metadata.get(key_count) if metadata else None
        previous_hash = metadata.get(key_hash) if metadata else None

        changed = (previous_count != current_count) or (previous_hash != last_file_hash)

        if json_file is None:
            json_file = os.path.join(folder, f"{key_count}.json")
            logging.info(f"Metadata file not found → Will create: {json_file}")

        if changed or json_file is None:
            update_json_metadata(json_file, {key_count: current_count, key_hash: last_file_hash})
            logging.info(f"Updated metadata at: {json_file}")
        else:
            logging.info("No changes detected → Metadata remains unchanged.")

        return changed

    except Exception as e:
        logging.error(f"Error in check_and_update_metadata: {e}")
        return False

def feature_engineering(df):
    try:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        df["lag_1h_consumption"] = df["value"].shift(1)
        df["rolling_mean_24h"] = df["value"].rolling(window=24).mean()
        df["temp_hour_interaction"] = df["Air temperature"] * df["hour"]

        electricity_price_per_MW = 40
        df["price"] = df["value"] * electricity_price_per_MW

        scaler = StandardScaler()
        numeric_cols = ["Air temperature", "Wind speed", "value", "price"]
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df
    except Exception as e:
        logging.error(f"Error in feature_engineering: {e}")
        raise

# def split_and_save(df, output_dir):
#     try:
#         os.makedirs(output_dir, exist_ok=True)
#         train_size = int(len(df) * 0.7)
#         val_size = int(len(df) * 0.15)

#         train_df = df.iloc[:train_size].reset_index(drop=True)
#         val_df = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
#         test_df = df.iloc[train_size+val_size:].reset_index(drop=True)

#         train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
#         val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
#         test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

#         logging.info("Data successfully split and saved.")
#     except Exception as e:
#         logging.error(f"Error during split_and_save: {e}")

if __name__ == "__main__":
    # Ini inputnya file-file di folder fileExtracted dan menghasilkan file di folder processed_data
    RAW_PATH = "fileExtracted/"
    PROCESSED_PATH = "processed_data/"

    # Concat two dataframe from fileExtracted and do some feature engineering
    df_merged_new = load_and_merge_data(RAW_PATH)
    df_merged_new = feature_engineering(df_merged_new)

    # Inform user on data_aware status such as True if there is changes to the data and False if there is no changes to the data
    new_file_status = check_and_update_metadata(PROCESSED_PATH, df_merged_new.shape[0], merged_data=df_merged_new)
    print("Apakah data-nya ada yang baru? ", new_file_status)
    