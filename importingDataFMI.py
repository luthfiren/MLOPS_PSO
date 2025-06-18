import datetime as dt
import random
import requests
import time
import pandas as pd
import warnings
import os
import json
import logging
from fmiopendata.wfs import download_stored_query

# ─── User-Agent Rotation ───────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (compatible; research-bot/1.0; +https://reserachpurpose1.org/info)",
    "Mozilla/5.0 (compatible; academic-access/2.0; +https://reserachpurpose2.org/bot)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (compatible; polite-data-fetcher/1.0; +https://reserachpurpose3.org/policy)",
    "CustomDataFetcher/1.0 (compatible; +https://reserachpurpose14.org/bot)"
]

# ─── Proxy Pool (Minimum 5) ────────────────────────
PROXIES = [
    "http://proxy1.example.net:8080",
    "http://proxy2.example.net:8080",
    "http://proxy3.example.net:8080",
    "http://proxy4.example.net:8080",
    "http://proxy5.example.net:8080",
]

# ─── Function: Polite Request with Rolling Headers ─
def polite_request(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "DNT": "1",  # Optional: Politely request not to be tracked
            }
            proxy = random.choice(PROXIES)
            print(f"Requesting {url} with {headers['User-Agent']} via {proxy}")

            response = requests.get(url, headers=headers, proxies={"http": proxy, "https": proxy}, params=params, timeout=20)
            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            wait_time = random.uniform(3, 5)
            print(f"Attempt {attempt+1}/{retries} failed: {e}. Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)
    print(f"All {retries} attempts failed for URL: {url}")
    return None

def get_finland_country_level_weather_debug(start_time_str, end_time_str, params_to_fetch=None, bbox_coords="19.0,59.5,31.5,70.1"):
    if params_to_fetch is None:
        params_to_fetch = [
            "Air temperature", "Precipitation amount", "Wind speed", "Pressure (msl)", "Humidity", "Visibility"
        ]

    stored_query_id = "fmi::observations::weather::multipointcoverage"
    all_stations_data = []

    try:
        request_args = [
            f"bbox={bbox_coords}",
            "starttime=" + start_time_str,
            "endtime=" + end_time_str,
            "timestep=60",
            "timeseries=True"
        ]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            obs_data = download_stored_query(stored_query_id, args=request_args)

        if not obs_data.data:
            logging.warning("OBSERVATIONS ARE EMPTY: obs_data.data is empty.")
            return pd.DataFrame()

        for station_id, station_details in obs_data.data.items():
            station_name = station_details.get('name', 'N/A')
            station_lat = station_details.get('lat')
            station_lon = station_details.get('lon')
            times = station_details.get('times', [])

            if not times:
                continue

            for i, time_val in enumerate(times):
                row = {
                    'Station ID': station_id,
                    'Station Name': station_name,
                    'Latitude': station_lat,
                    'Longitude': station_lon,
                    'Time (UTC)': time_val.replace(tzinfo=None)
                }
                for param_key in params_to_fetch:
                    if param_key in station_details and 'values' in station_details[param_key]:
                        if i < len(station_details[param_key]['values']):
                            row[param_key] = station_details[param_key]['values'][i]
                all_stations_data.append(row)

        if not all_stations_data:
            logging.warning("No observations extracted after parsing.")
            return pd.DataFrame()

        df = pd.DataFrame(all_stations_data)
        df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])
        df = df.set_index(['Time (UTC)', 'Station ID']).sort_index()

        actual_params_in_df = [p for p in params_to_fetch if p in df.columns]
        if not actual_params_in_df:
            logging.warning("Requested parameters not found in data for aggregation.")
            return pd.DataFrame()

        country_level_df = df.groupby('Time (UTC)')[actual_params_in_df].mean()

        for param in params_to_fetch:
            if param in df.columns:
                country_level_df[f'{param} (Min)'] = df.groupby('Time (UTC)')[param].min()
                country_level_df[f'{param} (Max)'] = df.groupby('Time (UTC)')[param].max()
                country_level_df[f'{param} (Mean)'] = df.groupby('Time (UTC)')[param].mean()
                country_level_df[f'{param} (Median)'] = df.groupby('Time (UTC)')[param].median()

        logging.info(f"Aggregated data collected from {len(df.index.get_level_values('Station ID').unique())} stations.")
        return country_level_df.reset_index()

    except Exception as e:
        logging.error(f"Error in get_finland_country_level_weather_debug: {e}")
        return pd.DataFrame()

def load_last_ingestion_time(json_file):
    if os.path.isfile(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                last_ingestion_str = data.get("last_ingestion_time")
                if last_ingestion_str:
                    return dt.datetime.fromisoformat(last_ingestion_str.replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    logging.warning("'last_ingestion_time' not found, defaulting to utcnow()")
        except Exception as e:
            logging.warning(f"Error reading JSON '{json_file}': {e}")
    else:
        logging.warning(f"File '{json_file}' not found. Using utcnow().")
    return dt.datetime.utcnow()

def append_new_data(csv_filename, new_data):
    if os.path.isfile(csv_filename):
        existing_df = pd.read_csv(csv_filename, sep=';')
        existing_df['Time (UTC)'] = pd.to_datetime(existing_df['Time (UTC)'])

        new_data = new_data[~new_data['Time (UTC)'].isin(existing_df['Time (UTC)'])]
        if new_data.empty:
            logging.info("No new data to append (all timestamps already exist).")
            return
        new_data.to_csv(csv_filename, mode='a', index=False, header=False, sep=';')
    else:
        new_data.to_csv(csv_filename, index=False, sep=';')
    logging.info(f"Appended {len(new_data)} new rows to '{csv_filename}'.")

def finalize_csv(csv_filename):
    try:
        df = pd.read_csv(csv_filename, sep=';')
        df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])
        df = df.sort_values(by='Time (UTC)', ascending=False).drop_duplicates(subset='Time (UTC)').reset_index(drop=True)
        df.to_csv(csv_filename, index=False, sep=';')
        logging.info(f"Sorted & deduplicated data saved to '{csv_filename}'.")
    except Exception as e:
        logging.error(f"Error finalizing CSV '{csv_filename}': {e}")

if __name__ == "__main__":
    bbox_coordinates = "19.0,59.5,31.5,70.1"
    hours_to_fetch = 24
    total_hours_to_crawl = (24*2)
    output_dir = "fileExtracted"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, "finland_country_level_weather.csv")
    json_filename = os.path.join(output_dir, "lastIngestionTime.json")
    log_file = os.path.join(output_dir, "ingestion.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s,%(msecs)03d %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    end_time_naive_utc = load_last_ingestion_time(json_filename)
    start_time_naive_utc = end_time_naive_utc - dt.timedelta(hours=total_hours_to_crawl)

    logging.info(f"Starting ingestion for file FMI {start_time_naive_utc.isoformat()}Z → {end_time_naive_utc.isoformat()}Z")

    while total_hours_to_crawl > 0:
        hours_this_round = min(hours_to_fetch, total_hours_to_crawl)
        round_start_time = start_time_naive_utc
        round_end_time = round_start_time + dt.timedelta(hours=hours_this_round)

        start_str = round_start_time.isoformat(timespec="seconds") + "Z"
        end_str = round_end_time.isoformat(timespec="seconds") + "Z"

        desired_params = [
            "Air temperature", "Precipitation amount", "Wind speed",
            "Cloud amount", "Relative humidity", "Dew-point temperature"
        ]

        weather_data = get_finland_country_level_weather_debug(start_str, end_str, desired_params, bbox_coordinates)

        if not weather_data.empty:
            try:
                append_new_data(csv_filename, weather_data)
            except Exception as e:
                logging.error(f"❌ Error appending data: {e}")
        else:
            logging.info("⚠ No data retrieved.")

        start_time_naive_utc = round_end_time
        total_hours_to_crawl -= hours_this_round

    finalize_csv(csv_filename)
    logging.info("CSV FMI Data complete: sorted and deduplicated.")