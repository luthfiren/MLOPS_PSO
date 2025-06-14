import datetime as dt
from fmiopendata.wfs import download_stored_query
import pandas as pd
import warnings
import os
import json
import logging

# The function definition (get_finland_country_level_weather_debug) remains the same as before.
# We will just change how it's called in the main block.
def get_finland_country_level_weather_debug(start_time_str, end_time_str, params_to_fetch=None, bbox_coords="19.0,59.5,31.5,70.1"):
    """
    Downloads weather observations from FMI for multiple stations across Finland
    and aggregates them to provide a country-level summary.
    This version now takes already formatted time strings.

    Args:
        start_time_str (str): The start datetime for the observations in ISO format (e.g., "YYYY-MM-DDTHH:MM:SSZ").
        end_time_str (str): The end datetime for the observations in ISO format (e.g., "YYYY-MM-DDTHH:MM:SSZ").
        params_to_fetch (list, optional): List of parameters to fetch (e.g., ["Air temperature", "Wind speed"]).
                                         If None, fetches common parameters.
        bbox_coords (str): Bounding box coordinates in 'lon_min,lat_min,lon_max,lat_lat' format.

    Returns:
        pandas.DataFrame: A DataFrame with aggregated country-level weather data.
                          Returns an empty DataFrame if no data is found.
    """
    if params_to_fetch is None:
        params_to_fetch = [
            "Air temperature",
            "Precipitation amount",
            "Wind speed",
            "Pressure (msl)",
            "Humidity",
            "Visibility"
        ]

    stored_query_id = "fmi::observations::weather::multipointcoverage"
    all_stations_data = []
    
    try:
        # Construct the args as a list of strings, using the pre-formatted time strings
        request_args = [
            f"bbox={bbox_coords}",
            "starttime=" + start_time_str, # Use the pre-formatted string directly
            "endtime=" + end_time_str,     # Use the pre-formatted string directly
            "timestep=60", # 60 minutes for observations
            "timeseries=True" # Forces station_name/ID as top-level key
        ]
        
        # Suppress the specific UserWarning regarding invalid bbox if it still appears
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning) 
            obs_data = download_stored_query(stored_query_id, args=request_args)

        if not obs_data.data:
            logging.warning("OBSERVATIONS ARE EMPTY: obs_data.data is empty.")
            return pd.DataFrame()

        # --- Adjusted Data Extraction ---
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

        # Aggregate data
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

# # --- Main execution block ---
# if __name__ == "__main__":
#     bbox_coordinates = "19.0,59.5,31.5,70.1" # Finland mainland coordinates
    
#     # Try a very wide time range, e.g., last 7 days, using the UTCNow approach
# #     hours_to_fetch = 25200 # Fetch last 7 days of data
#     hours_to_fetch = 2 # Fetch last 7 days of data

#     # CRITICAL: Generate naive UTC datetime, then format to string with 'T' and 'Z'
#     end_time_naive_utc = dt.datetime.utcnow() - dt.timedelta(hours=1)
#     start_time_naive_utc = end_time_naive_utc - dt.timedelta(hours=hours_to_fetch)

#     # Format them into strings with 'T' and 'Z' suffix before passing
#     start_time_str = start_time_naive_utc.isoformat(timespec="seconds") + "Z"
#     end_time_str = end_time_naive_utc.isoformat(timespec="seconds") + "Z"
        
#     print(start_time_str, end_time_str)

# #     # --- Test Case 1: Minimal Parameters ---
# #     print("\n=== Running Test Case 1: Minimal Parameters (Air Temperature) ===")
# #     minimal_params = ["Air temperature"]
# #     finland_weather_df_minimal = get_finland_country_level_weather_debug(
# #         start_time_str, end_time_str, minimal_params, bbox_coordinates
# #     )

# #     if not finland_weather_df_minimal.empty:
# #         print("\n--- TEST CASE 1 SUCCESS: Minimal Parameters Weather Summary ---")
# #         print(finland_weather_df_minimal.head())
# #         print(f"\nNumber of time points: {len(finland_weather_df_minimal)}")
# #         print(f"Aggregated parameters: {finland_weather_df_minimal.columns.tolist()}")
# #     else:
# #         print("\n--- TEST CASE 1 FAILED: No data with minimal parameters. ---")

# #     # --- Test Case 2: All Desired Parameters (if Test Case 1 succeeds) ---
# #     if not finland_weather_df_minimal.empty: # Only run if minimal worked
# #         print("\n\n=== Running Test Case 2: All Desired Parameters ===")
# #         desired_params_full = [
# #             "Air temperature",
# #             "Precipitation amount",
# #             "Wind speed",
# #             "Pressure (msl)",
# #             "Humidity",
# #             "Visibility"
# #         ]
# #         finland_weather_df_full = get_finland_country_level_weather_debug(
# #             start_time_str, end_time_str, desired_params_full, bbox_coordinates
# #         )

# #         if not finland_weather_df_full.empty:
# #             print("\n--- TEST CASE 2 SUCCESS: Full Parameters Weather Summary ---")
# #             print(finland_weather_df_full.head())
# #             print(f"\nNumber of time points: {len(finland_weather_df_full)}")
# #             print(f"Aggregated parameters: {finland_weather_df_full.columns.tolist()}")
# #         else:
# #             print("\n--- TEST CASE 2 FAILED: No data with full parameters. ---")
# #     else:
# #         print("\nSkipping Test Case 2 as Test Case 1 failed.")

#     # Define the parameters you want to fetch
#     desired_params = [
#         "Air temperature",
#         "Precipitation amount",
#         "Wind speed",
#         "Cloud amount",
#         "Relative humidity",
#         "Dew-point temperature",
#     ]

#     # Call your function to get the aggregated weather data
#     finland_weather_data = get_finland_country_level_weather_debug(
#         start_time_str, end_time_str, desired_params, bbox_coordinates
#     )

#     if not finland_weather_data.empty:
#         print("\n--- Weather Data Successfully Retrieved ---")
        
#         # 1. Get the data (the DataFrame itself)
#         #    'finland_weather_data' is already your DataFrame.
#         print("First 5 rows of the DataFrame:")
#         print(finland_weather_data.head())
        
#         # 2. Get the columns
#         print("\nColumns in the DataFrame:")
#         print(finland_weather_data.columns.tolist())
        
#         # 3. Save the data to a CSV file
#         csv_filename = "finland_country_level_weather.csv"
#         try:
#             finland_weather_data.to_csv(csv_filename)
#             print(f"\nWeather data saved successfully to '{csv_filename}'")
#         except Exception as e:
#             print(f"Error saving data to CSV: {e}")
#     else:
#         print("\n--- No weather data was retrieved. CSV not created. ---")

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
      
    # ── Config ─────────────────────────────────────────────
    bbox_coordinates = "19.0,59.5,31.5,70.1"
    hours_to_fetch = 168
    total_hours_to_crawl = (24*60)  # 3 years
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

    # ── Set Time Range ─────────────────────────────────────
    end_time_naive_utc = load_last_ingestion_time(json_filename)
    start_time_naive_utc = end_time_naive_utc - dt.timedelta(hours=total_hours_to_crawl)

    logging.info(f"Starting ingestion for file FMI {start_time_naive_utc.isoformat()}Z → {end_time_naive_utc.isoformat()}Z")
    
    # ── Main Loop ──────────────────────────────────────────    
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

        weather_data = get_finland_country_level_weather_debug(
            start_str, end_str, desired_params, bbox_coordinates
        )

        if not weather_data.empty:
            try:
                append_new_data(csv_filename, weather_data)
            except Exception as e:
                logging.error(f"❌ Error appending data: {e}")
        else:
            logging.info("⚠ No data retrieved.")

        # Progress update
        start_time_naive_utc = round_end_time
        total_hours_to_crawl -= hours_this_round

    # ── Finalization ───────────────────────────────────────
    finalize_csv(csv_filename)
    logging.info("CSV FMI Data complete: sorted and deduplicated.")