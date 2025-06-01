import requests, csv, io, os, logging, json
from datetime import datetime, timezone, timedelta
from time import sleep

# == configuration ==
datasetId = 329
apiKey = "0de19d67f7fa4e7bb177e5c163b3c802"
pageSize = 5
maxRetries = 3

# == Folder and File Setup ==
logFolder = "MLOpsAutomation/logs"
os.makedirs(logFolder, exist_ok=True)
stateFile = os.path.join(logFolder, "lastIngestionTime.json")
logPath = os.path.join(logFolder, "ingestion.log")
savingPath = os.path.join(logFolder, "extractionDataset.csv")

# == Setup Logging ==
logging.basicConfig(filename=logPath, level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# == Track State File ==
def load_last_success_time():
    if os.path.exists(stateFile):
        with open(stateFile, "r") as f:
            return datetime.fromisoformat(json.load(f)["last_ingestion_time"])
    else:
        # Default: yesterday 00:00 UTC
        return (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

def save_current_time_as_last_success(rfc3339_str):
    with open(stateFile, "w") as f:
        json.dump({"last_ingestion_time": rfc3339_str}, f)

# == Incremental Load ==
startTime = load_last_success_time()
endTime = datetime.now(timezone.utc).replace(microsecond=0)

startTimeExtraction = startTime.isoformat().replace("+00:00", "Z")
endTimeExtraction = endTime.isoformat().replace("+00:00", "Z")

parameter = {
    "startTime": startTimeExtraction,
    "endTime": endTimeExtraction,
    "format": "csv",
    "pageSize": pageSize, # Change this to number of rows you want to save (Production mode != development mode)
    "sortOrder": "asc"
}

url = f"https://data.fingrid.fi/api/datasets/{datasetId}/data"
headerCall = {
    "x-api-key": apiKey,
    "Cache-Control": "no-cache"
}

def fetch_data():
    for attempt in range(1, maxRetries+1):
        try:
            response = requests.get(url, headers=headerCall, params=parameter)
            if response.status_code == 200:
                return response.json()
            else:
                logging.warning(f"Attempt {attempt} failed with status {response.status_code}")
        except Exception as e:
            logging.error(f"Attempt {attempt} failed with error: {str(e)}")
        sleep(2 ** attempt)
    raise Exception("All retries failed for data fetch")

try:
    logging.info(f"Starting ingestion for {startTimeExtraction} → {endTimeExtraction}")
    data = fetch_data()
    csv_content = data["data"]
    
    new_rows = list(csv.DictReader(io.StringIO(csv_content), delimiter=';'))
    if not new_rows:
        logging.warning("No new data to ingest")
    else:
        file_exists = os.path.isfile(savingPath)
        fieldNames = list(new_rows[0].keys())
        write_header = not file_exists

        # Optional: Schema enforcement
        if file_exists:
            with open(savingPath, "r", encoding="utf-8") as f:
                existing_headers = next(csv.reader(f, delimiter=';'))
                assert existing_headers == list(new_rows[0].keys()), "Schema mismatch!"
        
        mode = "a" if file_exists else "w"
        
        with open(savingPath, mode, newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldNames, delimiter=';')
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
        
        save_current_time_as_last_success(endTimeExtraction)
        logging.info(f"Ingestion complete. Rows written: {data['pagination']['to']}")
except AssertionError as schema_error:
    logging.error(f"Schema validation failed: {schema_error}")
except Exception as e:
    logging.error(f"Critical failure: {e}")