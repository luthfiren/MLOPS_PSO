import sys
import os
import pandas as pd
import tempfile
import pytest
import json
import hashlib
import numpy as np
import importlib.util
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# def test_clean_missing_value():
#     df = pd.DataFrame({"value": [1, None, 3]})
#     cleaned = clean_missing_values(df)
#     print(cleaned.isnull().sum().sum())
#     assert cleaned.isnull().sum().sum() == 0, "Masih ada missing values setelah proses"


# ------------------------------------
# Helper Function
# ------------------------------------

# Helper for creating CSV dummy
def create_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")

# -------------------------------------
# preprocessing file is being inspected
from replacement_of_preprocessingipynb import get_csv_files, detect_and_load_csvs, load_and_merge_data, calculate_file_hash, get_json_files, read_json_metadata_from_files, update_json_metadata, check_and_update_metadata, feature_engineering
# -------------------------------------

def test_get_csv_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.csv")
        file2 = os.path.join(tmpdir, "file2.csv")
        file3 = os.path.join(tmpdir, "file3.txt")
        
        with open(file1, 'w') as f:
            f.write("col1,col2\n1,2")
        with open(file2, 'w') as f:
            f.write("col1,col2\n3,4")
        with open(file3, 'w') as f:
            f.write("not csv content")
        
        result = get_csv_files(tmpdir)
        
        # Assert → pastikan hanya file .csv yang terambil
        assert file1 in result
        assert file2 in result
        assert file3 not in result
        assert len(result) == 2

def test_get_csv_files_invalid_folder():
    assert get_csv_files("folder_yang_tidak_ada") == []
    
def test_detect_and_load_csvs_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_csv(os.path.join(tmpdir, "fingrid.csv"), "endTime;value", ["2025-01-01 00:00;100"])
        create_csv(os.path.join(tmpdir, "fmi.csv"), "Time (UTC);Temperature", ["2025-01-01 00:00;5.2"])
        fingrid, fmi = detect_and_load_csvs(tmpdir)
        assert fingrid.shape == (1, 2)
        assert fmi.shape == (1, 2)
        assert "endTime" in fingrid.columns
        assert "Time (UTC)" in fmi.columns

def test_detect_and_load_csvs_missing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Hanya buat 1 file → simulate incomplete input
        create_csv(
            os.path.join(tmpdir, "only_fingrid.csv"),
            header="endTime;value",
            rows=["2025-01-01 00:00;100"]
        )

        with pytest.raises(ValueError, match="Data Fingrid atau FMI tidak ditemukan secara lengkap."):
            detect_and_load_csvs(tmpdir)
            
def test_load_and_merge_data_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_csv(os.path.join(tmpdir, "fingrid.csv"), "endTime;value", ["2025-01-01 00:00;100"])
        create_csv(os.path.join(tmpdir, "fmi.csv"), "Time (UTC);Temperature", ["2025-01-01 00:00;5.2"])
        merged = load_and_merge_data(tmpdir)
        assert isinstance(merged, pd.DataFrame)
        assert merged.shape[0] == 1
        assert "timestamp" in merged.columns
        assert pd.api.types.is_datetime64_any_dtype(merged['timestamp'])
        assert not merged.isnull().any().any()
        assert not merged.duplicated(subset=["timestamp"]).any()
        
def test_load_and_merge_data_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_csv(os.path.join(tmpdir, "fingrid.csv"), "endTime;value", ["2025-01-01 00:00;100"])
        with pytest.raises(ValueError):
            load_and_merge_data(tmpdir)
            
def test_calculate_file_hash_success():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"test data")
        tmp.flush()
    try:
        assert calculate_file_hash(tmp.name) == hashlib.sha256(b"test data").hexdigest()
    finally:
        os.remove(tmp.name)
        
def test_calculate_file_hash_file_not_found():
    assert calculate_file_hash("no_file.txt") is None
    
def test_calculate_file_hash_invalid_algo():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"data")
        tmp.flush()
    try:
        assert calculate_file_hash(tmp.name, algo="invalid_algo") is None
    finally:
        os.remove(tmp.name)
        
def test_get_json_files_returns_only_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        files = ["a.json", "b.json", "c.txt"]
        [open(os.path.join(tmpdir, f), "w").close() for f in files]
        result = get_json_files(tmpdir)
        assert set(os.path.basename(f) for f in result) == {"a.json", "b.json"}

def test_get_json_files_empty_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert get_json_files(tmpdir) == []
                
def test_read_json_metadata_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.json")
        json.dump({"merged_count": 10, "data_hash": "abc"}, open(path, "w"))
        f, m = read_json_metadata_from_files(tmpdir)
        assert f == path
        assert m == {"merged_count": 10, "data_hash": "abc"}
        
def test_read_json_metadata_missing_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.json")
        json.dump({"merged_count": 5}, open(path, "w"))
        _, m = read_json_metadata_from_files(tmpdir)
        assert m["merged_count"] == 5
        assert m["data_hash"] is None
        
def test_read_json_metadata_invalid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "broken.json")
        with open(path, "w") as f: f.write("{invalid_json")
        fpath, meta = read_json_metadata_from_files(tmpdir)
        assert fpath is None and meta == {}
                
def test_read_json_metadata_no_json_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath, meta = read_json_metadata_from_files(tmpdir)
        assert fpath is None and meta == {}
        
def test_update_json_metadata_create_new():
    with tempfile.NamedTemporaryFile(delete=False) as tmp: tmp_name = tmp.name
    os.remove(tmp_name)
    updates = {"merged_count": 5, "data_hash": "xyz123"}
    update_json_metadata(tmp_name, updates)
    assert json.load(open(tmp_name)) == updates
    os.remove(tmp_name)
    
def test_update_json_metadata_append_to_existing():
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
        json.dump({"existing_key": "value"}, tmp)
    update_json_metadata(tmp.name, {"merged_count": 7, "data_hash": "abc999"})
    data = json.load(open(tmp.name))
    assert data == {"existing_key": "value", "merged_count": 7, "data_hash": "abc999"}
    os.remove(tmp.name)

def test_update_json_metadata_invalid_path():
    try:
        update_json_metadata("/nonexistent/folder/metadata.json", {"key": "value"})
    except Exception:
        assert False, "Should not raise exception"
        
def test_check_and_update_metadata_success():
    with tempfile.TemporaryDirectory() as tmp:
        df = pd.DataFrame({"timestamp": ["2025-01-01 00:00", "2025-01-01 01:00"], "value": [100, 110]})
        changed = check_and_update_metadata(tmp, len(df), df)
        json_file = next(f for f in os.listdir(tmp) if f.endswith(".json"))
        data = json.load(open(os.path.join(tmp, json_file)))
        assert data["merged_count"] == len(df)
        assert data["data_hash"] == calculate_file_hash(os.path.join(tmp, "merged_data.csv"))
        assert changed

def test_check_and_update_metadata_no_change():
    with tempfile.TemporaryDirectory() as tmp:
        df = pd.DataFrame({"timestamp": ["2025-01-01 00:00"], "value": [100]})
        path = os.path.join(tmp, "merged_data.csv"); df.to_csv(path, index=False)
        metadata = {"merged_count": len(df), "data_hash": calculate_file_hash(path)}
        with open(os.path.join(tmp, "merged_count.json"), "w") as f: json.dump(metadata, f)
        assert not check_and_update_metadata(tmp, len(df), df)
        
def test_check_and_update_metadata_invalid_folder():
    assert not check_and_update_metadata("/nonexistent/folder/path", 1, pd.DataFrame())

def test_feature_engineering_success():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=30, freq="h"),
        "value": np.linspace(100, 200, 30),
        "Air temperature": np.linspace(-5, 5, 30),
        "Wind speed": np.linspace(0, 10, 30),
    })
    result = feature_engineering(df.copy())
    expected = ["hour", "day_of_week", "month", "is_weekend", "lag_1h_consumption",
                "rolling_mean_24h", "temp_hour_interaction", "price"]
    assert all(col in result.columns for col in expected)
    assert result.shape[0] == 30 and result["rolling_mean_24h"].isnull().sum() == 23
    for col in ["Air temperature", "Wind speed", "value", "price"]:
        assert np.isclose(result[col].dropna().mean(), 0, atol=1e-2)
        assert np.isclose(result[col].dropna().std(), 1, atol=0.05)

def test_feature_engineering_invalid_input():
    try:
        feature_engineering(pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=10, freq="h")}))
    except Exception as e:
        assert isinstance(e, Exception)
                
# -------------------------------------
# importingDataFinGrid file is being inspected
import importingDataFinGrid as ff
# -------------------------------------

def test_load_and_save_last_success_time():
    with tempfile.TemporaryDirectory() as tmpdir:
        ff.stateFile = os.path.join(tmpdir, "state.json")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        ff.save_current_time_as_last_success(now.isoformat())
        assert ff.load_last_success_time() == now

def test_load_last_success_time_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        ff.stateFile = os.path.join(tmpdir, "nonexistent.json")
        result = ff.load_last_success_time()
        assert isinstance(result, datetime)
        
def test_fetch_data_success(monkeypatch):
    class MockResponse:
        status_code = 200
        def json(self): return {"result": "ok"}

    monkeypatch.setattr(ff.requests, "get", lambda *a, **kw: MockResponse())
    ff.url, ff.headerCall, ff.parameter, ff.maxRetries = "url", {}, {}, 1
    assert ff.fetch_data() == {"result": "ok"}

def test_fetch_data_fail(monkeypatch):
    monkeypatch.setattr(ff.requests, "get", lambda *a, **kw: type("X", (), {"status_code": 500})())
    ff.url, ff.headerCall, ff.parameter, ff.maxRetries = "url", {}, {}, 1
    with pytest.raises(Exception):
        ff.fetch_data()

def test_fetch_data_success(monkeypatch):
    class MockResponse:
        status_code = 200
        def json(self): return {"result": "ok"}

    monkeypatch.setattr(ff.requests, "get", lambda *a, **kw: MockResponse())
    ff.url, ff.headerCall, ff.parameter, ff.maxRetries = "url", {}, {}, 1
    assert ff.fetch_data() == {"result": "ok"}

def test_fetch_data_fail(monkeypatch):
    monkeypatch.setattr(ff.requests, "get", lambda *a, **kw: type("X", (), {"status_code": 500})())
    monkeypatch.setattr(ff, "sleep", lambda x: None)  # <-- NO WAIT
    ff.url, ff.headerCall, ff.parameter, ff.maxRetries = "url", {}, {}, 1
    with pytest.raises(Exception):
        ff.fetch_data()
        
# -------------------------------------
# importingDataFinGrid file is being inspected
from importingDataFMI import load_last_ingestion_time, append_new_data, finalize_csv
# -------------------------------------

def test_load_last_ingestion_time_with_valid_json():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        json.dump({"last_ingestion_time": "2025-06-01T12:00:00Z"}, tmpfile)
        tmpfile_name = tmpfile.name

    result = load_last_ingestion_time(tmpfile_name)
    assert isinstance(result, datetime)
    assert result == datetime(2025, 6, 1, 12, 0)

    os.remove(tmpfile_name)


def test_load_last_ingestion_time_with_invalid_json():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        tmpfile.write("{invalid_json}")
        tmpfile_name = tmpfile.name

    result = load_last_ingestion_time(tmpfile_name)
    assert isinstance(result, datetime)  # fallback to utcnow()

    os.remove(tmpfile_name)


def test_append_new_data_creates_file():
    df = pd.DataFrame({
        'Time (UTC)': pd.date_range("2025-01-01", periods=3, freq='h'),
        'Air temperature': [1, 2, 3]
    })

    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile_name = tmpfile.name

    os.remove(tmpfile_name)  # Simulate non-existent file


    append_new_data(tmpfile_name, df)
    assert os.path.isfile(tmpfile_name)

    loaded = pd.read_csv(tmpfile_name, sep=';')
    assert loaded.shape[0] == 3

    os.remove(tmpfile_name)


def test_finalize_csv_removes_duplicates_and_sorts():
    data = pd.DataFrame({
        'Time (UTC)': ["2025-01-01 03:00", "2025-01-01 02:00", "2025-01-01 02:00"],
        'Air temperature': [3, 2, 2]
    })

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        data.to_csv(tmpfile.name, index=False, sep=';')
        tmpfile_name = tmpfile.name

    finalize_csv(tmpfile_name)
    df_final = pd.read_csv(tmpfile_name, sep=';')

    assert df_final.shape[0] == 2
    assert pd.to_datetime(df_final['Time (UTC)']).is_monotonic_decreasing

    os.remove(tmpfile_name)
    
## -------------------------------------
# test model:
from model import ALL_MODEL_CLASSES
# ------------------------------------- 
@pytest.fixture
def dummy_timeseries_df():
    n = 100
    return pd.DataFrame({
        "unique_id": ["ts1"] * n,
        "ds": pd.date_range("2021-01-01", periods=n, freq="H"),
        "y": np.random.normal(100, 10, size=n)
    })

@pytest.fixture
def dummy_timeseries_df():
    # Jumlah data cukup panjang agar fold training cukup untuk seasonal model (misal seasonal_periods=12)
    n = 200
    return pd.DataFrame({
        "unique_id": ["ts1"] * n,
        "ds": pd.date_range("2021-01-01", periods=n, freq="h"),
        "y": np.random.normal(100, 10, size=n)
    })

@pytest.mark.parametrize("ModelCls", list(ALL_MODEL_CLASSES.values()))
def test_model_lifecycle(ModelCls, dummy_timeseries_df, tmp_path):
    model = ModelCls()

    folds = model.create_folds(dummy_timeseries_df, n_splits=3, test_size=24)
    assert len(folds) > 0

    avg_score = model.train_with_fold(folds)
    assert avg_score >= 0

    last_train, last_test = folds[-1]
    pred_df = last_test.copy()
    pred = model.predict(pred_df, h=len(pred_df))
    assert isinstance(pred, pd.DataFrame)

    # Map kolom prediksi apapun ke yhat jika belum ada
    if "yhat" not in pred.columns:
        pred_cols = [c for c in pred.columns if c not in ["ds", "unique_id", "y"]]
        if pred_cols:
            pred = pred.rename(columns={pred_cols[0]: "yhat"})
    # Drop kolom y dari pred supaya merge tidak kena suffix y_x/y_y
    if "y" in pred.columns:
        pred = pred.drop(columns=["y"])

    # Evaluasi (fungsi evaluate di model harus pakai merged['y'] dan merged['yhat'])
    score = model.evaluate(last_test, pred)
    assert score >= 0

    # Save model ke temporary path pytest
    save_path = tmp_path / "model_champion.joblib"
    try:
        model.save(str(save_path))
    except TypeError:
        model.save(model_path=str(save_path))
    assert save_path.exists()
