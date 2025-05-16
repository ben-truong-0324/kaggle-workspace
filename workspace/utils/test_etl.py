import pandas as pd
from etl import CSVSource, validate_and_transform, detect_data_drift, EXPECTED_SCHEMA


# === Unit Tests ===
def test_csv_source_reads_chunks():
    source = CSVSource("tests/sample.csv", chunk_size=5)
    chunks = list(source.get_batches())
    assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks), "All chunks must be DataFrames"
    assert sum(len(chunk) for chunk in chunks) > 0, "Total rows must be > 0"


def test_validate_and_transform_valid():
    data = pd.DataFrame([{k: v() for k, v in EXPECTED_SCHEMA.items()} for _ in range(5)])
    valid, errors = validate_and_transform(data)
    assert len(valid) == 5, "Should return all valid rows"
    assert len(errors) == 0, "Should return no errors"


def test_validate_and_transform_invalid():
    data = pd.DataFrame([{k: None for k in EXPECTED_SCHEMA.keys()}])
    valid, errors = validate_and_transform(data)
    assert len(valid) == 0, "Should return 0 valid rows"
    assert len(errors) == 1, "Should catch one row with errors"


def test_drift_detection():
    df1 = pd.DataFrame({"a": [0.1, 0.2, 0.3]})
    df2 = pd.DataFrame({"a": [0.9, 0.95, 1.0]})
    drift = detect_data_drift(df1, df2)
    assert "a" in drift, "Should detect drift due to mean shift"
