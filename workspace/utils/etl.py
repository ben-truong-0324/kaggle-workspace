import json
from pathlib import Path
import pandas as pd
import shutil
import inspect

def run_default_etl(dataset_name: str, target_column: str, ) -> dict:

    etl_description = "Default ETL logic"

    base_data_dir = Path("/home/jovyan/data") #default jupyter jovyan
    dataset_specific_base_path = base_data_dir / dataset_name
    raw_data_target_dir = dataset_specific_base_path / "raw"
    processed_data_dir = dataset_specific_base_path / "processed"
    store_path = dataset_specific_base_path / "etl_generated_store.json"

    # Ensure processed folder exists
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Load existing store
    if store_path.exists():
        with open(store_path, "r") as f:
            store = json.load(f)
    else:
        store = {}

    # Generate next UUID (v1, v2, etc.)
    existing_versions = [int(k[1:]) for k in store.keys() if k.startswith("v") and k[1:].isdigit()]
    next_version_number = max(existing_versions, default=0) + 1
    uuid = f"v{next_version_number}"

    # Load raw data
    csv_files = list(raw_data_target_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in raw data directory.")
    print(csv_files)
    df = pd.read_csv(csv_files[0])
    print("Dataset loaded successfully. Only one file loaded")
    print("Dataset shape:", df.shape)
    print("Dataset columns:", df.columns.tolist())
    print("Dataset head:\n", df.head())

    # Verify target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Basic example split
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    etl_source = inspect.getsource(run_default_etl)

    store[uuid] = {
        "etl_description": etl_description,
        "target_column": target_column,
        "etl_source_code": etl_source
    }
    


    with open(store_path, "w") as f:
        json.dump(store, f, indent=4)

    return {
        "etl_version": uuid,
        "etl_description": etl_description,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val
    }


def get_raw_dataset(dataset_name, target_column):
    base_data_dir = Path("/home/jovyan/data") #default jupyter jovyan
    dataset_specific_base_path = base_data_dir / dataset_name
    raw_data_target_dir = dataset_specific_base_path / "raw"
    processed_data_dir = dataset_specific_base_path / "processed"
    store_path = dataset_specific_base_path / "etl_generated_store.json"
    # Ensure processed folder exists
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Load existing store
    if store_path.exists():
        with open(store_path, "r") as f:
            store = json.load(f)
    else:
        store = {}

    # Load raw data
    csv_files = list(raw_data_target_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in raw data directory.")
    print(csv_files)
    df = pd.read_csv(csv_files[0])
    print("Dataset loaded successfully. Only one file loaded")
    print("Dataset shape:", df.shape)
    print("Dataset columns:", df.columns.tolist())
    print("Dataset head:\n", df.head())

    # Verify target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    return X, y


def run_custom_etl(dataset_name: str, target_column: str, ) -> dict:
    X, y = get_raw_dataset(dataset_name, target_column)

    from sklearn.preprocessing import MinMaxScaler
    # === Normalize features to 0–1 range ===
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)


    # Basic example split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    etl_source = inspect.getsource(run_custom_etl)
    # Generate next UUID (v1, v2, etc.)
    existing_versions = [int(k[1:]) for k in store.keys() if k.startswith("v") and k[1:].isdigit()]
    next_version_number = max(existing_versions, default=0) + 1
    uuid = f"v{next_version_number}"
    etl_description = "Custom ETL logic: normalize to 0-1 range for all features with sklearn.preprocessing.MinMaxScaler"
    store[uuid] = {
        "etl_description": etl_description,
        "target_column": target_column,
        "etl_source_code": etl_source
    }
    
    with open(store_path, "w") as f:
        json.dump(store, f, indent=4)

    return {
        "etl_version": uuid,
        "etl_description": etl_description,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val
    }