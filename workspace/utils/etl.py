import json
from pathlib import Path
import pandas as pd
import shutil
import inspect
from sklearn.cluster import KMeans


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

# to simulate steraming
# def run_streaming_style_etl(source: DataSource, target_column: str):
#     error_log = []
#     df_all = []

#     for chunk in source.get_batches():
#         valid_chunk, errors = validate_and_transform(chunk)
#         error_log.extend(errors)
#         if not valid_chunk.empty:
#             df_all.append(valid_chunk)

#     if error_log:
#         with open(ERROR_LOG_PATH, "w") as f:
#             json.dump(error_log, f, indent=4)

#     full_df = pd.concat(df_all, ignore_index=True)
#     full_df = full_df.sort_values("alcohol")  # simulate time-based sorting

#     X = full_df.drop(columns=[target_column])
#     y = full_df[target_column]
def run_custom_etl(dataset_name: str, target_column: str, ) -> dict:
    X, y = get_raw_dataset(dataset_name, target_column)

    # === Create composite features ===
    X["acid_index"] = (
        X["fixed acidity"] + X["volatile acidity"] + X["citric acid"]
    )
    X["sugar_density_ratio"] = X["residual sugar"] / X["density"]
    X["sulfur_ratio"] = X["free sulfur dioxide"] / X["total sulfur dioxide"].replace(0, 1e-3)
    X["alcohol_pH_ratio"] = X["alcohol"] / X["pH"].replace(0, 1e-3)
    X["sulphate_acid_ratio"] = X["sulphates"] / (X["volatile acidity"] + 1e-3)

   
    from sklearn.preprocessing import MinMaxScaler
    # === Normalize features to 0–1 range ===
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # === Clustering ===
    kmeans = KMeans(n_clusters=6, random_state=42)
    X_scaled_df['kmeans_cluster'] = kmeans.fit_predict(X_scaled_df)

    # # === PCA ===
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=7)
    # X_pca = pca.fit_transform(X_scaled)
    # for i in range(7):
    #     X_scaled_df[f"pca_{i+1}"] = X_pca[:, i]
    # X = X_scaled_df

    # # === Isomap ===
    # from sklearn.manifold import Isomap
    # isomap = Isomap(n_components=7)
    # X_iso = isomap.fit_transform(X_scaled)
    # for i in range(X_iso.shape[1]):
    #     X_scaled_df[f"isomap_{i+1}"] = X_iso[:, i]
    # X = X_scaled_df

    # # === UMAP ===
    # import umap
    # reducer = umap.UMAP(n_components=7, random_state=42)
    # X_umap = reducer.fit_transform(X_scaled)
    # for i in range(X_umap.shape[1]):
    #     X_scaled_df[f"umap_{i+1}"] = X_umap[:, i]
    # X = X_scaled_df

    # === ICA ===
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=7, random_state=42)
    X_ica = ica.fit_transform(X_scaled)
    for i in range(X_ica.shape[1]):
        X_scaled_df[f"ica_{i+1}"] = X_ica[:, i]
    X = X_scaled_df


    # Basic example split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    etl_source = inspect.getsource(run_custom_etl)
    # Generate next UUID (v1, v2, etc.)
    base_data_dir = Path("/home/jovyan/data") #default jupyter jovyan
    dataset_specific_base_path = base_data_dir / dataset_name
    store_path = dataset_specific_base_path / "etl_generated_store.json"
    # Load existing store
    if store_path.exists():
        with open(store_path, "r") as f:
            store = json.load(f)
    else:
        store = {}

    existing_versions = [int(k[1:]) for k in store.keys() if k.startswith("v") and k[1:].isdigit()]
    next_version_number = max(existing_versions, default=0) + 1
    uuid = f"v{next_version_number}"
    etl_description = "composite features, MinMaxScaler 0-1, kmeans, ICA"
    
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



def detect_data_drift(X_train, X_val):
    drift_flags = {}
    for col in X_train.columns:
        train_mean = X_train[col].mean()
        val_mean = X_val[col].mean()
        delta = abs(train_mean - val_mean)
        if delta > 0.1:  # arbitrary threshold
            drift_flags[col] = f"Mean shift detected: {train_mean:.3f} vs {val_mean:.3f}"
    return drift_flags


EXPECTED_SCHEMA = {
    'fixed acidity': float,
    'volatile acidity': float,
    'citric acid': float,
    'residual sugar': float,
    'chlorides': float,
    'free sulfur dioxide': float,
    'total sulfur dioxide': float,
    'density': float,
    'pH': float,
    'sulphates': float,
    'alcohol': float,
    'quality': int  # Target
}

# Error log path
ERROR_LOG_PATH = Path("/home/jovyan/data/etl_error_log.json")


def validate_and_transform(chunk):
    errors = []
    processed_rows = []
    for i, row in chunk.iterrows():
        try:
            row_dict = row.to_dict()
            # Type check and conversion
            for col, expected_type in EXPECTED_SCHEMA.items():
                if pd.isnull(row_dict.get(col)):
                    raise ValueError(f"Missing value in {col}")
                row_dict[col] = expected_type(row_dict[col])

            processed_rows.append(row_dict)
        except Exception as e:
            errors.append({"index": i, "error": str(e), "row": row.to_dict()})

    return pd.DataFrame(processed_rows), errors




class DataSource:
    def get_batches(self):
        raise NotImplementedError("Subclasses must implement get_batches")


class CSVSource(DataSource):
    def __init__(self, csv_path, chunk_size=100):
        self.csv_path = Path(csv_path)
        self.chunk_size = chunk_size

    def get_batches(self):
        return pd.read_csv(self.csv_path, chunksize=self.chunk_size)
