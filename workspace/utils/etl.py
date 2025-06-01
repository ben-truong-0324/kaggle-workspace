import json
from pathlib import Path
import shutil
import inspect


import pandas as pd
import numpy as np
import json

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import FastICA

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

def fill_missing_values(df, knn_impute=False, zero_default=False):
    df = df.copy()

    if knn_impute:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.columns.difference(numeric_cols)

        imputer = KNNImputer(n_neighbors=7)
        df_numeric_imputed = pd.DataFrame(
            imputer.fit_transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index
        )
        # Combine numeric + non-numeric
        df_filled = pd.concat([df_numeric_imputed, df[non_numeric_cols]], axis=1)
        df_filled = df_filled[df.columns]  # restore original column order

    elif zero_default:
        df_filled = df.fillna(0)

    else:
        df_filled = df.fillna(method="ffill").fillna(method="bfill")
    df_filled = df_filled.fillna(0)
    return df_filled


def get_raw_dataset(dataset_name, target_column, dataset_type = "raw", drop_na = False):
    base_data_dir = Path("/home/jovyan/data") #default jupyter jovyan
    dataset_specific_base_path = base_data_dir / dataset_name
    raw_data_target_dir = dataset_specific_base_path / dataset_type
    
    # Load raw data
    csv_files = list(raw_data_target_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in raw data directory.")
    print(csv_files)
    df = pd.read_csv(csv_files[0])
    print("Dataset loaded successfully. Only one file loaded")   

    if drop_na:
        df = df.dropna() 
    else:
        df = fill_missing_values(df, knn_impute=True, zero_default = False)


    # Verify target column exists
    
    if target_column not in df.columns:
        # raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        y = None
        X = df
    else:
        y = df[target_column]
        X = df.drop(target_column, axis=1)
    
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



def run_recommended_etl(dataset_name: str, target_column: str, \
                        recommendation: dict, \
                        task_type: str = "multinomial_classification") -> dict:
    X, y = get_raw_dataset(dataset_name, target_column)

    # Apply transformations from recommendation
    numeric_log = []
    numeric_standard = []
    numeric_minmax = []
    categorical_onehot = []
    categorical_label = []
    drop_cols = []

    for col, suggs in recommendation.items():
        s1 = suggs.get("suggestion_1", "")
        s2 = suggs.get("suggestion_2", "")

        if "drop" in s1:
            drop_cols.append(col)
        elif s1 == "log_transform":
            numeric_log.append(col)
        elif s1 == "minmax_scale":
            numeric_minmax.append(col)
        elif s1 == "standard_scale":
            numeric_standard.append(col)
        elif s1 == "one_hot_encode":
            categorical_onehot.append(col)
        elif s1 == "label_encode":
            categorical_label.append(col)

    transformers = []

    if numeric_log:
        transformers.append(
            ("log", Pipeline([
                ("impute", SimpleImputer(strategy="mean")),
                ("log", FunctionTransformer(np.log1p)),
                ("scale", StandardScaler())
            ]), numeric_log)
        )

    if numeric_standard:
        transformers.append(
            ("standard", Pipeline([
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler())
            ]), numeric_standard)
        )

    if numeric_minmax:
        transformers.append(
            ("minmax", Pipeline([
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", MinMaxScaler())
            ]), numeric_minmax)
        )

    if categorical_onehot:
        transformers.append(
            ("onehot", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_onehot)
        )

    if categorical_label:
        transformers.append(
            ("label", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder())
            ]), categorical_label)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Drop specified columns
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # Train/val split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # Fit-transform
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    def get_feature_names(preprocessor):
        output_features = []
        for name, trans, cols in preprocessor.transformers_:
            if trans in ("drop", None):
                continue
            try:
                if hasattr(trans, 'get_feature_names_out'):
                    names = trans.get_feature_names_out(cols)
                elif isinstance(trans, Pipeline):
                    for step_name, step in reversed(trans.steps):
                        if hasattr(step, 'get_feature_names_out'):
                            names = step.get_feature_names_out(cols)
                            break
                    else:
                        names = cols  # no steps had it
                else:
                    names = cols
            except Exception:
                names = cols
            output_features.extend(names)
        return output_features


    feature_names = get_feature_names(preprocessor)
    X_train = pd.DataFrame(X_train, columns=feature_names, index=X_train_raw.index)
    X_val = pd.DataFrame(X_val, columns=feature_names, index=X_val_raw.index)

    X_train = fill_missing_values(X_train, knn_impute=True, zero_default = False)
    X_val = fill_missing_values(X_val, knn_impute=True, zero_default = False)


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
    etl_description = "ran eda rec"
    store[uuid] = {
        "etl_description": etl_description,
        "target_column": target_column,
        "transform_config": recommendation,
        "etl_source_code": etl_source,

    }
    with open(store_path, "w") as f:
        json.dump(store, f, indent=4)

    return {
        "etl_version": uuid,
        "etl_description": etl_description,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "transform_config": recommendation,
        "preprocessor": preprocessor,
    }

   

def transform_raw_dataframe(df: pd.DataFrame, service_cols: list = None) -> pd.DataFrame:
    df = df.copy()

    # Drop irrelevant
    df = df.drop(columns=["PassengerId",
                        #  "Name"
                        ], errors='ignore')

    for col in ["CryoSleep", "VIP"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].map({"true": True, "false": False, '0': False, '1': True})

    for col in ["HomePlanet", "Destination"]:
        if df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes

    # service_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["total_spent"] = df[service_cols].fillna(0).sum(axis=1)
    df["max_spend_category"] = df[service_cols].idxmax(axis=1).astype("category").cat.codes

    if "Cabin" in df.columns:
        cabin_parts = df["Cabin"].str.split("/", expand=True)
        if cabin_parts.shape[1] == 3:
            cabin_parts.columns = ["deck", "cabin_num", "side"]
            df["deck"] = cabin_parts["deck"].astype("category").cat.codes
            df["cabin_num"] = pd.to_numeric(cabin_parts["cabin_num"], errors="coerce")
            df["side"] = cabin_parts["side"].astype("category").cat.codes
        df = df.drop(columns=["Cabin"])

    return df


def apply_knn_imputation(df: pd.DataFrame, imputer: KNNImputer, fit: bool = False) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns
    if fit:
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = imputer.transform(df[numeric_cols])
    return df

def fit_feature_generators(X_train: pd.DataFrame):
    scaler = MinMaxScaler()
    ica = FastICA(n_components=7, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_ica = ica.fit_transform(X_train_scaled)
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    # for i in range(X_ica.shape[1]):
    #     X_train_df[f"ica_{i+1}"] = X_ica[:, i]

    kmeans = KMeans(n_clusters=5, random_state=42).fit(X_train_df)
    gmm = GaussianMixture(n_components=5, random_state=42).fit(X_train_df)

    return scaler, ica, kmeans, gmm




def transform_with_clustering(X_raw, scaler, ica, kmeans, gmm):
    X_scaled = scaler.transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_raw.columns)
    kmeans_gmm_input = X_scaled_df.copy()

    # ICA
    X_ica = ica.transform(X_scaled)
    for i in range(X_ica.shape[1]):
        X_scaled_df[f"ica_{i+1}"] = X_ica[:, i]

    # KMeans
    kmeans_labels = kmeans.predict(kmeans_gmm_input)
    kmeans_distances = pairwise_distances(kmeans_gmm_input, kmeans.cluster_centers_)
    for i in range(kmeans_distances.shape[1]):
        X_scaled_df[f"kmeans_dist_{i}"] = kmeans_distances[:, i]
    X_scaled_df["kmeans_cluster"] = kmeans_labels

    # GMM
    gmm_probs = gmm.predict_proba(kmeans_gmm_input)
    for i in range(gmm_probs.shape[1]):
        X_scaled_df[f"gmm_cluster_prob_{i}"] = gmm_probs[:, i]

    return X_scaled_df

def add_service_features(X, service_cols, thresholds):
    X = X.copy()
    for col in service_cols:
        X[f"has_used_{col}"] = X[col].fillna(0) > 0
        X[f"{col}_tp25"] = X[col] > thresholds[col]["q25"]
        X[f"{col}_tp50"] = X[col] > thresholds[col]["q50"]
        X[f"{col}_tp75"] = X[col] > thresholds[col]["q75"]
    return X


def run_custom_etl(dataset_name: str, target_column: str, test_split: float = .2) -> dict:
    X, y = get_raw_dataset(dataset_name, target_column, drop_na = False)

    service_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    X = transform_raw_dataframe(X,service_cols = service_cols)
    imputer = KNNImputer(n_neighbors=9)
    X = apply_knn_imputation(X, imputer, fit=True) 


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, random_state=42)

    # Step 1: Learn from train set
    word_stats = fit_name_word_correlations(X_train, y_train)

    # Step 2: Transform each set
    X_train = apply_name_word_features(X_train, word_stats)
    X_val = apply_name_word_features(X_val, word_stats)

    scaler, ica, kmeans, gmm = fit_feature_generators(X_train)

    X_train = transform_with_clustering(X_train, scaler, ica, kmeans, gmm)
    X_val = transform_with_clustering(X_val, scaler, ica, kmeans, gmm)

    # === Compute quartiles from X_train only ===
    quartile_thresholds = {}
    for col in service_cols:
        q25, q50, q75 = X_train[col].quantile([0.25, 0.5, 0.75])
        quartile_thresholds[col] = {"q25": q25, "q50": q50, "q75": q75}

    X_train = add_service_features(X_train, service_cols, quartile_thresholds)
    X_val = add_service_features(X_val, service_cols, quartile_thresholds)

    X_train = X_train.drop(columns=service_cols)
    X_val = X_val.drop(columns=service_cols)

    # Generate next UUID (v1, v2, etc.)
    base_data_dir = Path("/home/jovyan/data") #default jupyter jovyan
    dataset_specific_base_path = base_data_dir / dataset_name
    store_path = dataset_specific_base_path / "etl_generated_store.json"
    etl_source = inspect.getsource(run_custom_etl)
    # Load existing store
    if store_path.exists():
        with open(store_path, "r") as f:
            store = json.load(f)
    else:
        store = {}

    existing_versions = [int(k[1:]) for k in store.keys() if k.startswith("v") and k[1:].isdigit()]
    next_version_number = max(existing_versions, default=0) + 1
    uuid = f"v{next_version_number}"
    etl_description = "knnimpute, encode str to num, composite, drop, minmaxscale, ica"
    
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
        "y_val": y_val,
        "ica_transformer": ica,
        "minmax_scaler": scaler,
        "knn_imputer": imputer,
        "kmeans_cluster": kmeans,
        "gmm_cluster": gmm,
        "quartile_thresholds": quartile_thresholds,
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


def validate_X_schema(X, expected_columns=None):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("❌ X must be a pandas DataFrame.")
    if expected_columns is not None:
        missing = set(expected_columns) - set(X.columns)
        if missing:
            raise ValueError(f"❌ X is missing expected columns: {missing}")
    print("✅ X schema valid.")

def validate_y_schema(y, expected_dtype=None):
    if not isinstance(y, pd.Series):
        raise TypeError("❌ y must be a pandas Series.")
    if expected_dtype is not None:
        if str(y.dtype) != expected_dtype:
            raise TypeError(f"❌ y has dtype {y.dtype}, expected {expected_dtype}")
    if y.isnull().any():
        raise ValueError("❌ y contains null values.")
    print("✅ y schema valid.")



def fit_name_word_correlations(X_train, y_train, column="Name", min_count=5):
    import re
    from collections import defaultdict
    from scipy.stats import fisher_exact
    import pandas as pd

    def tokenize(text):
        return re.findall(r"\b\w+\b", str(text).lower())

    word_counts = defaultdict(lambda: [0, 0])  # word: [count_0, count_1]
    token_map = {}  # index → token list

    for i, (name, label) in enumerate(zip(X_train[column], y_train)):
        tokens = tokenize(name)
        token_map[i] = tokens
        for token in set(tokens):
            word_counts[token][label] += 1

    # Total per class
    total_0 = sum(y_train == 0)
    total_1 = sum(y_train == 1)

    # Compute stats
    word_stats = {}
    for word, (c0, c1) in word_counts.items():
        total = c0 + c1
        if total < min_count:
            continue
        rate = c1 / total
        table = [[c1, c0], [total_1 - c1, total_0 - c0]]
        _, p = fisher_exact(table)
        word_stats[word] = {
            "count_0": c0,
            "count_1": c1,
            "class1_rate": rate,
            "p_value": p
        }

    return word_stats


def apply_name_word_features(X, word_stats, column="Name"):
    import re
    import pandas as pd

    def tokenize(text):
        return re.findall(r"\b\w+\b", str(text).lower())

    def get_best_stats(tokens):
        best_pos = {"p": 1.0, "prob": 0.5}
        best_neg = {"p": 1.0, "prob": 0.5}
        for token in tokens:
            stats = word_stats.get(token)
            if not stats:
                continue
            if stats["class1_rate"] > 0.5 and stats["p_value"] < best_pos["p"]:
                best_pos = {"p": stats["p_value"], "prob": stats["class1_rate"]}
            elif stats["class1_rate"] < 0.5 and stats["p_value"] < best_neg["p"]:
                best_neg = {"p": stats["p_value"], "prob": stats["class1_rate"]}
        return best_pos["p"], best_pos["prob"], best_neg["p"], best_neg["prob"]

    # Generate new columns
    pos_p, pos_prob, neg_p, neg_prob = [], [], [], []

    for name in X[column]:
        tokens = tokenize(name)
        p_pos, prob_pos, p_neg, prob_neg = get_best_stats(tokens)
        pos_p.append(p_pos)
        pos_prob.append(prob_pos)
        neg_p.append(p_neg)
        neg_prob.append(prob_neg)

    X = X.copy()
    X["name_corr_pos_p_value"] = pos_p
    X["name_corr_pos_prob"] = pos_prob
    X["name_corr_neg_p_value"] = neg_p
    X["name_corr_neg_prob"] = neg_prob
    X.drop(columns=[column], inplace=True)
    return X
