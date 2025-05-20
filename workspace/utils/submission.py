import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.decomposition import FastICA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

from utils.etl import fill_missing_values, transform_raw_dataframe, apply_knn_imputation, \
                    transform_with_clustering, add_service_features, get_raw_dataset

def get_X_test(dataset_name, knn_imputer, scaler, ica_transformer, kmeans_cluster, gmm_cluster, quartile_thresholds,service_cols):
    # test_df, passenger_ids = get_submission_dataset()  # your existing loader
    #  === Load raw test CSV ===
    # base_data_dir = Path("/home/jovyan/data") #default jupyter jovyan
    # dataset_specific_base_path = base_data_dir / dataset_name
    # raw_data_target_dir = dataset_specific_base_path / "test"
    # csv_files = list(raw_data_target_dir.glob("*.csv"))
    # if not csv_files:
    #     raise FileNotFoundError("No test CSV found in /data/test/raw/")
    # test_df = pd.read_csv(csv_files[0])
    # # === Extract PassengerId ===
    # if "PassengerId" not in test_df.columns:
    #     raise ValueError("Missing 'PassengerId' column in test file.")
    # passenger_ids = test_df["PassengerId"]
    # print("✅ Test dataset loaded. Only first csv in folder loaded")
    # df = fill_missing_values(df, knn_impute=True, zero_default = False)

    test_df, _ = get_raw_dataset(dataset_name, "placeholder", dataset_type = "test", drop_na = False)
    if "PassengerId" not in test_df.columns:
        raise ValueError("Missing 'PassengerId' column in test file.")
    passenger_ids = test_df["PassengerId"]

    X_test = transform_raw_dataframe(df = test_df, service_cols= service_cols)
    X_test = apply_knn_imputation(X_test, knn_imputer)
    X_test = transform_with_clustering(X_test, scaler, ica_transformer, kmeans_cluster, gmm_cluster)
    X_test = add_service_features(X_test, service_cols, quartile_thresholds)
    X_test = X_test.drop(columns=service_cols, errors="ignore")
    return X_test, passenger_ids

# def get_X_test(dataset_name,knn_imputer = None, minmax_scaler = None, ica_transformer = None):
#     # === Load raw test CSV ===
#     base_data_dir = Path("/home/jovyan/data") #default jupyter jovyan
#     dataset_specific_base_path = base_data_dir / dataset_name
#     raw_data_target_dir = dataset_specific_base_path / "test"
#     csv_files = list(raw_data_target_dir.glob("*.csv"))
#     if not csv_files:
#         raise FileNotFoundError("No test CSV found in /data/test/raw/")
    
#     df = pd.read_csv(csv_files[0])
#     print("✅ Test dataset loaded. Only first csv in folder loaded")
    
#     df = fill_missing_values(df, knn_impute=True, zero_default = False)

#     # === Extract PassengerId ===
#     if "PassengerId" not in df.columns:
#         raise ValueError("Missing 'PassengerId' column in test file.")
#     passenger_ids = df["PassengerId"]

#     # === Apply same ETL as training ===
#     X = df.drop(columns=["PassengerId", "Name"], errors='ignore').copy()

  
#     for col in ["CryoSleep", "VIP"]:
#         if X[col].dtype == object:
#             X[col] = X[col].astype(str).str.strip().str.lower()
#             X[col] = X[col].map({"true": True, "false": False, '0': False, '1': True})
   
#     for col in ["HomePlanet", "Destination"]:
#         if X[col].dtype == object:
#             X[col] = X[col].astype("category").cat.codes

#     service_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
#     for col in service_cols:
#         X[f"has_used_{col}"] = X[col].fillna(0) > 0
#         q25, q50, q75 = X[col].quantile([0.25, 0.5, 0.75])
#         X[f"{col}_tp25"] = X[col] > q25
#         X[f"{col}_tp50"] = X[col] > q50
#         X[f"{col}_tp75"] = X[col] > q75

#     X["total_spent"] = X[service_cols].fillna(0).sum(axis=1)
#     X["max_spend_category"] = X[service_cols].idxmax(axis=1).astype("category").cat.codes
#     X = X.drop(columns=service_cols)
    
#     if "Cabin" in df.columns:
#         cabin_parts = df["Cabin"].str.split("/", expand=True)
#         if cabin_parts.shape[1] == 3:
#             cabin_parts.columns = ["deck", "cabin_num", "side"]
#             X["deck"] = cabin_parts["deck"].astype("category").cat.codes
#             X["cabin_num"] = pd.to_numeric(cabin_parts["cabin_num"], errors="coerce")
#             X["side"] = cabin_parts["side"].astype("category").cat.codes
#         X = X.drop(columns=["Cabin"], errors="ignore")
    
    
#     numeric_cols = X.select_dtypes(include='number').columns
#     X_numeric = X[numeric_cols]
#     # imputer = KNNImputer(n_neighbors=9)
#     X_numeric_imputed = pd.DataFrame(knn_imputer.transform(X_numeric), columns=numeric_cols, index=X.index)
#     X[numeric_cols] = X_numeric_imputed

#     X_scaled = minmax_scaler.transform(X)
#     X = pd.DataFrame(X_scaled, columns=X.columns)
#     # X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    
#     X_ica = ica_transformer.transform(X)
#     for i in range(X_ica.shape[1]):
#         X[f"ica_{i+1}"] = X_ica[:, i]


#     return X, passenger_ids
