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

def create_submission_file(
    passenger_ids: pd.Series,
    predictions: np.ndarray, # Or pd.Series
    target_column_name: str,
    output_filename: str = "submission.csv",
    target_is_boolean: bool = False
):
    """
    Creates a submission CSV file.

    Args:
        passenger_ids (pd.Series): Series containing PassengerIds.
        predictions (np.ndarray or pd.Series): Array or Series of model predictions.
        target_column_name (str): The name of the target column in the submission file.
        output_filename (str, optional): Name of the output CSV file. 
                                         Defaults to "submission.csv".
        target_is_boolean (bool, optional): If True, predictions will be cast to bool.
                                            Defaults to False.
    """
    print(f"\n--- Creating submission file: {output_filename} ---")
    
    final_predictions = predictions
    if target_is_boolean:
        final_predictions = predictions.astype(bool)

    if not isinstance(passenger_ids, pd.Series):
        try:
            passenger_ids = pd.Series(passenger_ids)
        except Exception as e:
            raise TypeError(f"passenger_ids could not be converted to a pandas Series. Error: {e}")

    if not isinstance(final_predictions, pd.Series):
         try:
            final_predictions = pd.Series(final_predictions, name=target_column_name)
         except Exception as e:
            raise TypeError(f"predictions could not be converted to a pandas Series. Error: {e}")


    submission_df = pd.DataFrame({
        "PassengerId": passenger_ids,
        target_column_name: final_predictions,
    })
    
    try:
        submission_df.to_csv(output_filename, index=False)
        print(f"🎉 {output_filename} generated successfully!")
        print(submission_df.head())
    except Exception as e:
        print(f"❌ Error saving submission file: {e}")
        raise
    
    return submission_df
