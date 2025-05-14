import os
import shutil
import uuid
import hashlib
import json
import importlib.util
import inspect
import pickle
import mlflow
from datetime import datetime

DATA_DIR = "/home/jovyan/data"

def hash_file(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def run_new_etl(dataset_name: str, raw_data_path: str, etl_script_path: str) -> dict:
    """
    Run or reuse an ETL pipeline and track it in MLflow.

    Args:
        dataset_name (str): e.g. 'red-wine-quality-cortez-et-al-2009'
        raw_data_path (str): path to the raw dataset (CSV or folder)
        etl_script_path (str): path to Python script defining `run_etl(raw_data_path)`

    Returns:
        dict: {
            'etl_version': UUID,
            'processed_path': str,
            'etl_script_path': str
        }
    """
    # Hash the ETL script to detect changes
    etl_hash = hash_file(etl_script_path)
    etl_version = str(uuid.uuid4())
    processed_dir = os.path.join(DATA_DIR, "processed", dataset_name, etl_version)
    os.makedirs(processed_dir, exist_ok=True)

    # Start MLflow run for ETL tracking
    mlflow.set_experiment(f"{dataset_name}_ETL")
    with mlflow.start_run(run_name=f"ETL_{dataset_name}_{etl_version}"):
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("etl_script_hash", etl_hash)
        mlflow.log_param("etl_version", etl_version)
        mlflow.log_param("run_time", datetime.now().isoformat())

        # Save script for reproducibility
        saved_script_path = os.path.join(processed_dir, "etl_script.py")
        shutil.copy(etl_script_path, saved_script_path)
        mlflow.log_artifact(saved_script_path)

        # Import and run ETL
        spec = importlib.util.spec_from_file_location("etl_module", etl_script_path)
        etl_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(etl_module)

        if not hasattr(etl_module, "run_etl"):
            raise AttributeError("ETL script must define a `run_etl(raw_data_path)` function")

        print(f"Running ETL from: {etl_script_path}")
        result = etl_module.run_etl(raw_data_path)

        # Save processed result
        result_path = os.path.join(processed_dir, "data.pkl")
        with open(result_path, "wb") as f:
            pickle.dump(result, f)

        mlflow.log_artifact(result_path)

        print(f"✅ ETL complete. Version: {etl_version}")
        print(f"Processed data saved to: {result_path}")

    return {
        "etl_version": etl_version,
        "processed_path": result_path,
        "etl_script_path": saved_script_path
    }
