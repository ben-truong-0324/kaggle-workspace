import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import os
os.environ["WANDB_BASE_URL"] = "http://wandb:8080"
os.environ["WANDB_DEBUG"] = "true"
os.environ["WANDB_DEBUG_LOG_PATH"] = "/tmp/wandb_debug.log"
import wandb

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def conditionally_encode_labels(y_train, y_val):
    """
    Conditionally apply LabelEncoder to y_train and y_val
    if class labels are not already 0-indexed and consecutive.

    Returns:
        y_train_encoded (pd.Series)
        y_val_encoded (pd.Series)
        label_encoder (LabelEncoder or None)
        label_encoder_applied (bool)
    """
    y_labels_sorted = sorted(pd.Series(y_train).unique())
    should_encode = y_labels_sorted[0] != 0 or y_labels_sorted != list(range(len(y_labels_sorted)))

    if should_encode:
        print("🔁 Applying LabelEncoder to remap class labels to [0...N-1]")
        le = LabelEncoder()
        y_train_encoded = pd.Series(le.fit_transform(y_train))
        y_val_encoded = pd.Series(le.transform(y_val))
        return y_train_encoded, y_val_encoded, le, True
    else:
        print("✅ Labels already normalized, no encoding needed")
        return pd.Series(y_train), pd.Series(y_val), None, False


def train_sklearn_model(model, X_train, y_train, X_val, y_val, task_type):
    if task_type not in ("regression", "multioutput_regression", "prob_vector"):
        raise ValueError("Only regression/probability-vector/multioutput tasks supported in this template.")

    # Fit model directly (no encoding needed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Optionally, ensure output is 2D (n_samples, n_bins)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    y_pred = np.clip(y_pred, 0, None)
    y_pred = y_pred / (y_pred.sum(axis=1, keepdims=True) + 1e-9)
    return model, y_pred


def train_nn_model(
    model, X_train, y_train, X_val, y_val,
    epochs=10,
    lr = .005,
    task_type="classification",  # "binary_classification", "multiclass_classification", "regression", "prob_vector"
    eval_metric_name="val_metric",
    eval_metric_fn=None,
    bin_centers=None # Added for Earth Mover's Distance
):
    """
    Trains a neural network model.

    Args:
        model (torch.nn.Module): The neural network model to train.
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.DataFrame or np.ndarray): Training targets.
        X_val (pd.DataFrame or np.ndarray): Validation features.
        y_val (pd.DataFrame or np.ndarray): Validation targets.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        task_type (str): Type of machine learning task.
        eval_metric_name (str): Name of the evaluation metric for logging.
        eval_metric_fn (callable): Custom evaluation metric function.
        bin_centers (np.ndarray, optional): Bin centers, required for EMD in prob_vector.

    Returns:
        tuple: (trained_model, y_pred_val_numpy)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # === Convert Inputs to Tensors ===
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)

    # Determine criterion (loss function) and y_tensor dtype based on task_type
    if task_type == "regression":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
        criterion = nn.MSELoss()
    elif task_type == "binary_classification":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
        criterion = nn.BCEWithLogitsLoss() # Expects raw logits
    elif task_type == "multiclass_classification":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
        criterion = nn.CrossEntropyLoss() # Expects raw logits (no softmax in model)
    elif task_type == "prob_vector":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)
        criterion = nn.KLDivLoss(reduction='batchmean') # 'batchmean' averages loss over the batch
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    # === Loaders ===
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            preds = model(xb)
            # Adjust preds for loss function based on task_type
            if task_type == "prob_vector":
                # KLDivLoss expects log-probabilities for input
                loss = criterion(torch.log(preds), yb)
            else:
                loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        # === Validation ===
        model.eval()
        with torch.no_grad():
            val_preds_raw = model(X_val_tensor)

            # Convert predictions and true values to NumPy for evaluation
            # The format of y_pred_np and y_true_np depends on the task_type
            if task_type == "regression":
                y_pred_np = val_preds_raw.cpu().numpy().squeeze()
                y_true_np = y_val_tensor.cpu().numpy().squeeze()
            elif task_type == "binary_classification":
                # For eval metrics, we need probabilities or hard labels
                y_pred_probs_np = torch.sigmoid(val_preds_raw).cpu().numpy().squeeze()
                y_pred_np = (y_pred_probs_np > 0.5).astype(int) # Hard labels for classification metrics
                y_true_np = y_val_tensor.cpu().numpy().round().astype(int).squeeze()
            elif task_type == "multiclass_classification":
                # For eval metrics, we need class indices
                y_pred_np = torch.argmax(val_preds_raw, dim=1).cpu().numpy()
                y_true_np = y_val_tensor.cpu().numpy().astype(int)
            elif task_type == "prob_vector":
                # y_pred_np should be the probability distribution
                y_pred_np = val_preds_raw.cpu().numpy()
                y_true_np = y_val_tensor.cpu().numpy()
                # Ensure they are 2D arrays (n_samples, n_bins)
                if y_pred_np.ndim == 1: y_pred_np = y_pred_np.reshape(1, -1)
                if y_true_np.ndim == 1: y_true_np = y_true_np.reshape(1, -1)
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

            # === Evaluate & Log ===
            # Use the custom eval_metric_fn if provided
            if eval_metric_fn is not None:
                try:
                    # Pass bin_centers if eval_metric_name is EMD
                    if eval_metric_name == "Earth Mover's Distance":
                        if bin_centers is None:
                            raise ValueError("bin_centers must be provided for Earth Mover's Distance metric.")
                        score = eval_metric_fn(y_true_np, y_pred_np, bin_centers=bin_centers)
                    else:
                        score = eval_metric_fn(y_true_np, y_pred_np)

                    # Log to MLflow and WandB if available
                    if 'mlflow' in globals():
                        mlflow.log_metric(eval_metric_name, score, step=epoch)
                    if 'wandb' in globals():
                        wandb.log({eval_metric_name: score, "epoch": epoch})
                except Exception as e:
                    print(f"❌ Metric function '{eval_metric_name}' failed: {str(e)}")
                    # Decide whether to raise or just log and continue
                    # raise # Uncomment if you want to stop on metric calculation errors
            else:
                # Fallback to standard validation loss if no custom metric function is provided
                val_loss_value = criterion(val_preds_raw, y_val_tensor).item()
                if 'mlflow' in globals():
                    mlflow.log_metric("val_loss", val_loss_value, step=epoch)
                if 'wandb' in globals():
                    wandb.log({"val_loss": val_loss_value, "epoch": epoch})

        # Log training loss (always)
        if 'mlflow' in globals():
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        if 'wandb' in globals():
            wandb.log({"train_loss": epoch_loss, "epoch": epoch})

    return model, y_pred_np




def log_model_artifact(model, model_name, framework="sklearn"):
    artifact_path = f"{model_name}"
    
    if framework == "sklearn":
        import joblib
        fname = f"{artifact_path}.pkl"
        joblib.dump(model, fname)
        mlflow.sklearn.log_model(model, artifact_path="model")
    elif framework == "torch":
        import torch
        fname = f"{artifact_path}.pt"
        torch.save(model.state_dict(), fname)
        mlflow.pytorch.log_model(model, artifact_path="model")
    else:
        raise ValueError("Unsupported framework")

    artifact = wandb.Artifact(artifact_path, type="model")
    artifact.add_file(fname)
    wandb.log_artifact(artifact)
    

# def log_final_metrics(eval_metrics):
#     for metric, value in eval_metrics.items():
#         if isinstance(value, (int, float)):
#             mlflow.log_metric(metric, value)
#     wandb.log(eval_metrics)
def log_final_metrics(eval_metrics):

    flat_metrics = {}
    for metric, value in eval_metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(metric, value)
            flat_metrics[metric] = value

        elif isinstance(value, dict):
            # Log to W&B as is
            wandb.log({metric: value})

            # Log to MLflow as JSON artifact
            # with open(f"{metric}.json", "w") as f:
            #     json.dump(value, f, indent=2)
            # mlflow.log_artifact(f"{metric}.json")

        else:
            # fallback — log what you can
            wandb.log({metric: value})

    # Log flat metrics to W&B in one call (avoids duplicates)
    wandb.log(flat_metrics)


