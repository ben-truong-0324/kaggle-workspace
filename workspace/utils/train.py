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
import torch.nn.functional as F
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
    task_type="classification",
    eval_metric_name="val_metric",
    eval_metric_fn=None,
    bin_centers=None,
    patience=5
):
    """
    Trains a neural network model with early stopping.

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
        patience (int): Number of epochs to wait for improvement before stopping.

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
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)
        criterion = nn.MSELoss()
    elif task_type == "binary_classification":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "multiclass_classification":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long).to(device)
        criterion = nn.CrossEntropyLoss()
    elif task_type == "prob_vector":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    # === Loaders ===
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Early Stopping Variables ---
    best_val_score = float('inf') if (eval_metric_fn is None or eval_metric_name in ["KL Divergence", "Jensen-Shannon Divergence", "Cross-Entropy", "L1 Loss", "L2 Loss", "Earth Mover's Distance"]) else float('-inf')
    # If eval_metric_fn is a "loss" (lower is better), we want to minimize it.
    # If eval_metric_fn is a "score" (higher is better, like accuracy), we want to maximize it.
    # We default to minimizing loss, as your primary metrics (KL, JS, CE, L1, L2, EMD) are losses.
    # If you introduce an accuracy-like metric, adjust `best_val_score` init and comparison.
    epochs_no_improve = 0
    best_model_state = None # To save the best model state_dict

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            preds = model(xb)

            if task_type == "prob_vector":
                log_probs = F.log_softmax(preds, dim=1)
                loss = criterion(log_probs, yb)
            elif task_type in ["binary_classification", "multiclass_classification"]:
                loss = criterion(preds, yb) # These criteria expect raw logits
            else: # Regression
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

            # Convert predictions and true values to NumPy for evaluation on CPU
            if task_type == "regression":
                y_pred_np = val_preds_raw.cpu().numpy().squeeze()
                y_true_np = y_val_tensor.cpu().numpy().squeeze()
            elif task_type == "binary_classification":
                y_pred_probs_np = torch.sigmoid(val_preds_raw).cpu().numpy().squeeze()
                y_pred_np = (y_pred_probs_np > 0.5).astype(int)
                y_true_np = y_val_tensor.cpu().numpy().round().astype(int).squeeze()
            elif task_type == "multiclass_classification":
                y_pred_np = torch.argmax(val_preds_raw, dim=1).cpu().numpy()
                y_true_np = y_val_tensor.cpu().numpy().astype(int)
            elif task_type == "prob_vector":
                y_pred_np = F.softmax(val_preds_raw, dim=1).cpu().numpy() # For eval metrics, need probabilities
                y_true_np = y_val_tensor.cpu().numpy()
                if y_pred_np.ndim == 1: y_pred_np = y_pred_np.reshape(1, -1)
                if y_true_np.ndim == 1: y_true_np = y_true_np.reshape(1, -1)
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

            # === Evaluate & Log Validation Metric ===
            current_val_score = None
            if eval_metric_fn is not None:
                try:
                    if eval_metric_name == "Earth Mover's Distance":
                        if bin_centers is None:
                            raise ValueError("bin_centers must be provided for Earth Mover's Distance metric.")
                        current_val_score = eval_metric_fn(y_true_np, y_pred_np, bin_centers=bin_centers)
                    else:
                        current_val_score = eval_metric_fn(y_true_np, y_pred_np)

                    if 'mlflow' in globals():
                        mlflow.log_metric(eval_metric_name, current_val_score, step=epoch)
                    # if 'wandb' in globals():
                    if wandb.run:
                        wandb.log({eval_metric_name: current_val_score, "epoch": epoch})
                except Exception as e:
                    print(f"❌ Metric function '{eval_metric_name}' failed: {str(e)}")
                    # Decide whether to raise or just log and continue
            else:
                # Fallback to standard validation loss for early stopping if no custom metric
                # This should be consistent with the training criterion
                if task_type == "prob_vector":
                    val_loss_value = criterion(F.log_softmax(val_preds_raw, dim=1), y_val_tensor).item()
                else:
                    val_loss_value = criterion(val_preds_raw, y_val_tensor).item()
                
                current_val_score = val_loss_value # Use validation loss for early stopping
                
                if 'mlflow' in globals():
                    mlflow.log_metric("val_loss", current_val_score, step=epoch)
                if wandb.run:
                    wandb.log({"val_loss": current_val_score, "epoch": epoch})

        # Log training loss (always)
        if 'mlflow' in globals():
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        if wandb.run:
            wandb.log({"train_loss": epoch_loss, "epoch": epoch})

        # --- Early Stopping Logic ---
        if current_val_score is not None:
            # For loss-like metrics (lower is better)
            if (eval_metric_fn is None or eval_metric_name in ["KL Divergence", "Jensen-Shannon Divergence", "Cross-Entropy", "L1 Loss", "L2 Loss", "Earth Mover's Distance"]):
                if current_val_score < best_val_score:
                    best_val_score = current_val_score
                    epochs_no_improve = 0
                    best_model_state = model.state_dict() # Save best model weights
                    # print(f"Epoch {epoch+1}/{epochs}: New best validation score ({eval_metric_name if eval_metric_fn else 'val_loss'}): {best_val_score:.4f}")
                else:
                    epochs_no_improve += 1
                    # print(f"Epoch {epoch+1}/{epochs}: No improvement for {epochs_no_improve} epochs. Best: {best_val_score:.4f}. Current: {current_val_score:.4f}")
            # For score-like metrics (higher is better, e.g., accuracy, R2)
            else:
                if current_val_score > best_val_score:
                    best_val_score = current_val_score
                    epochs_no_improve = 0
                    best_model_state = model.state_dict() # Save best model weights
                    # print(f"Epoch {epoch+1}/{epochs}: New best validation score ({eval_metric_name if eval_metric_fn else 'val_loss'}): {best_val_score:.4f}")
                else:
                    epochs_no_improve += 1
                    # print(f"Epoch {epoch+1}/{epochs}: No improvement for {epochs_no_improve} epochs. Best: {best_val_score:.4f}. Current: {current_val_score:.4f}")

            if epochs_no_improve >= patience:
                # print(f"Early stopping triggered after {epoch+1} epochs (patience={patience}).")
                break # Exit training loop

    # --- Load Best Model Weights if Early Stopping Occurred ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state from early stopping.")
    else:
        # If no improvement ever happened (e.g., patience = epochs), the last state is used.
        # This can also happen if only 1 epoch runs.
        print("No improvement recorded or patience not met, using final model state.")

    # Get final predictions from the (potentially restored) best model
    model.eval()
    with torch.no_grad():
        final_val_preds_raw = model(X_val_tensor)
        if task_type == "prob_vector":
            final_y_pred_np = F.softmax(final_val_preds_raw, dim=1).cpu().numpy()
        elif task_type == "binary_classification":
            final_y_pred_np = (torch.sigmoid(final_val_preds_raw) > 0.5).int().cpu().numpy().squeeze()
        elif task_type == "multiclass_classification":
            final_y_pred_np = torch.argmax(final_val_preds_raw, dim=1).cpu().numpy()
        else: # Regression
            final_y_pred_np = final_val_preds_raw.cpu().numpy().squeeze()

    return model, final_y_pred_np



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


