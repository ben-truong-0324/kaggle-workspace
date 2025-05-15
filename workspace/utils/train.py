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

def train_sklearn_model(model, X_train, y_train, X_val, y_val, task_type):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, y_pred


def train_nn_model(
    model, X_train, y_train, X_val, y_val,
    epochs=10,
    lr = lr,
    task_type="classification",  # "binary_classification", "multiclass_classification", or "regression"
    eval_metric_name="val_metric",
    eval_metric_fn=None
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    # === Convert Inputs ===
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)

    if task_type == "regression":
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
        criterion = nn.MSELoss()
    elif task_type == "binary_classification":
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "multiclass_classification":
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        y_val = torch.tensor(y_val.values, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    # === Loaders ===
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        # === Validation ===
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)

            if task_type == "regression":
                y_pred_np = val_preds.cpu().numpy().squeeze()
                y_true_np = y_val.cpu().numpy().squeeze()

            elif task_type == "binary_classification":
                probs = torch.sigmoid(val_preds)
                y_pred_np = (probs > 0.5).int().cpu().numpy().squeeze()
                y_true_np = y_val.cpu().numpy().round().astype(int).squeeze()

            elif task_type == "multiclass_classification":
                y_pred_np = torch.argmax(val_preds, dim=1).cpu().numpy()
                y_true_np = y_val.cpu().numpy().astype(int).squeeze()

            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

            # Final assurance: both should be 1D ints for classification
            if task_type != "regression":
                y_pred_np = y_pred_np.astype(int).reshape(-1)
                y_true_np = y_true_np.astype(int).reshape(-1)

            # === Evaluate & Log ===
            if eval_metric_fn is not None:
                try:
                    score = eval_metric_fn(y_true_np, y_pred_np)
                    mlflow.log_metric(eval_metric_name, score, step=epoch)
                    wandb.log({eval_metric_name: score, "epoch": epoch})
                except Exception as e:
                    print("❌ Metric function failed:", str(e))
                    raise  

            if eval_metric_fn is not None:
                score = eval_metric_fn(y_true_np, y_pred_np)
                mlflow.log_metric(eval_metric_name, score, step=epoch)
                wandb.log({eval_metric_name: score, "epoch": epoch})
            else:
                val_loss = criterion(val_preds, y_val).item()
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                wandb.log({"val_loss": val_loss, "epoch": epoch})

        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        wandb.log({"train_loss": epoch_loss, "epoch": epoch})

    return model, y_pred_np



def evaluate_model(y_true, y_pred, task_type):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        mean_absolute_error, mean_squared_error, r2_score
    )
    import numpy as np
    import pandas as pd

    metrics = {}

    if task_type in ["binary_classification", "multiclass_classification"]:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # ROC AUC only for binary
        if len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
            except:
                metrics["roc_auc"] = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        class_metrics = {
            label: {
                "precision": round(values["precision"], 3),
                "recall": round(values["recall"], 3),
                "f1": round(values["f1-score"], 3),
                "support": int(values["support"]),
            }
            for label, values in class_report.items()
            if label not in ["accuracy", "macro avg", "weighted avg"]
        }
        metrics["per_class_metrics"] = class_metrics

    elif task_type == "regression":
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        metrics["r2"] = r2_score(y_true, y_pred)

        # === Bin target into quantiles and evaluate per bin
        bins = pd.qcut(y_true, q=5, duplicates='drop')
        bin_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "bin": bins
        })
        bin_stats = bin_df.groupby("bin").apply(
            lambda df: {
                "mae": mean_absolute_error(df["y_true"], df["y_pred"]),
                "rmse": mean_squared_error(df["y_true"], df["y_pred"], squared=False),
                "count": len(df)
            }
        )
        metrics["binned_performance"] = {str(k): v for k, v in bin_stats.to_dict().items()}

    return metrics


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
            with open(f"{metric}.json", "w") as f:
                json.dump(value, f, indent=2)
            mlflow.log_artifact(f"{metric}.json")

        else:
            # fallback — log what you can
            wandb.log({metric: value})

    # Log flat metrics to W&B in one call (avoids duplicates)
    wandb.log(flat_metrics)




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
