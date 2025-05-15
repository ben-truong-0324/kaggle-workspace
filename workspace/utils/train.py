import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_sklearn_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, y_pred


# def train_nn_model(model, X_train, y_train, X_val, y_val, epochs=10, task_type="classification"):
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     from torch.utils.data import TensorDataset, DataLoader

#     X_train = torch.tensor(X_train.values, dtype=torch.float32)
#     y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
#     X_val = torch.tensor(X_val.values, dtype=torch.float32)
#     y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

#     criterion = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)

#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0.0

#         for xb, yb in train_loader:
#             preds = model(xb)
#             loss = criterion(preds, yb)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         epoch_loss /= len(train_loader)

#         # Evaluate
#         model.eval()
#         with torch.no_grad():
#             val_preds = model(X_val)
#             if task_type == "classification":
#                 val_preds_binary = (torch.sigmoid(val_preds) > 0.5).float()
#                 acc = (val_preds_binary.eq(y_val).sum().item()) / y_val.shape[0]
#                 mlflow.log_metric("val_acc", acc, step=epoch)
#                 wandb.log({"epoch": epoch, "train_loss": epoch_loss, "val_acc": acc})
#             else:
#                 val_loss = criterion(val_preds, y_val).item()
#                 mlflow.log_metric("val_loss", val_loss, step=epoch)
#                 wandb.log({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})

#         mlflow.log_metric("train_loss", epoch_loss, step=epoch)

#     return model, val_preds

def train_nn_model(
    model, X_train, y_train, X_val, y_val,
    epochs=10,
    task_type="classification",
    eval_metric_name="val_acc",
    eval_metric_fn=None
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    criterion = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)

            if task_type == "classification":
                probs = torch.sigmoid(val_preds)
                y_pred_np = (probs > 0.5).cpu().numpy()
            else:
                y_pred_np = val_preds.cpu().numpy()

            y_true_np = y_val.cpu().numpy()

            if eval_metric_fn is not None:
                score = eval_metric_fn(y_true_np, y_pred_np)
                mlflow.log_metric(eval_metric_name, score, step=epoch)
                wandb.log({eval_metric_name: score, "epoch": epoch})
            else:
                # Fallback default
                if task_type == "classification":
                    acc = (y_pred_np == y_true_np).mean()
                    mlflow.log_metric("val_acc", acc, step=epoch)
                    wandb.log({"val_acc": acc, "epoch": epoch})
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
        roc_auc_score, confusion_matrix,
        mean_absolute_error, mean_squared_error, r2_score
    )

    metrics = {}

    if task_type == "classification":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Try AUC only if binary classification
        if len(set(y_true)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
            except:
                metrics["roc_auc"] = None

        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    elif task_type == "regression":
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        metrics["r2"] = r2_score(y_true, y_pred)

    return metrics


# def get_next_train_script(dataset_name: str, train_description="Default training script") -> dict:
#     """
#     Returns next training version and logs description.
#     Example return:
#     {
#         "train_version": "t1",
#         "train_description": "...",
#     }
#     """
    
#     dataset_specific_base_path = base_data_dir / dataset_name
#     TRAIN_TRACKING_PATH = dataset_specific_base_path / "model_store.json"

#     if TRAIN_TRACKING_PATH.exists():
#         with open(TRAIN_TRACKING_PATH, "r") as f:
#             store = json.load(f)
#     else:
#         store = {}

#     existing_versions = [int(k[1:]) for k in store.keys() if k.startswith("t") and k[1:].isdigit()]
#     next_version_number = max(existing_versions, default=0) + 1
#     train_version = f"t{next_version_number}"

#     store[train_version] = {
#         "train_description": train_description
#     }

#     with open(TRAIN_TRACKING_PATH, "w") as f:
#         json.dump(store, f, indent=4)

#     return {
#         "train_version": train_version,
#         "train_description": train_description
#     }



# # train.py
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# def train_skl(X_train, y_train, X_val=None, y_val=None, model_cls=DecisionTreeClassifier, **model_kwargs):
#     """
#     Train a scikit-learn model dynamically.
#     - `model_cls`: the sklearn class (e.g., DecisionTreeClassifier)
#     - `model_kwargs`: parameters for the model
#     """
#     model = model_cls(**model_kwargs)
#     model.fit(X_train, y_train)

#     if X_val is not None and y_val is not None:
#         preds = model.predict(X_val)
#         acc = accuracy_score(y_val, preds)
#         print(f"[sklearn] Validation Accuracy: {acc:.4f}")

#     return model


# def train_nn(X_train, y_train, X_val=None, y_val=None, epochs=10):
#     """
#     Placeholder for a PyTorch or TensorFlow model training.
#     You can expand this as needed.
#     """
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     from torch.utils.data import TensorDataset, DataLoader

#     # Dummy model example
#     class SimpleNN(nn.Module):
#         def __init__(self, input_dim):
#             super().__init__()
#             self.linear = nn.Linear(input_dim, 1)

#         def forward(self, x):
#             return self.linear(x)

#     # Prepare data
#     X_train = torch.tensor(X_train.values, dtype=torch.float32)
#     y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

#     model = SimpleNN(input_dim=X_train.shape[1])
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)

#     for epoch in range(epochs):
#         for xb, yb in train_loader:
#             preds = model(xb)
#             loss = criterion(preds, yb)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f"[NN] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

#     return model


# def pred(model, X, framework="sklearn"):
#     if framework == "sklearn":
#         return model.predict(X)
#     elif framework == "torch":
#         import torch
#         model.eval()
#         with torch.no_grad():
#             X_tensor = torch.tensor(X.values, dtype=torch.float32)
#             return model(X_tensor).squeeze().numpy()
#     else:
#         raise ValueError("Unsupported framework for prediction.")



