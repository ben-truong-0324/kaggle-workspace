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

import matplotlib.pyplot as plt


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
    """
    Trains a scikit-learn model and returns the model, main predictions,
    and class probabilities (if applicable).

    Args:
        model: Scikit-learn model instance.
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.DataFrame, pd.Series, or np.ndarray): Training target.
        X_val (pd.DataFrame or np.ndarray): Validation features.
        y_val (pd.DataFrame, pd.Series, or np.ndarray): Validation target (not used directly in this function
                                                            for training, but typically present for context).
        task_type (str): Type of task ("regression", "multioutput_regression",
                         "prob_vector", "multinomial_classification").

    Returns:
        tuple: (
            trained_model (sklearn.base.BaseEstimator),
            y_pred_main (np.ndarray): Main predictions (labels, values, or probability vector).
            y_pred_probabilities (np.ndarray or None): Class probabilities from predict_proba(),
                                                       or None if not applicable.
        )
    """
    if task_type not in ("regression", "multioutput_regression", "prob_vector", "multinomial_classification"):
        raise ValueError(f"Task type '{task_type}' not supported. Supported: regression, multioutput_regression, prob_vector, multinomial_classification.")

    # Prepare y_train for scikit-learn (typically 1D for classifiers/regressors)
    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
        y_train_processed = y_train.values.ravel()
    else:
        y_train_processed = np.asarray(y_train).ravel() # Ensure it's a NumPy array and 1D

    # Fit model
    model.fit(X_train, y_train_processed)

    # Get main predictions
    y_pred_raw = model.predict(X_val)

    y_pred_main = None
    y_pred_probabilities = None

    if task_type == "multinomial_classification":
        y_pred_main = y_pred_raw  # These are class labels, typically (n_samples,)
        if y_pred_main.ndim == 1:
            y_pred_main = y_pred_main.reshape(-1, 1)  # Consistent output shape (n_samples, 1)

        if hasattr(model, "predict_proba"):
            y_pred_probabilities = model.predict_proba(X_val)  # Shape (n_samples, n_classes)
        else:
            print(f"Warning: Model {model} for '{task_type}' does not have a predict_proba method. "
                  "Class probabilities cannot be returned.")

    elif task_type == "prob_vector":
        # For "prob_vector", y_pred_raw is processed to become the probability vector.
        # y_pred_probabilities will remain None as y_pred_main *is* the probability output.
        current_pred = y_pred_raw
        
        # Ensure current_pred is 2D for consistent processing, even if it's (N,1)
        if current_pred.ndim == 1:
            current_pred = current_pred.reshape(-1, 1)

        current_pred = np.clip(current_pred, 0, None)
        pred_sum = current_pred.sum(axis=1, keepdims=True)
        
        # Handle rows that sum to 0:
        # If a row sums to 0, dividing by safe_sum=1 makes it all zeros.
        # If uniform probability is desired for such rows (and multiple bins):
        num_bins = current_pred.shape[1]
        zero_sum_rows_mask = (pred_sum.squeeze() == 0)

        safe_sum = np.where(zero_sum_rows_mask, 1, pred_sum.squeeze()).reshape(-1, 1)
        y_pred_main = current_pred / (safe_sum + 1e-9) # Epsilon for numerical stability

        if num_bins > 1: # Apply uniform only if there are multiple bins
            y_pred_main[zero_sum_rows_mask, :] = 1.0 / num_bins
        elif num_bins == 1: # If only one bin, a zero sum row correctly remains 0
             y_pred_main[zero_sum_rows_mask, :] = 0.0


    elif task_type in ("regression", "multioutput_regression"):
        y_pred_main = y_pred_raw
        if y_pred_main.ndim == 1 and task_type == "regression": # Standard regression output usually 1D target
            y_pred_main = y_pred_main.reshape(-1, 1)
        # For multioutput_regression, y_pred_raw might already be 2D (n_samples, n_outputs)
        # y_pred_probabilities remains None for regression.

    else:
        # This case should ideally not be reached due to the initial check,
        # but as a fallback:
        y_pred_main = y_pred_raw
        if y_pred_main.ndim == 1:
             y_pred_main = y_pred_main.reshape(-1,1)

    return model, y_pred_main, y_pred_probabilities



###########################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianDirectionalLoss(nn.Module):
    def __init__(self, bin_centers, right_penalty_multiplier=1.2):
        super().__init__()
        if not isinstance(bin_centers, torch.Tensor):
            # Convert if it's a list or numpy array, ensure it's a tensor
            bin_centers = torch.tensor(bin_centers, dtype=torch.float32)
        self.bin_centers = bin_centers  # Expected to be 1D tensor
        self.right_penalty_multiplier = right_penalty_multiplier
        self.num_bins = len(bin_centers)

        # Ensure bin_centers is registered as a buffer if you want it to move with model.to(device)
        # and be part of state_dict, though for loss functions, manual .to(device) is common.
        self.register_buffer('_bin_centers_internal', self.bin_centers, persistent=False)


    def forward(self, preds_logits, yb_indices):
        """
        Calculates the custom loss.
        Args:
            preds_logits (torch.Tensor): Raw output from the model (logits).
                                         Shape: (batch_size, num_bins)
            yb_indices (torch.Tensor): True bin indices (LongTensor).
                                       Shape: (batch_size)
        Returns:
            torch.Tensor: Scalar mean loss for the batch.
        """
        current_bin_centers = self._bin_centers_internal
        if current_bin_centers.device != preds_logits.device:
            current_bin_centers = current_bin_centers.to(preds_logits.device)

        # Part 1: Bayesian-inspired similarity component (Hellinger distance based)
        # -----------------------------------------------------------------------
        # Convert logits to probabilities
        p_pred = F.softmax(preds_logits, dim=1)  # Shape: (batch_size, num_bins)

        # Create one-hot encoded target distribution from true bin indices
        # yb_indices needs to be LongTensor for F.one_hot
        p_target_one_hot = F.one_hot(yb_indices, num_classes=self.num_bins).float()
        # Shape: (batch_size, num_bins)

        # Calculate Bhattacharyya coefficient (BC) for each sample: sum(sqrt(p_i * q_i))
        # This measures the overlap between the two distributions.
        sqrt_elementwise_product = torch.sqrt(p_pred * p_target_one_hot)
        bhattacharyya_coeffs = torch.sum(sqrt_elementwise_product, dim=1)  # Shape: (batch_size)

        # The base loss for each sample is derived from Hellinger distance (1 - BC).
        # A higher BC (more similarity) means lower loss. Max BC is 1 (loss 0). Min BC is 0 (loss 1).
        loss_similarity_per_sample = 1.0 - bhattacharyya_coeffs

        # Part 2: Asymmetric Directional Penalty Multiplier
        # -------------------------------------------------
        # Calculate the expected value of the predicted distribution
        # Unsqueeze current_bin_centers to (1, num_bins) for broadcasting with p_pred (batch_size, num_bins)
        expected_value_pred = torch.sum(p_pred * current_bin_centers.unsqueeze(0), dim=1)  # Shape: (batch_size)

        # Get the numerical value of the actual (true) bin
        value_actual_bin = current_bin_centers[yb_indices]  # Shape: (batch_size)

        # Determine if the prediction's expected value is "to the right" of the actual value
        is_pred_to_the_right = expected_value_pred > value_actual_bin  # Boolean tensor, Shape: (batch_size)

        # Create multipliers: 1.2 if prediction is to the right, 1.0 otherwise
        loss_multipliers = torch.ones_like(loss_similarity_per_sample) # Default multiplier is 1.0
        loss_multipliers[is_pred_to_the_right] = self.right_penalty_multiplier

        # Apply the multiplier to the similarity-based loss
        final_loss_per_sample = loss_similarity_per_sample * loss_multipliers

        # Return the mean loss over the batch
        return torch.mean(final_loss_per_sample)
    ################################################




def train_nn_model(
    model, X_train, y_train, X_val, y_val,
    epochs=10,
    lr = .005,
    task_type="multinomial_classification", # Ensure this is set correctly
    eval_metric_name="val_metric",
    eval_metric_fn=None,
    bin_centers=None,
    patience=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # === Convert Inputs to Tensors ===
    X_train_tensor = torch.tensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val, dtype=torch.float32).to(device)

    # Convert from DataFrame to 1D NumPy array of ints
    y_train = y_train.squeeze().astype(int).values if isinstance(y_train, pd.DataFrame) else y_train
    y_val = y_val.squeeze().astype(int).values if isinstance(y_val, pd.DataFrame) else y_val

    y_train_np = y_train.values if isinstance(y_train, pd.DataFrame) else np.asarray(y_train)
    y_val_np = y_val.values if isinstance(y_val, pd.DataFrame) else np.asarray(y_val)

    criterion = None
    if task_type == "regression":
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1).to(device)
        criterion = nn.MSELoss()
    elif task_type == "binary_classification":
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1).to(device)
        criterion = nn.BCEWithLogitsLoss() # Expects raw logits from model
    elif task_type == "multinomial_classification":
        # Convert from DataFrame to 1D NumPy array of ints
        y_train_tensor = torch.tensor(y_train_np.squeeze(), dtype=torch.long).to(device)
        y_val_tensor = torch.tensor(y_val_np.squeeze(), dtype=torch.long).to(device)
        criterion = nn.CrossEntropyLoss() # Expects raw logits from model
    elif task_type == "prob_vector":
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).to(device)
        criterion = nn.KLDivLoss(reduction='batchmean') # Expects log-probabilities as input
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_score = float('inf')
    # Adjust for metrics where higher is better (e.g., accuracy, F1)
    higher_is_better_metrics = ["accuracy", "f1_score", "r2_score", "auc_roc"] # Add more if needed
    if eval_metric_fn is not None and any(m_name in eval_metric_name.lower() for m_name in higher_is_better_metrics):
        best_val_score = float('-inf')

    epochs_no_improve = 0
    best_model_state = None
    y_preds_per_epoch_list = []

    custom_loss_fn_instance = None
    if task_type == "prob_vector" and bin_centers is not None:
        custom_loss_fn_instance = BayesianDirectionalLoss(
            bin_centers=bin_centers,
            right_penalty_multiplier=1.2
        ).to(device)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss_sum = 0.0

        for xb, yb in train_loader:
            preds_raw = model(xb) # Raw logits 
            current_loss = None
            if task_type == "prob_vector":
                if custom_loss_fn_instance:
                    current_loss = custom_loss_fn_instance(preds_raw, yb)
                else: # Fallback to KLDivLoss
                    log_probs = F.log_softmax(preds_raw, dim=1) # KLDivLoss expects log_softmax input
                    current_loss = criterion(log_probs, yb) # yb should be target probabilities
            else: 
                current_loss = criterion(preds_raw, yb)
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            epoch_train_loss_sum += current_loss.item() * xb.size(0)
        avg_epoch_train_loss = epoch_train_loss_sum / len(train_loader.dataset)

        # === Validation ===
        model.eval()
        current_y_pred_processed_np = None # For y_preds_per_epoch_list
        y_true_for_eval_np = None          # For eval_metric_fn

        with torch.no_grad():
            val_preds_raw_logits = model(X_val_tensor) # Raw logits for the entire validation set

            # Process predictions for y_preds_per_epoch_list and for eval_metric_fn
            if task_type == "regression":
                current_y_pred_processed_np = val_preds_raw_logits.cpu().numpy().squeeze()
                y_true_for_eval_np = y_val_tensor.cpu().numpy().squeeze()
            elif task_type == "binary_classification":
                val_probs_pos_class = torch.sigmoid(val_preds_raw_logits).cpu().numpy() # Prob P(1)
                current_y_pred_processed_np = (val_probs_pos_class > 0.5).astype(int).squeeze() # Labels
                y_true_for_eval_np = y_val_tensor.cpu().numpy().round().astype(int).squeeze()
            elif task_type == "multinomial_classification":
                current_y_pred_processed_np = torch.argmax(val_preds_raw_logits, dim=1).cpu().numpy() # Labels
                y_true_for_eval_np = y_val_tensor.cpu().numpy() # Already (N,), long
            elif task_type == "prob_vector":
                current_y_pred_processed_np = F.softmax(val_preds_raw_logits, dim=1).cpu().numpy() # Probabilities
                y_true_for_eval_np = y_val_tensor.cpu().numpy()
            
            if current_y_pred_processed_np is not None:
                y_preds_per_epoch_list.append(current_y_pred_processed_np.copy())

            # === Evaluate & Log Validation Metric for Early Stopping ===
            current_val_score_for_stopping = None
            if eval_metric_fn is not None:
                try:
                    # Prepare appropriate y_pred for the metric function
                    # Some metrics might want probabilities, others labels
                    # For simplicity, this example uses current_y_pred_processed_np (often labels)
                    # If eval_metric_fn needs probabilities for classification, adjust here:
                    y_pred_for_eval = current_y_pred_processed_np
                    if task_type == "binary_classification" and "auc" in eval_metric_name.lower(): # e.g. roc_auc
                        y_pred_for_eval = torch.sigmoid(val_preds_raw_logits).cpu().numpy().squeeze() # Pass P(1)
                    elif task_type == "multinomial_classification" and "auc" in eval_metric_name.lower():
                        y_pred_for_eval = F.softmax(val_preds_raw_logits, dim=1).cpu().numpy() # Pass (N,C) probs

                    if eval_metric_name == "Earth Mover's Distance":
                        if bin_centers is None: raise ValueError("bin_centers needed for EMD.")
                        current_val_score_for_stopping = eval_metric_fn(y_true_for_eval_np, y_pred_for_eval, bin_centers=bin_centers)
                    else:
                        current_val_score_for_stopping = eval_metric_fn(y_true_for_eval_np, y_pred_for_eval)
                except Exception as e:
                    print(f"❌ Metric function '{eval_metric_name}' failed: {str(e)}. Using validation loss.")
                    # Fallback to validation loss
                    if task_type == "prob_vector": # KLDiv expects log_softmax
                        current_val_score_for_stopping = criterion(F.log_softmax(val_preds_raw_logits, dim=1), y_val_tensor).item()
                    else:
                        current_val_score_for_stopping = criterion(val_preds_raw_logits, y_val_tensor).item()
            else: # Default to validation loss
                if task_type == "prob_vector":
                    current_val_score_for_stopping = criterion(F.log_softmax(val_preds_raw_logits, dim=1), y_val_tensor).item()
                else:
                    current_val_score_for_stopping = criterion(val_preds_raw_logits, y_val_tensor).item()
            
            log_metric_name_for_stopping = eval_metric_name if eval_metric_fn and current_val_score_for_stopping is not None else "val_loss"
            
            # WandB/MLFlow logging (assuming they are initialized globally)
            # if 'wandb' in globals() and wandb.run: wandb.log({log_metric_name_for_stopping: current_val_score_for_stopping, "train_loss": avg_epoch_train_loss, "epoch": epoch})
            # if 'mlflow' in globals() and mlflow.active_run(): 
            #     mlflow.log_metric("train_loss", avg_epoch_train_loss, step=epoch)
            #     mlflow.log_metric(log_metric_name_for_stopping, current_val_score_for_stopping, step=epoch)
        
        print(f"Epoch {epoch+1}/{epochs} => Train Loss: {avg_epoch_train_loss:.4f}, Val Score ({log_metric_name_for_stopping}): {current_val_score_for_stopping:.4f}")

        # --- Early Stopping Logic ---
        improved = False
        if best_val_score == float('inf'): # Minimize
            if current_val_score_for_stopping < best_val_score: improved = True
        elif best_val_score == float('-inf'): # Maximize
            if current_val_score_for_stopping > best_val_score: improved = True
        # Check based on metric name if not explicitly set by higher_is_better_metrics
        elif any(m_name in log_metric_name_for_stopping.lower() for m_name in higher_is_better_metrics): # Maximize
            if current_val_score_for_stopping > best_val_score: improved = True
        else: # Minimize (default)
            if current_val_score_for_stopping < best_val_score: improved = True
            
        if improved:
            best_val_score = current_val_score_for_stopping
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"    New best validation score: {best_val_score:.4f}")
        else:
            epochs_no_improve += 1
            # print(f"    No improvement for {epochs_no_improve} epochs. Best: {best_val_score:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model state with val score: {best_val_score:.4f}")
    else:
        print("Using final model state (no improvement or patience not met).")

    # === Generate Final Outputs from the Best Model State ===
    model.eval()
    final_y_pred_main = None
    final_y_pred_probas_for_roc = None
    with torch.no_grad():
        final_val_preds_raw_logits = model(X_val_tensor)

        if task_type == "regression":
            final_y_pred_main = final_val_preds_raw_logits.cpu().numpy()
            if final_y_pred_main.ndim > 1 and final_y_pred_main.shape[1] == 1 : final_y_pred_main = final_y_pred_main.squeeze(1)
            # else, it's multi-output regression, keep as is (N, M)
            final_y_pred_probas_for_roc = None
        elif task_type == "binary_classification":
            probs_pos_class = torch.sigmoid(final_val_preds_raw_logits) # Shape (N, 1)
            final_y_pred_main = (probs_pos_class > 0.5).int().cpu().numpy().squeeze() # Labels (N,)
            
            probs_neg_class = 1.0 - probs_pos_class
            final_y_pred_probas_for_roc = torch.cat((probs_neg_class, probs_pos_class), dim=1).cpu().numpy() # Shape (N, 2)
        elif task_type == "multinomial_classification":
            final_y_pred_main = torch.argmax(final_val_preds_raw_logits, dim=1).cpu().numpy() # Labels (N,), argmax for label with highest score
            final_y_pred_probas_for_roc = F.softmax(final_val_preds_raw_logits, dim=1).cpu().numpy() # Probs (N, C), softmax to assign softmax prob value in [0,1] for each label
        elif task_type == "prob_vector":
            final_y_pred_main = F.softmax(final_val_preds_raw_logits, dim=1).cpu().numpy() # Probs (N, K)
            final_y_pred_probas_for_roc = None # Main output is already the probability vector
        else: # Should have been caught earlier
            raise ValueError(f"Unsupported task_type for final prediction processing: {task_type}")
    #is model still on GPU to(Device)?
    return model, final_y_pred_main, final_y_pred_probas_for_roc, y_preds_per_epoch_list




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




def train_nn_model_with_demo(
    model, X_train, y_train, X_val, y_val,
    epochs=10,  # Main training epochs, demo mode uses a fixed 10 epochs
    lr=0.005,
    task_type="multinomial_classification", # Ensure this is set correctly
    eval_metric_name="val_metric", # Unused by this demo logic
    eval_metric_fn=None, # Unused by this demo logic
    bin_centers=None, # Unused by this demo logic
    patience=5 # Unused by this demo logic
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if task_type == "multinomial_classification":
        print(f"INFO: Running DEMO for task_type: {task_type} on a subset of training data.")
        print("INFO: This demo will run for a fixed 10 epochs with visualizations of logits.")
        print(f"INFO: The 'epochs={epochs}' parameter from the function call is ignored for this demo mode.")
        print("INFO: Parameters X_val, y_val, eval_metric_fn, patience, etc., are also not used by this demo.\n")

        # --- 1. Data Preparation for Demo (Toy Sample from provided X_train, y_train) ---
        num_demo_instances = 7
        
        if X_train is None or y_train is None or X_train.shape[0] == 0 or y_train.shape[0] == 0:
            print("ERROR: X_train or y_train is None or empty. Cannot run demo.")
            return model, None
            
        if X_train.shape[0] < num_demo_instances:
            print(f"WARNING: X_train has {X_train.shape[0]} instances, fewer than the demo's requested {num_demo_instances}. Using all available instances.")
            num_demo_instances = X_train.shape[0]
        
        if num_demo_instances == 0: # Should be caught by previous check, but as a safeguard
            print("ERROR: No data instances available in X_train for the demo.")
            return model, None

        unique_labels = list(set(y_train))  # Get unique labels

        # Select first occurrence of each label until we have 3 different ones
        selected_indices = []
        selected_labels = []

        y_train_np = y_train.to_numpy()

        for label in unique_labels:
            if len(selected_labels) < num_demo_instances:  # Continue until 3 different labels are found
                idx = torch.where(torch.tensor(y_train_np == label))[0][0] 
                selected_indices.append(idx.item())
                selected_labels.append(label)

        X_sample = X_train[selected_indices]
        y_sample_labels = y_train[selected_indices]

        num_classes = y_train.nunique()
        
        print(f"INFO: Inferred num_classes for demo: {num_classes}")

        print("--- Demo Sample Data ---")
        print(f"X_sample shape: {X_sample.shape}")
        print(f"Original y_sample_labels (indices):\n{y_sample_labels}")

        # --- 3. Loss Function and Optimizer ---
        criterion = nn.CrossEntropyLoss()
        # Ensure model parameters are available for optimizer
        try:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        except ValueError as e:
             print(f"ERROR: Could not initialize optimizer. Ensure the model has parameters. Original error: {e}")
             return model, None


        # --- 4. Training Loop (fixed 10 epochs for demo) ---
        demo_epochs = 5
        print(f"--- Starting DEMO Training for {demo_epochs} epochs (lr={lr}) ---")

        X_sample = torch.tensor(X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample, dtype=torch.float32).to(device)
        # y_sample_labels = torch.tensor(y_sample_labels).long() 
        # y_sample_labels = torch.tensor(y_sample_labels).unsqueeze(1).float()
        y_sample_labels = torch.tensor(y_sample_labels).long()

        # print(criterion)
    
        final_loss_val = None
        for epoch in range(demo_epochs):
            model.train() # Set model to training mode
            optimizer.zero_grad()
            
            outputs_logits = model(X_sample)
            
            loss = criterion(outputs_logits, y_sample_labels)
            
            loss.backward()
            optimizer.step()
            final_loss_val = loss.item()

            # --- 5. Visualization ---
            model.eval() # Set model to evaluation mode for visualization
            with torch.no_grad():
                current_logits_np = outputs_logits.cpu().numpy()
                log_softmax_values = torch.log_softmax(outputs_logits, dim=1)
                current_log_softmax_np = log_softmax_values.cpu().numpy()
                y_actual_np = y_sample_labels.cpu().numpy()

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7)) # Increased width slightly
                bar_width = 0.20 
                indices = np.arange(num_classes)

                # Plot 1: Raw Logits
                plotted_true_marker_legend_ax1 = False
                for i in range(num_demo_instances):
                    offset = (i - (num_demo_instances - 1) / 2) * bar_width
                    ax1.bar(indices + offset, current_logits_np[i, :], width=bar_width, label=f'Sample {i} Logits')
                    for c_idx in range(num_classes):
                        # print(y_actual_np[i])
                        if y_actual_np[i] == c_idx:
                            x_pos = indices[c_idx] + offset
                            y_pos = current_logits_np[i, c_idx]
                            label_marker = None
                            if not plotted_true_marker_legend_ax1:
                                label_marker = "Actual Class (Target=1)"
                                plotted_true_marker_legend_ax1 = True
                            ax1.scatter(x_pos, y_pos, color='red', marker='*', s=150, zorder=5, label=label_marker, edgecolors='black')
                            ax1.text(x_pos, y_pos + np.copysign(0.15, y_pos) if y_pos!=0 else 0.15, '★ Actual', color='red', ha='center', va='bottom' if y_pos >=0 else 'top', fontsize=8, fontweight='bold')
                
                ax1.set_xticks(indices)
                ax1.set_xticklabels([f'Class {j}' for j in range(num_classes)])
                ax1.set_ylabel('Logit Value')
                ax1.set_title(f'Raw Logits (Epoch {epoch+1})')
                ax1.legend(loc='best', fontsize='small')
                ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
                ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')

                # Plot 2: LogSoftmax
                plotted_true_marker_legend_ax2 = False
                for i in range(num_demo_instances):
                    offset = (i - (num_demo_instances - 1) / 2) * bar_width
                    ax2.bar(indices + offset, current_log_softmax_np[i, :], width=bar_width, label=f'Sample {i} LogSoftmax')
                    for c_idx in range(num_classes):
                        if y_actual_np[i] == c_idx:
                            x_pos = indices[c_idx] + offset
                            y_pos = current_log_softmax_np[i, c_idx]
                            label_marker = None
                            if not plotted_true_marker_legend_ax2:
                                label_marker = "Actual Class (Target=1)"
                                plotted_true_marker_legend_ax2 = True
                            ax2.scatter(x_pos, y_pos, color='darkgreen', marker='*', s=150, zorder=5, label=label_marker, edgecolors='black')
                            ax2.text(x_pos, y_pos + np.copysign(0.15, y_pos) if y_pos!=0 else 0.15, '★ Actual', color='darkgreen', ha='center', va='bottom' if y_pos >=0 else 'top', fontsize=8, fontweight='bold')

                ax2.set_xticks(indices)
                ax2.set_xticklabels([f'Class {j}' for j in range(num_classes)])
                ax2.set_ylabel('LogSoftmax Value')
                ax2.set_title(f'LogSoftmax (Epoch {epoch+1})')
                ax2.legend(loc='best', fontsize='small')
                ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
                ax2.axhline(0, color='black', linewidth=0.5, linestyle='--') # LogSoftmax values are <= 0
                
                fig.suptitle(f'Logit Processing DEMO - Epoch {epoch+1}/{demo_epochs} - Loss: {loss.item():.4f}', fontsize=15)
                plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Adjust layout for suptitle
                plt.show()

            print(f"DEMO Epoch [{epoch+1}/{demo_epochs}], Loss: {loss.item():.4f}")

        print(f"\n--- DEMO Training Complete ---")
        if final_loss_val is not None:
            print(f"Final DEMO Loss after {demo_epochs} epochs: {final_loss_val:.4f} ✨")
        return model, final_loss_val

    else: # task_type is not "multinomial_classification"
        print(f"INFO: Skipping logit processing demo because task_type is '{task_type}' (expected 'multinomial_classification').")
        # Potentially, original