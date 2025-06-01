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
import math
import torch.nn.functional as F

import matplotlib.pyplot as plt

from collections import defaultdict


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

def track_label_losses(outputs_logits, targets, device):
    """
    Computes and tracks the loss for each individual label.
    
    Args:
        outputs_logits (torch.Tensor): Output logits from the model.
        targets (torch.Tensor): Target labels.
        device (torch.device): The device tensors are on.
        
    Returns:
        dict: Dictionary where keys are labels and values are lists of loss values for each label.
    """
    # Initialize a dictionary to store losses for each label
    label_losses = {}
    
    # Ensure targets is on the same device as outputs_logits for comparison
    targets = targets.to(device)

    # Compute all losses at once using CrossEntropyLoss with no reduction
    criterion_here = nn.CrossEntropyLoss(reduction='none')
    per_sample_losses = criterion_here(outputs_logits, targets)
    
    # Get unique labels present in the current batch
    unique_labels_in_batch = torch.unique(targets).cpu().numpy()

    for label_val in unique_labels_in_batch:
        # Create a tensor for the current label_val, ensuring it's on the correct device and dtype
        label_tensor = torch.tensor(label_val, device=targets.device, dtype=targets.dtype)
        # Find indices where the target matches the current label_val
        indices_for_label = (targets == label_tensor).nonzero(as_tuple=True)[0]
       
        # If there are samples for this label, compute the average loss
        if len(indices_for_label) > 0:
            avg_loss_for_label = per_sample_losses[indices_for_label].mean().item()
            label_losses[int(label_val)] = avg_loss_for_label
        else:
            # Handle cases where a label might appear in unique_labels but have no samples (should be rare with torch.unique)
            label_losses[int(label_val)] = np.nan 
            
    return label_losses


def plot_label_losses(history_train_losses_with_epochs, history_val_losses_with_epochs=None):
    """
    Plots the per-label loss progression over epochs for training and validation sets.

    Args:
        history_train_losses_with_epochs (list): A list of dictionaries for training, 
                                                 where each dictionary is {'epoch': epoch_num, 'losses': {label: loss_val}}.
        history_val_losses_with_epochs (list, optional): A list of dictionaries for validation,
                                                          structured similarly to training history. Defaults to None.
    """
    if not history_train_losses_with_epochs:
        print("INFO: No per-label training loss history to plot.")
        return

    # Determine all unique labels and epochs from both histories
    all_labels = set()
    for item in history_train_losses_with_epochs:
        all_labels.update(item['losses'].keys())
    if history_val_losses_with_epochs:
        for item in history_val_losses_with_epochs:
            all_labels.update(item['losses'].keys())
    
    if not all_labels:
        print("INFO: No labels found in the loss history.")
        return
        
    unique_labels_to_plot = sorted(list(all_labels))

    plot_data_train = {label: [] for label in unique_labels_to_plot}
    plot_data_val = {label: [] for label in unique_labels_to_plot}

    recorded_epochs_train = {item['epoch']: item['losses'] for item in history_train_losses_with_epochs}
    recorded_epochs_val = {}
    if history_val_losses_with_epochs:
        recorded_epochs_val = {item['epoch']: item['losses'] for item in history_val_losses_with_epochs}

    all_epochs_numeric = sorted(list(set(recorded_epochs_train.keys()) | set(recorded_epochs_val.keys())))

    if not all_epochs_numeric:
        print("INFO: No epoch data found to plot.")
        return

    for epoch_num in all_epochs_numeric:
        train_losses_at_epoch = recorded_epochs_train.get(epoch_num, {})
        val_losses_at_epoch = recorded_epochs_val.get(epoch_num, {})
        for label_key in unique_labels_to_plot:
            plot_data_train[label_key].append(train_losses_at_epoch.get(label_key, np.nan))
            plot_data_val[label_key].append(val_losses_at_epoch.get(label_key, np.nan))
    
    plt.figure(figsize=(14, 8)) # Increased figure size for better readability
    
    has_plotted_train = False
    has_plotted_val = False

    for label_val in unique_labels_to_plot:
        losses_train_at_recorded_epochs = plot_data_train[label_val]
        if not all(np.isnan(l) for l in losses_train_at_recorded_epochs):
            plt.plot(all_epochs_numeric, losses_train_at_recorded_epochs, marker='o', linestyle='-', label=f'Train Loss Label {label_val}')
            has_plotted_train = True
        
        if history_val_losses_with_epochs:
            losses_val_at_recorded_epochs = plot_data_val[label_val]
            if not all(np.isnan(l) for l in losses_val_at_recorded_epochs):
                plt.plot(all_epochs_numeric, losses_val_at_recorded_epochs, marker='x', linestyle='--', label=f'Val Loss Label {label_val}')
                has_plotted_val = True
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss per Label')
    title_parts = ["Per-Label Loss Over Demo Epochs"]
    if has_plotted_train and has_plotted_val:
        title_parts.append("(Train & Validation)")
    elif has_plotted_train:
        title_parts.append("(Training)")
    title_parts.append("(Tracked Periodically)")
    plt.title(' '.join(title_parts))

    if all_epochs_numeric:
        # Ensure x-ticks are sensible, especially if many epochs are tracked
        if len(all_epochs_numeric) > 20: # Heuristic for too many ticks
             tick_indices = np.linspace(0, len(all_epochs_numeric) - 1, num=10, dtype=int)
             plt.xticks([all_epochs_numeric[i] for i in tick_indices])
        else:
             plt.xticks(all_epochs_numeric)
    
    if has_plotted_train or has_plotted_val:
        plt.legend(loc='best', fontsize='small')
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


def train_nn_model_with_demo(
    model, X_train, y_train, X_val, y_val,
    epochs=10,
    lr=0.005,
    task_type="multinomial_classification", 
    eval_metric_name="val_metric", # Not used in this demo
    eval_metric_fn=None, # Not used in this demo
    with_class_weight=False,
    bin_centers=None, # Not used in this demo
    patience=5 # Not used in this demo
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if task_type == "multinomial_classification":
        print(f"INFO: Running DEMO for task_type: {task_type} on a subset of training data.")
        print("INFO: Validation data will be used for visualization if provided.")
        print("INFO: Parameters eval_metric_fn, patience, etc., are not used by this demo.\n")
        
        if X_train is None or y_train is None or X_train.shape[0] == 0 or \
           (hasattr(y_train, 'shape') and y_train.shape[0] == 0) or \
           (isinstance(y_train, list) and not y_train):
            print("ERROR: X_train or y_train is None or empty. Cannot run demo.")
            return model, None
            
        # --- Prepare Training Data ---
        X_sample_data = X_train
        y_sample_labels_data = y_train

        X_sample_tensor = torch.tensor(X_sample_data.values if isinstance(X_sample_data, pd.DataFrame) else X_sample_data, dtype=torch.float32).to(device)
        
        y_s_numpy = y_sample_labels_data.to_numpy() if isinstance(y_sample_labels_data, (pd.Series, pd.DataFrame)) else np.array(y_sample_labels_data)
        y_sample_labels_tensor = torch.tensor(y_s_numpy, dtype=torch.long).to(device)

        num_classes_train = len(np.unique(y_s_numpy)) 
        print(f"INFO: Inferred num_classes for demo from training data: {num_classes_train}")
        print(f"INFO: Demo will run on {X_sample_tensor.shape[0]} training instances.")
        
        # --- Prepare Validation Data (if available) ---
        X_val_tensor, y_val_tensor, y_val_numpy = None, None, None
        if X_val is not None and y_val is not None:
            if X_val.shape[0] > 0 and ((hasattr(y_val, 'shape') and y_val.shape[0] > 0) or (isinstance(y_val, list) and y_val)):
                print(f"INFO: Validation data provided with {X_val.shape[0]} instances.")
                X_val_tensor = torch.tensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val, dtype=torch.float32).to(device)
                y_val_numpy = y_val.to_numpy() if isinstance(y_val, (pd.Series, pd.DataFrame)) else np.array(y_val)
                y_val_tensor = torch.tensor(y_val_numpy, dtype=torch.long).to(device)
            else:
                print("INFO: Validation data (X_val or y_val) is empty, skipping validation metrics.")
        else:
            print("INFO: No validation data (X_val or y_val is None), skipping validation metrics.")

        print(f"INFO: Per-epoch visualizations and loss tracking will occur every 5 epochs.")

        # Compute class frequencies and weights
        y_train_np = np.array(y_train)
        unique_labels, counts = np.unique(y_train_np, return_counts=True)
        num_classes = len(unique_labels)
        class_weights = torch.FloatTensor([1.0 / count for count in counts]).to(device)
        
        if with_class_weight:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss() 
        try:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        except ValueError as e:
             print(f"ERROR: Could not initialize optimizer. Ensure the model has parameters. Original error: {e}")
             return model, None

        # --- Training Loop ---
        demo_epochs = epochs
        print(f"--- Starting DEMO Training for {demo_epochs} epochs (lr={lr}) ---")

        final_loss_val_train = None
        history_label_losses_train = [] 
        history_label_losses_val = []

        for epoch in range(demo_epochs):
            model.train() 
            optimizer.zero_grad()
            
            outputs_logits_train_batch = model(X_sample_tensor)
            loss_train_batch = criterion(outputs_logits_train_batch, y_sample_labels_tensor)
            
            loss_train_batch.backward()
            optimizer.step()
            final_loss_val_train = loss_train_batch.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"\n--- Generating visualizations and tracking losses for Epoch {epoch + 1} ---")
                model.eval() 
                val_loss_for_title = None

                with torch.no_grad():
                    # --- Track per-label losses (TRAIN) ---
                    # Re-evaluate on training data in eval mode for consistent metric calculation
                    current_outputs_logits_train = model(X_sample_tensor)
                    current_epoch_label_losses_train = track_label_losses(current_outputs_logits_train.detach(), y_sample_labels_tensor, device)
                    history_label_losses_train.append({'epoch': epoch + 1, 'losses': current_epoch_label_losses_train})

                    # --- Calculate overall validation loss and track per-label losses (VAL) if data exists ---
                    if X_val_tensor is not None and y_val_tensor is not None:
                        current_outputs_logits_val = model(X_val_tensor)
                        loss_val_batch = criterion(current_outputs_logits_val, y_val_tensor)
                        val_loss_for_title = loss_val_batch.item()
                        
                        current_epoch_label_losses_val = track_label_losses(current_outputs_logits_val.detach(), y_val_tensor, device)
                        history_label_losses_val.append({'epoch': epoch + 1, 'losses': current_epoch_label_losses_val})

                    # --- Prepare data for bar plots ---
                    # Training metrics
                    softmax_values_train = torch.softmax(current_outputs_logits_train, dim=1)
                    log_softmax_values_train = torch.log_softmax(current_outputs_logits_train, dim=1)
                    
                    current_logits_train_np = current_outputs_logits_train.cpu().numpy()
                    current_softmax_train_np = softmax_values_train.cpu().numpy()
                    current_log_softmax_train_np = log_softmax_values_train.cpu().numpy()
                    current_nll_train_np = -current_log_softmax_train_np 
                    y_actual_train_np = y_sample_labels_tensor.cpu().numpy()
                    unique_labels_train_batch = np.unique(y_actual_train_np)

                    # Validation metrics (if available)
                    current_logits_val_np, current_softmax_val_np, current_log_softmax_val_np, current_nll_val_np = [None]*4
                    y_actual_val_np, unique_labels_val_batch = None, np.array([])
                    if X_val_tensor is not None and y_val_tensor is not None:
                        softmax_values_val = torch.softmax(current_outputs_logits_val, dim=1)
                        log_softmax_values_val = torch.log_softmax(current_outputs_logits_val, dim=1)

                        current_logits_val_np = current_outputs_logits_val.cpu().numpy()
                        current_softmax_val_np = softmax_values_val.cpu().numpy()
                        current_log_softmax_val_np = log_softmax_values_val.cpu().numpy()
                        current_nll_val_np = -current_log_softmax_val_np
                        y_actual_val_np = y_val_tensor.cpu().numpy()
                        unique_labels_val_batch = np.unique(y_actual_val_np)
                    
                    # Combine unique labels for plotting axes
                    combined_unique_labels_for_plot = sorted(list(set(unique_labels_train_batch) | set(unique_labels_val_batch)))
                    if not combined_unique_labels_for_plot: # Fallback if somehow both are empty
                        combined_unique_labels_for_plot = sorted(list(unique_labels_train_batch))


                    # Calculate average metrics per label
                    metrics_train = {'logits': [], 'softmax': [], 'log_softmax': [], 'nll': []}
                    metrics_val = {'logits': [], 'softmax': [], 'log_softmax': [], 'nll': []}

                    for label_val in combined_unique_labels_for_plot:
                        # Training
                        indices_train = np.where(y_actual_train_np == label_val)[0]
                        if len(indices_train) > 0:
                            metrics_train['logits'].append(np.mean(current_logits_train_np[indices_train, label_val]))
                            metrics_train['softmax'].append(np.mean(current_softmax_train_np[indices_train, label_val]))
                            metrics_train['log_softmax'].append(np.mean(current_log_softmax_train_np[indices_train, label_val]))
                            metrics_train['nll'].append(np.mean(current_nll_train_np[indices_train, label_val]))
                        else:
                            for k in metrics_train: metrics_train[k].append(np.nan)
                        
                        # Validation
                        if y_actual_val_np is not None:
                            indices_val = np.where(y_actual_val_np == label_val)[0]
                            if len(indices_val) > 0:
                                metrics_val['logits'].append(np.mean(current_logits_val_np[indices_val, label_val]))
                                metrics_val['softmax'].append(np.mean(current_softmax_val_np[indices_val, label_val]))
                                metrics_val['log_softmax'].append(np.mean(current_log_softmax_val_np[indices_val, label_val]))
                                metrics_val['nll'].append(np.mean(current_nll_val_np[indices_val, label_val]))
                            else:
                                for k in metrics_val: metrics_val[k].append(np.nan)
                        else: # No validation data, fill with NaNs
                             for k in metrics_val: metrics_val[k].append(np.nan)


                # --- Plotting Bar Charts ---
                if not combined_unique_labels_for_plot:
                    print("INFO: No labels to plot for bar charts in this epoch.")
                else:
                    plot_indices = np.arange(len(combined_unique_labels_for_plot))
                    xticklabels = [f'Class {lbl}' for lbl in combined_unique_labels_for_plot]
                    bar_width = 0.35 

                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 8)) 

                    # Plot 1: Average Raw Logits
                    ax1.bar(plot_indices - bar_width/2, metrics_train['logits'], width=bar_width, color='skyblue', label='Train Avg. Logit')
                    if X_val_tensor is not None: ax1.bar(plot_indices + bar_width/2, metrics_val['logits'], width=bar_width, color='lightsteelblue', label='Val Avg. Logit')
                    ax1.set_ylabel('Average Logit Value')
                    ax1.set_title('Avg. Raw Logits of True Class')
                    ax1.set_xticks(plot_indices)
                    ax1.set_xticklabels(xticklabels, rotation=45, ha="right")
                    ax1.legend(loc='best', fontsize='small')
                    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
                    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')

                    # Plot 2: Average Softmax Probability
                    ax2.bar(plot_indices - bar_width/2, metrics_train['softmax'], width=bar_width, color='lightcoral', label='Train Avg. Softmax')
                    if X_val_tensor is not None: ax2.bar(plot_indices + bar_width/2, metrics_val['softmax'], width=bar_width, color='salmon', label='Val Avg. Softmax')
                    ax2.set_ylabel('Average Softmax Probability')
                    ax2.set_title('Avg. Softmax Prob. of True Class')
                    ax2.set_xticks(plot_indices)
                    ax2.set_xticklabels(xticklabels, rotation=45, ha="right")
                    ax2.legend(loc='best', fontsize='small')
                    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
                    ax2.set_ylim(0, 1.05) 
                    # ... (reference lines for softmax)

                    # Plot 3: Average LogSoftmax
                    ax3.bar(plot_indices - bar_width/2, metrics_train['log_softmax'], width=bar_width, color='lightgreen', label='Train Avg. LogSoftmax')
                    if X_val_tensor is not None: ax3.bar(plot_indices + bar_width/2, metrics_val['log_softmax'], width=bar_width, color='darkseagreen', label='Val Avg. LogSoftmax')
                    ax3.set_ylabel('Average LogSoftmax Value')
                    ax3.set_title('Avg. LogSoftmax of True Class')
                    ax3.set_xticks(plot_indices)
                    ax3.set_xticklabels(xticklabels, rotation=45, ha="right")
                    ax3.legend(loc='best', fontsize='small')
                    ax3.grid(True, axis='y', linestyle=':', alpha=0.6)
                    # ... (reference lines for log_softmax)
                    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')


                    # Plot 4: Average NLL
                    ax4.bar(plot_indices - bar_width/2, metrics_train['nll'], width=bar_width, color='gold', label='Train Avg. NLL')
                    if X_val_tensor is not None: ax4.bar(plot_indices + bar_width/2, metrics_val['nll'], width=bar_width, color='khaki', label='Val Avg. NLL')
                    ax4.set_ylabel('Average NLL Value')
                    ax4.set_title('Avg. NLL of True Class')
                    ax4.set_xticks(plot_indices)
                    ax4.set_xticklabels(xticklabels, rotation=45, ha="right")
                    ax4.legend(loc='best', fontsize='small')
                    ax4.grid(True, axis='y', linestyle=':', alpha=0.6)
                    # ... (reference lines for NLL)
                    ax4.axhline(0, color='black', linewidth=0.5, linestyle='--')

                    title_str = f'DEMO - Epoch {epoch+1}/{demo_epochs} - Train Batch Loss: {loss_train_batch.item():.4f}'
                    if val_loss_for_title is not None:
                        title_str += f' - Val Batch Loss: {val_loss_for_title:.4f}'
                    title_str += '\n(Metrics Averaged per True Class Label)'
                    fig.suptitle(title_str, fontsize=14) # Reduced font size slightly
                    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjusted rect for suptitle
                    plt.show()

                print(f"DEMO Epoch [{epoch+1}/{demo_epochs}], Train Batch Loss: {loss_train_batch.item():.4f}")
                if val_loss_for_title is not None:
                     print(f"DEMO Epoch [{epoch+1}/{demo_epochs}], Val Batch Loss: {val_loss_for_title:.4f}")
                
                # print(f"Per-label average NLL (Train - from tracking): {current_epoch_label_losses_train}")
                # if X_val_tensor is not None and y_val_tensor is not None and history_label_losses_val:
                #      print(f"Per-label average NLL (Val - from tracking): {history_label_losses_val[-1]['losses']}")

            model.train() # Ensure model is back in training mode for the next iteration

        # After the loop, plot the per-label loss history for train and val
        plot_label_losses(history_label_losses_train, history_label_losses_val if history_label_losses_val else None)

        print(f"\n--- DEMO Training Complete ---")
        if final_loss_val_train is not None:
            print(f"Final DEMO Training Batch Loss after {demo_epochs} epochs: {final_loss_val_train:.4f} ✨")
        
        # Return model and final training loss. Could also return validation loss if needed.
        return model, final_loss_val_train

    else: 
        print(f"INFO: Skipping logit processing demo because task_type is '{task_type}' (expected 'multinomial_classification').")
        return model, None


