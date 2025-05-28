import numpy as np
import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, r2_score
import sklearn.metrics
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import xgboost as xgb 



def evaluate_feature_feedback(
    X_train,
    y_train,
    X_val,
    y_val,
    show_table=True,
    drop_threshold=0.005,
    auto_drop=True,
    show_model_perf=True,
    save_csv_path=None,
    auto_tune_threshold=False,
    n_trials=10,
    min_thresh=0.0001,
    max_thresh=0.02
):
    """
    Trains RandomForest and computes:
    - Feature importance, mutual info, Pearson corr
    - Suggests low-importance features to drop
    - Optionally searches for best drop threshold

    Returns:
        dict: feedback_df, low_info_features, (cleaned X/y), model scores
    """
    is_regression = pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 10
    Model = RandomForestRegressor if is_regression else RandomForestClassifier
    XGB_Model = xgb.XGBRegressor if is_regression else xgb.XGBClassifier 

    metric_func = r2_score if is_regression else accuracy_score
    metric_name = "R²" if is_regression else "Accuracy"

    # Initial model
    rf = Model(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    score_before = metric_func(y_val, y_pred)
    print(f"✅ RandomForest {metric_name} (original): {score_before:.4f}")

    # 2. XGBoost Model
    print("\n--- XGBoost ---")
    xgb_eval_metric = 'rmse' if is_regression else 'logloss'
    if not is_regression and len(y_train.unique()) > 2: # Check if multi-class classification
        xgb_eval_metric = 'mlogloss'

    xgb_model_instance = XGB_Model(
        n_estimators=100,
        random_state=42,
        eval_metric=xgb_eval_metric,
        n_jobs=-1 
    )
    y_train_xgb = y_train.astype(int) if y_train.dtype == 'bool' else y_train
    y_val_xgb = y_val.astype(int) if y_val.dtype == 'bool' else y_val
    try:
        xgb_model_instance.fit(X_train, y_train_xgb) # Use y_train_xgb
        y_pred_xgb_proba = None
        if not is_regression:
            y_pred_xgb_proba = xgb_model_instance.predict_proba(X_val) # Get probabilities for more detailed eval if needed
        y_pred_xgb = xgb_model_instance.predict(X_val)
        score_xgb = metric_func(y_val_xgb, y_pred_xgb) # Use y_val_xgb
        print(f"✅ XGBoost {metric_name}: {score_xgb:.4f}")
    except xgb.core.XGBoostError as e:
        print(f"❌ XGBoost Error: {e}")
        print("   This might be due to string labels in y_train for classification. Ensure y_train is numerically encoded (0 to n_classes-1).")
        print(f"   y_train unique values: {y_train.unique()}")
    except Exception as e:
        print(f"❌ An unexpected error occurred with XGBoost: {e}")


    # Feature stats
    feature_names = X_train.columns
    importances = rf.feature_importances_

    if is_regression:
        mi_scores = mutual_info_regression(X_train, y_train, discrete_features=False)
    else:
        mi_scores = mutual_info_classif(X_train, y_train)

    correlations = [
        np.corrcoef(X_train[col], y_train)[0, 1] if np.std(X_train[col]) > 0 else np.nan
        for col in feature_names
    ]

    feedback_df = pd.DataFrame({
        "feature": feature_names,
        "rf_importance": importances,
        "mutual_info": mi_scores,
        "pearson_corr": correlations
    }).sort_values("rf_importance", ascending=False)

    def run_with_threshold(thresh):
        low_info_features = feedback_df[feedback_df["rf_importance"] < thresh]["feature"].tolist()
        X_train_clean = X_train.drop(columns=low_info_features, errors='ignore')
        X_val_clean = X_val.drop(columns=low_info_features, errors='ignore')
        rf_clean = Model(n_estimators=100, random_state=42)
        rf_clean.fit(X_train_clean, y_train)
        y_pred_clean = rf_clean.predict(X_val_clean)
        score_after = metric_func(y_val, y_pred_clean)
        return {
            "drop_threshold": thresh,
            "score_after": score_after,
            "dropped_features": low_info_features,
            "X_train_clean": X_train_clean,
            "X_val_clean": X_val_clean
        }

    best_result = None

    if auto_tune_threshold:
        print(f"\n🔍 Random search for best drop_threshold ({n_trials} trials)...")
        trial_results = []
        for _ in range(n_trials):
            trial_thresh = round(random.uniform(min_thresh, max_thresh), 5)
            trial_result = run_with_threshold(trial_thresh)
            trial_results.append(trial_result)

        best_result = max(trial_results, key=lambda x: x["score_after"])
        drop_threshold = best_result["drop_threshold"]

        print(f"🏆 Best threshold: {drop_threshold} → score: {best_result['score_after']:.4f} "
              f"({len(best_result['dropped_features'])} features dropped)")
    else:
        best_result = run_with_threshold(drop_threshold)

    if show_table:
        print("\n📊 Top 10 Features by RF Importance:")
        display(feedback_df.head(10))
        print("\n🧹 Least Informative Features (below threshold):")
        print(best_result["dropped_features"])
        display(feedback_df.tail(10))

    if show_model_perf:
        delta = best_result["score_after"] - score_before
        print(f"\n🔁 RandomForest {metric_name} after dropping: {best_result['score_after']:.4f}")
        print(f"{'🟢 Improved' if delta > 0 else '🔴 Degraded' if delta < 0 else '⚪ No Change'} by {delta:.4f}")

    if save_csv_path:
        feedback_df.to_csv(save_csv_path, index=False)
        print(f"📁 Feature feedback saved to: {save_csv_path}")

    return {
        "feedback_df": feedback_df,
        "low_info_features": best_result["dropped_features"],
        "X_train_clean": best_result["X_train_clean"] if auto_drop else X_train,
        "X_val_clean": best_result["X_val_clean"] if auto_drop else X_val,
        "model_score_before": score_before,
        "model_score_after": best_result["score_after"],
        "drop_threshold": drop_threshold
    }



def display_system_memory_info(context_message="Current system memory"):
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
        print("INFO: `psutil` library not found. Cannot display current available system memory.")
        print("run: pip install psutil")
        print("-" * 30)
    if PSUTIL_AVAILABLE:
        mem_info = psutil.virtual_memory()
        available_memory_mb = mem_info.available / (1024 * 1024)
        available_memory_gb = available_memory_mb / 1024
        total_memory_mb = mem_info.total / (1024 * 1024)
        total_memory_gb = total_memory_mb / 1024
        print(f"\n--- {context_message} ---")
        print(f"  System Total RAM:     {total_memory_mb:.2f} MB ({total_memory_gb:.2f} GB)")
        print(f"  System Available RAM: {available_memory_mb:.2f} MB ({available_memory_gb:.2f} GB)")
        print(f"  System RAM Used (%):  {mem_info.percent}%")
        print("--------------------------------------")
    else:
        pass



def kl_divergence(y_true, y_pred):
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Normalize and clip
    y_true = np.clip(y_true / np.sum(y_true, axis=1, keepdims=True), eps, 1 - eps)
    y_pred = np.clip(y_pred / np.sum(y_pred, axis=1, keepdims=True), eps, 1 - eps)
    return np.mean([entropy(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])

def js_divergence(y_true, y_pred):
    # Jensen-Shannon is symmetric and bounded
    eps = 1e-9
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = np.clip(y_true / np.sum(y_true, axis=1, keepdims=True), eps, 1 - eps)
    y_pred = np.clip(y_pred / np.sum(y_pred, axis=1, keepdims=True), eps, 1 - eps)
    return np.mean([jensenshannon(y_t, y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)])  # squared = actual JS div

def cross_entropy_loss(y_true, y_pred):
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred / np.sum(y_pred, axis=1, keepdims=True), eps, 1 - eps) # Ensure y_pred is normalized and clipped
    return -np.mean([np.sum(y_t * np.log(y_p)) for y_t, y_p in zip(y_true, y_pred)])

def l2_loss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean([np.sum((y_t - y_p) ** 2) for y_t, y_p in zip(y_true, y_pred)])

def l1_loss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean([np.sum(np.abs(y_t - y_p)) for y_t, y_p in zip(y_true, y_pred)])

def earth_movers_distance(y_true, y_pred, bin_centers=None):
    # This expects true and pred to be [n_samples, n_bins], and bin_centers to be [n_bins]
    if bin_centers is None:
        raise ValueError("Must provide bin_centers for Earth Mover's Distance when task_type is 'prob_vector'.")
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Normalize distributions for EMD
    y_true = y_true / np.sum(y_true, axis=1, keepdims=True)
    y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)

    emds = []
    for yt, yp in zip(y_true, y_pred):
        emds.append(wasserstein_distance(bin_centers, bin_centers, u_weights=yt, v_weights=yp))
    return np.mean(emds)

# Map of string -> (metric name, metric function)
# Note: EMD needs bin_centers, so its function call will be handled specially
metric_lookup = {
    "kl_divergence": ("KL Divergence", kl_divergence),
    "js_divergence": ("Jensen-Shannon Divergence", js_divergence),
    "cross_entropy": ("Cross-Entropy", cross_entropy_loss),
    "l2": ("L2 Loss", l2_loss),
    "l1": ("L1 Loss", l1_loss),
    "earth_movers_distance": ("Earth Mover's Distance", earth_movers_distance), # Added EMD
}


def evaluate_model(y_true, y_pred, task_type, eval_metric_name=None, eval_metric_fn=None, bin_centers=None):
    """
    Evaluates a model's predictions against true values based on the task type.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        task_type (str): Type of machine learning task
                         ("binary_classification", "multiclass_classification",
                          "regression", "prob_vector").
        eval_metric_name (str, optional): Name of the custom evaluation metric.
                                          Required for "prob_vector" task_type.
        eval_metric_fn (callable, optional): Custom evaluation metric function.
                                            Required for "prob_vector" task_type.
        bin_centers (array-like, optional): Centers of the bins, required for
                                            "earth_movers_distance" metric in "prob_vector" task.

    Returns:
        dict: A dictionary of evaluation metrics.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        mean_absolute_error, mean_squared_error, r2_score
    )

    metrics = {}

    if task_type in ["binary_classification", "multiclass_classification"]:
        # Ensure y_pred is in the correct format (e.g., class labels, not probabilities)
        # if the model output probabilities, you might need to convert them to class labels
        # e.g., y_pred_labels = np.argmax(y_pred_probs, axis=1) if y_pred is probabilities
        # For simplicity, assuming y_pred here are already class labels.

        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # ROC AUC only for binary
        if len(np.unique(y_true)) == 2:
            try:
                # For ROC AUC, y_pred should be probabilities for the positive class
                # If y_pred is class labels, this will likely fail or give misleading results.
                # Assuming y_pred is already probabilities if this is intended.
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
            except ValueError: # Catch error if y_pred is not probabilities
                metrics["roc_auc"] = None
                print("Warning: ROC AUC requires probability estimates for binary classification.")

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
            if label not in ["accuracy", "macro avg", "weighted avg"] # Exclude overall metrics
        }
        metrics["per_class_metrics"] = class_metrics

    elif task_type == "regression":
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        metrics["r2"] = r2_score(y_true, y_pred)

        # === Bin target into quantiles and evaluate per bin
        # Ensure y_true is a Series for pd.qcut
        if not isinstance(y_true, pd.Series):
            y_true_series = pd.Series(y_true)
        else:
            y_true_series = y_true

        bins = pd.qcut(y_true_series, q=5, duplicates='drop')
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

    elif task_type == "prob_vector":
        if eval_metric_fn is None or eval_metric_name is None:
            raise ValueError(
                "For 'prob_vector' task_type, 'eval_metric_fn' and 'eval_metric_name' "
                "must be provided to evaluate_model."
            )

        # Ensure y_true and y_pred are numpy arrays for consistency with metric functions
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

        # Handle Earth Mover's Distance specifically as it requires bin_centers
        if eval_metric_name == "Earth Mover's Distance":
            if bin_centers is None:
                raise ValueError(
                    "For 'Earth Mover's Distance' metric, 'bin_centers' must be provided "
                    "to evaluate_model when task_type is 'prob_vector'."
                )
            metrics[eval_metric_name.lower().replace(' ', '_')] = eval_metric_fn(y_true_np, y_pred_np, bin_centers=bin_centers)
        else:
            metrics[eval_metric_name.lower().replace(' ', '_')] = eval_metric_fn(y_true_np, y_pred_np)

        # Optionally, you might want to include L1/L2 as general distance metrics
        # even if not the primary eval_metric_fn
        metrics["l1_loss_overall"] = l1_loss(y_true_np, y_pred_np)
        metrics["l2_loss_overall"] = l2_loss(y_true_np, y_pred_np)


    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return metrics

