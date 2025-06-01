import numpy as np
import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc)
import sklearn.metrics

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import xgboost as xgb 

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mticker



# Probability-based metrics
def kl_divergence(y_true, y_pred):
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    y_true = np.clip(y_true, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)
    return np.mean([entropy(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])

def js_divergence(y_true, y_pred):
    # Jensen-Shannon is symmetric and bounded
    return np.mean([jensenshannon(y_t, y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)])  # squared = actual JS div

def cross_entropy_loss(y_true, y_pred):
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1)
    return -np.mean([np.sum(y_t * np.log(y_p)) for y_t, y_p in zip(y_true, y_pred)])

def l2_loss(y_true, y_pred):
    return np.mean([np.sum((y_t - y_p) ** 2) for y_t, y_p in zip(y_true, y_pred)])

def l1_loss(y_true, y_pred):
    return np.mean([np.sum(np.abs(y_t - y_p)) for y_t, y_p in zip(y_true, y_pred)])

def earth_movers_distance(y_true, y_pred, bin_centers=None):
    # This expects true and pred to be [n_samples, n_bins], and bin_centers to be [n_bins]
    if bin_centers is None:
        raise ValueError("Must provide bin_centers for Earth Mover's Distance")
    # Compute EMD for each pair
    emds = []
    for yt, yp in zip(y_true, y_pred):
        # Convert prob dist to sample dist over bin centers
        emds.append(wasserstein_distance(bin_centers, bin_centers, u_weights=yt, v_weights=yp))
    return np.mean(emds)


###########################
metric_lookup = {
    "kl_divergence": ("KL Divergence", kl_divergence),
    "js_divergence": ("Jensen-Shannon Divergence", js_divergence),
    "cross_entropy": ("Cross-Entropy", cross_entropy_loss),
    "l2": ("L2 Loss", l2_loss),
    "l1": ("L1 Loss", l1_loss),
    "accuracy": ("accuracy", sklearn.metrics.accuracy_score),
    "f1": ("f1_score", lambda y_true, y_pred: sklearn.metrics.f1_score(y_true, y_pred, average="weighted")),
    "precision": ("precision", lambda y_true, y_pred: sklearn.metrics.precision_score(y_true, y_pred, average="weighted")),
    "recall": ("recall", lambda y_true, y_pred: sklearn.metrics.recall_score(y_true, y_pred, average="weighted")),
    "mse": ("mean_squared_error", sklearn.metrics.mean_squared_error),
    "mae": ("mean_absolute_error", sklearn.metrics.mean_absolute_error),
}

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


def evaluate_model(y_true, y_pred, task_type, eval_metric_name=None, eval_metric_fn=None, bin_centers=None):
    """
    Evaluates a model's predictions against true values based on the task type.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        task_type (str): Type of machine learning task
                         ("binary_classification", "multinomial_classification",
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

    if task_type in ["binary_classification", "multinomial_classification"]:
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



def get_ev_category(ev, threshold=.005, nan_as="Neutral"):
    """Categorizes an EV value."""
    if pd.isna(ev):
        return nan_as
    if ev > threshold:
        return "Profitable"
    elif ev < -threshold:
        return "Loss"
    else:
        return "Neutral"

def get_cluster_ev_mapping(y_target_df_with_ev, agg_func='mean'):
    if 'Cluster_Assignment' not in y_target_df_with_ev.columns or 'EV' not in y_target_df_with_ev.columns:
        raise ValueError("y_target_df_with_ev must contain 'Cluster_Assignment' and 'EV' columns.")
    if agg_func == 'mean':
        return y_target_df_with_ev.groupby('Cluster_Assignment')['EV'].mean()
    elif agg_func == 'median':
        return y_target_df_with_ev.groupby('Cluster_Assignment')['EV'].median()
    else:
        raise ValueError("agg_func must be 'mean' or 'median'")



def _map_labels_to_ev(labels, cluster_ev_mapping):
    # ... (same as before)
    processed_labels = []
    for label_item in labels:
        if isinstance(label_item, np.ndarray) and label_item.ndim > 0 :
            processed_labels.append(label_item[0])
        else:
            processed_labels.append(label_item)
    if not cluster_ev_mapping.empty:
        return np.array([cluster_ev_mapping.get(label, np.nan) for label in processed_labels])
    return np.full(len(processed_labels), np.nan)

def evaluate_model_predictions(
    y_true_df,
    y_pred_dict, 
    y_target_df_with_ev,
    y_pred_proba_dict=None,
    ev_profit_threshold=0.01,
    initial_portfolio_value=100000.0,
    trade_size=3000.0,
    hold_apr=0.05,
    model_classes_ordered_list_for_proba=None # For robust proba column mapping
):
    if not isinstance(y_true_df, pd.DataFrame) or y_true_df.shape[1] != 1:
        raise ValueError("y_true_df must be a pandas DataFrame with a single column of true labels.")
    y_true = y_true_df.iloc[:, 0].values

    if y_pred_proba_dict is None:
        y_pred_proba_dict = {}
        print("⚠️ Note: Predicted probabilities (y_pred_proba_dict) not provided. Some plots/analyses will be skipped or limited.\n")

    print("="*60); print("📊 Overall Evaluation Setup"); print("="*60)
    cluster_mean_ev_mapping = None
    try:
        cluster_mean_ev_mapping = get_cluster_ev_mapping(y_target_df_with_ev, agg_func='mean')
        cluster_median_ev_mapping = get_cluster_ev_mapping(y_target_df_with_ev, agg_func='median')
        print("\n--- Mean & Median EV for each TRUE Cluster (based on y_target_df_with_ev) ---")
        ev_stats_df = pd.DataFrame({'Mean EV': cluster_mean_ev_mapping, 'Median EV': cluster_median_ev_mapping}).sort_index()
        print(ev_stats_df); print("-"*(len(str(ev_stats_df).splitlines()[0])))
    except Exception as e:
        print(f"Could not create cluster EV mappings: {e}")

    predicted_labels_set = set()
    for preds_array in y_pred_dict.values():
        for label_array in preds_array: predicted_labels_set.add(label_array[0])
    base_labels = sorted(list(np.unique(y_true)))
    all_plot_labels = sorted(list(set(base_labels) | predicted_labels_set))
    print(f"\nIdentified unique labels for evaluation range: {all_plot_labels}")

    # Determine Actual EVs for all validation samples ONCE
    actual_evs_for_portfolio = np.full(len(y_true), np.nan)
    if cluster_mean_ev_mapping is not None:
        actual_evs_for_portfolio = _map_labels_to_ev(y_true, cluster_mean_ev_mapping)
        actual_outcome_categories = np.array([get_ev_category(ev, ev_profit_threshold, nan_as="Unknown") for ev in actual_evs_for_portfolio])
    else:
        print("⚠️ actual_evs_for_portfolio and actual_outcome_categories for portfolio simulation cannot be determined as cluster_mean_ev_mapping is unavailable.")


    for model_name, y_pred_array_2d in y_pred_dict.items():
        y_pred_1d = y_pred_array_2d.ravel() # Used for some simpler metrics
        print(f"\n\n{'='*25} Evaluating: {model_name.upper()} {'='*25}")

        plt.figure(figsize=(10, 12))
        print(f"\n## 1. Overall Accuracy & Per-Cluster Performance ({model_name}) ## 🎯")
        accuracy = accuracy_score(y_true, y_pred_1d)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        plt.figure(figsize=(max(6, len(all_plot_labels)*0.9), max(5, len(all_plot_labels)*0.7)))
        cm = confusion_matrix(y_true, y_pred_1d, labels=all_plot_labels)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_plot_labels, yticklabels=all_plot_labels)
        plt.title(f'Confusion Matrix - {model_name}'); plt.ylabel('True Cluster'); plt.xlabel('Predicted Cluster'); plt.show()
        print(f"\nClassification Report ({model_name}):")
        report_labels = sorted(list(set(y_true) | set(y_pred_1d)))
        report_target_names = [f"Clust {l}" for l in report_labels]
        report = classification_report(y_true, y_pred_1d, labels=report_labels, zero_division=0, target_names=report_target_names)
        print(report)
        print("\n--- Understanding Precision & Recall for Your Clusters ---")
        print("💡 Precision (Purity of Prediction for a Cluster):")
        print("  Of all instances your model PREDICTED as a specific cluster, what % ACTUALLY belonged to it?")
        print("💡 Recall (Completeness/Sensitivity for a Cluster):")
        print("  Of all instances that ACTUALLY belonged to a specific cluster, what % did your model CORRECTLY identify?")
        print("-----------------------------------------------------------")


        # SECTION 2: ROC AUC & Curves (largely same, needs careful class alignment)
        print(f"\n## 2. ROC AUC & Curves ({model_name}) ## 📈")
        # ... (Section 2 code as before, ensure robust class alignment for proba columns) ...
        y_pred_proba = y_pred_proba_dict.get(model_name)
        if y_pred_proba is not None:
            # Determine classes_for_roc from model.classes_ if passed, else from y_true
            # This is complex to make generic without model.classes_ explicitly.
            # For this version, we'll use unique_true_labels from the current fold.
            unique_true_labels_for_roc = sorted(list(np.unique(y_true)))
            
            # Best effort to determine the effective classes for y_pred_proba columns
            # If model_classes_ordered_list_for_proba is provided, use it. Otherwise, assume based on all_plot_labels.
            effective_proba_classes = model_classes_ordered_list_for_proba if model_classes_ordered_list_for_proba is not None else all_plot_labels
            
            if y_pred_proba.shape[1] != len(effective_proba_classes) and model_classes_ordered_list_for_proba is None:
                print(f"   Warning for ROC plot '{model_name}': y_pred_proba columns ({y_pred_proba.shape[1]}) "
                      f"doesn't match assumed class order length ({len(effective_proba_classes)} derived from all_plot_labels). "
                      "ROC results might be inaccurate if column order is not as assumed (e.g., sorted unique classes).")
            
            # Check if all unique_true_labels_for_roc are present in effective_proba_classes before binarizing
            if not set(unique_true_labels_for_roc).issubset(set(effective_proba_classes)):
                print(f"   Skipping ROC AUC for {model_name}: Not all true labels for this fold are represented in the assumed order of probability columns.")
            else:
                try:
                    # Filter y_pred_proba columns to only those in unique_true_labels_for_roc and in their order
                    # This is crucial if y_pred_proba has more columns than classes in this fold.
                    proba_indices_for_roc = [effective_proba_classes.index(lbl) for lbl in unique_true_labels_for_roc if lbl in effective_proba_classes]
                    
                    if len(proba_indices_for_roc) != len(unique_true_labels_for_roc) or not proba_indices_for_roc: # not all true labels found or empty
                         print(f"   Skipping ROC AUC for {model_name}: Could not align probabilities for all true classes in this fold.")
                    else:
                        y_pred_proba_filtered_for_roc = y_pred_proba[:, proba_indices_for_roc]

                        roc_auc_ovr_macro = roc_auc_score(y_true, y_pred_proba_filtered_for_roc, multi_class='ovr', average='macro', labels=unique_true_labels_for_roc)
                        print(f"ROC AUC (OvR, macro): {roc_auc_ovr_macro:.4f}")
                        roc_auc_ovr_weighted = roc_auc_score(y_true, y_pred_proba_filtered_for_roc, multi_class='ovr', average='weighted', labels=unique_true_labels_for_roc)
                        print(f"ROC AUC (OvR, weighted): {roc_auc_ovr_weighted:.4f}")

                        y_true_binarized = label_binarize(y_true, classes=unique_true_labels_for_roc)
                        n_classes_to_plot = y_true_binarized.shape[1]
                        plt.figure(figsize=(10, 7))
                        prop_cycle = plt.rcParams['axes.prop_cycle']; colors = prop_cycle.by_key()['color']
                        for i in range(n_classes_to_plot):
                            class_label_iter = unique_true_labels_for_roc[i]
                            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_proba_filtered_for_roc[:, i])
                            roc_auc_val = auc(fpr, tpr)
                            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'ROC Cluster {class_label_iter} (area = {roc_auc_val:.2f})')
                        plt.plot([0, 1], [0, 1], 'k--', lw=2); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(f'Multiclass ROC Curves (One-vs-Rest) - {model_name}')
                        plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.show()
                except ValueError as ve: print(f"   Could not calculate ROC AUC for {model_name} (ValueError): {ve}")
                except Exception as e: print(f"   Could not calculate ROC AUC for {model_name} (Other Error): {e}")
        else: print(f"   Skipped (predicted probabilities not provided/found for this model).")
        print("-----------------------------------------------------------")


        # SECTION 3: Box Plot of Model's Assigned Probability to the True Cluster
        print(f"\n## 3. Model Confidence in True Cluster Assignments ({model_name}) ## 🤔")
        y_pred_proba = y_pred_proba_dict.get(model_name) # Ensure we re-fetch for this section
        if y_pred_proba is not None:
            # Determine mapping of class labels to y_pred_proba column indices
            # Use model_classes_ordered_list_for_proba if provided, else assume all_plot_labels order
            idx_map_classes = model_classes_ordered_list_for_proba if model_classes_ordered_list_for_proba is not None else all_plot_labels
            
            if y_pred_proba.shape[1] != len(idx_map_classes) and model_classes_ordered_list_for_proba is None:
                 print(f"   Warning for '{model_name}' Confidence plot: y_pred_proba columns ({y_pred_proba.shape[1]}) "
                       f"!= assumed class order length ({len(idx_map_classes)} from all_plot_labels). Plot may be partial/incorrect.")

            class_to_idx_map_confidence = {label: idx for idx, label in enumerate(idx_map_classes)}
            
            probs_for_true_cluster_data = {'True_Cluster': [], 'Probability_Assigned_To_True_Cluster': []}
            for i in range(len(y_true)):
                true_label = y_true[i]
                if true_label in class_to_idx_map_confidence:
                    proba_col_idx = class_to_idx_map_confidence[true_label]
                    if proba_col_idx < y_pred_proba.shape[1]: # Check if index is valid
                        probs_for_true_cluster_data['True_Cluster'].append(true_label)
                        probs_for_true_cluster_data['Probability_Assigned_To_True_Cluster'].append(y_pred_proba[i, proba_col_idx])
            
            if probs_for_true_cluster_data['True_Cluster']:
                plot_df_confidence = pd.DataFrame(probs_for_true_cluster_data)
                plt.figure(figsize=(max(8, len(all_plot_labels) * 1.2), 7))
                sns.boxplot(x='True_Cluster', y='Probability_Assigned_To_True_Cluster', data=plot_df_confidence, order=all_plot_labels, palette="Blues_r")
                plt.title(f'Model Confidence in True Cluster Assignments - {model_name}'); plt.xlabel('Actual True Cluster')
                plt.ylabel('Probability Assigned by Model to that True Cluster')
                plt.xticks(ticks=range(len(all_plot_labels)), labels=[f"Clust {l}" for l in all_plot_labels], rotation=45, ha="right")
                plt.ylim(-0.05, 1.05); plt.grid(axis='y', linestyle=':', alpha=0.7); plt.tight_layout(); plt.show()
            else: print(f"   Skipping 'Model Confidence' plot for {model_name}: No probability data mapped for true clusters.")
        else: print(f"   Skipping 'Model Confidence' plot for {model_name} (predicted probabilities not provided/found).")
        print("-----------------------------------------------------------")

        # SECTION 5: Cumulative Probability Captured by Top-N Predictions (was Section 6)
        print(f"\n## 5. Cumulative Probability Captured by Top-N Predictions ({model_name}) ## 🧐")
        y_pred_proba = y_pred_proba_dict.get(model_name) # Re-fetch
        if y_pred_proba is not None:
            n_samples, n_classes_proba = y_pred_proba.shape
            if n_classes_proba > 0:
                sorted_probas = np.sort(y_pred_proba, axis=1)[:, ::-1]
                cumulative_probas = np.cumsum(sorted_probas, axis=1)
                plot_data_pareto = []
                for i in range(n_classes_proba):
                    for prob_val in cumulative_probas[:, i]: plot_data_pareto.append({'Top_N_Considered': f'Top-{i+1}', 'Cumulative_Probability': prob_val})
                if plot_data_pareto:
                    df_plot_pareto = pd.DataFrame(plot_data_pareto)
                    category_order = [f'Top-{i+1}' for i in range(n_classes_proba)]
                    plt.figure(figsize=(max(8, n_classes_proba * 1.5), 6))
                    sns.boxplot(x='Top_N_Considered', y='Cumulative_Probability', data=df_plot_pareto, order=category_order, palette="crest_r", showfliers=False)
                    plt.title(f'Cumulative Probability by Top-N Predictions - {model_name}'); plt.xlabel('Top N Predictions Considered')
                    plt.ylabel('Cumulative Probability'); plt.ylim(-0.05, 1.05); plt.grid(axis='y', linestyle=':', alpha=0.7)
                    plt.tight_layout(); plt.show()
                else: print(f"   Skipping 'Cumulative Probability' plot for {model_name}: No data generated.")
            else: print(f"   Skipping 'Cumulative Probability' plot for {model_name}: No probability columns found (n_classes_proba=0).")
        else: print(f"   Skipping 'Cumulative Probability' plot for {model_name} (predicted probabilities not provided/found).")
        print("-----------------------------------------------------------")
    print(f"\n\n{'='*30} End of Evaluation {'='*30}")





# def run_trading_simulation(
#     price_test_series: pd.Series,
#     y_pred_proba_test: np.ndarray,
#     cluster_mean_ev_mapping: pd.Series,
#     model_classes_ordered: list,
#     initial_portfolio_value: float = 100000.0,
#     trade_size: float = 3000.0,
#     hold_apr: float = 0.05,
#     ev_profit_threshold: float = 0.001,
#     position_hold_days: int = 30,
#     model_name: str = "Model Strategy",
#     periods_per_year: int = 252,
#     rolling_window_days: int = 62 # Parameter for rolling metrics
# ) -> tuple[pd.DataFrame, dict, dict]:

#     num_total_trading_days = len(price_test_series)
#     if num_total_trading_days == 0: return pd.DataFrame(), {}, {}
#     if len(y_pred_proba_test) != num_total_trading_days:
#         num_total_trading_days = min(num_total_trading_days, len(y_pred_proba_test))
#         print("num_total_trading_days not same as y_pred_proba_test length")
#         if num_total_trading_days == 0: 
#             print("error, price_test_series empty. returning null")
#             return pd.DataFrame(), {}, {}
#         price_test_series = price_test_series.iloc[:num_total_trading_days]
#         y_pred_proba_test = y_pred_proba_test[:num_total_trading_days]

#     # Define strategy names
#     strat_top1_name = f'{model_name} (Top-1)'
#     strat_top2_consensus_name = f'{model_name} (Top-2 Consensus)'
#     strat_cum90_name = f'{model_name} (Cum@90)'
#     strat_cum75_name = f'{model_name} (Cum@75)'
#     strat_cum60_name = f'{model_name} (Cum@60)'
#     strat_perfect_name = 'Perfect Foresight'
#     strat_benchmark_name = 'Hold Benchmark'
    
#     all_strategy_names_trading = [
#         strat_top1_name, strat_top2_consensus_name, strat_cum90_name,
#         strat_cum75_name, strat_cum60_name, strat_perfect_name
#     ]
#     all_portfolio_columns = all_strategy_names_trading + [strat_benchmark_name]

#     portfolio_values = pd.DataFrame(
#         index=price_test_series.index,
#         columns=all_portfolio_columns
#     )
#     portfolio_values.iloc[:, :] = initial_portfolio_value
    
#     daily_rate_hold = 0.0 
#     if num_total_trading_days > 0 and periods_per_year > 0 and hold_apr is not None:
#         daily_rate_hold = (1.0 + hold_apr)**(1.0 / periods_per_year) - 1.0
#         current_hold_val = initial_portfolio_value
#         benchmark_col_loc = portfolio_values.columns.get_loc(strat_benchmark_name)
#         for t_idx in range(num_total_trading_days):
#             current_hold_val *= (1 + daily_rate_hold)
#             portfolio_values.iloc[t_idx, benchmark_col_loc] = current_hold_val
#     else:
#         portfolio_values[strat_benchmark_name] = initial_portfolio_value

#     open_positions_top1 = []
#     open_positions_top2_consensus = []
#     open_positions_cum90 = []
#     open_positions_cum75 = []
#     open_positions_cum60 = []
#     open_positions_perfect = []

#     trade_stats_top1 = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}
#     trade_stats_top2_consensus = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}
#     trade_stats_cum90 = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}
#     trade_stats_cum75 = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}
#     trade_stats_cum60 = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}
#     trade_stats_perfect = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}

#     trade_actions_log_top1 = []
#     trade_actions_log_top2_consensus = []
#     trade_actions_log_cum90 = []
#     trade_actions_log_cum75 = []
#     trade_actions_log_cum60 = []
#     trade_actions_log_perfect = []

#     buy_short_stats = {name: {"buys_wins": 0, "buys_total": 0, "shorts_wins": 0, "shorts_total": 0} for name in all_strategy_names_trading}

#     idx_to_class_map = {i: label for i, label in enumerate(model_classes_ordered)}
#     if y_pred_proba_test.shape[1] != len(model_classes_ordered):
#         print(f"CRITICAL WARNING for {model_name}: y_pred_proba_test columns ({y_pred_proba_test.shape[1]}) "
#               f"!= model_classes_ordered length ({len(model_classes_ordered)}). Signal generation flawed.")

#     positive_cluster_indices = []
#     negative_cluster_indices = []
#     if cluster_mean_ev_mapping is not None:
#         for i, class_label in idx_to_class_map.items():
#             mean_ev = cluster_mean_ev_mapping.get(class_label, 0)
#             if mean_ev > ev_profit_threshold:
#                 positive_cluster_indices.append(i)
#             elif mean_ev < -ev_profit_threshold:
#                 negative_cluster_indices.append(i)
    
#     def get_strategy_artifacts(strat_name_key):
#         if strat_name_key == strat_top1_name: return open_positions_top1, trade_stats_top1, trade_actions_log_top1
#         if strat_name_key == strat_top2_consensus_name: return open_positions_top2_consensus, trade_stats_top2_consensus, trade_actions_log_top2_consensus
#         if strat_name_key == strat_cum90_name: return open_positions_cum90, trade_stats_cum90, trade_actions_log_cum90
#         if strat_name_key == strat_cum75_name: return open_positions_cum75, trade_stats_cum75, trade_actions_log_cum75
#         if strat_name_key == strat_cum60_name: return open_positions_cum60, trade_stats_cum60, trade_actions_log_cum60
#         if strat_name_key == strat_perfect_name: return open_positions_perfect, trade_stats_perfect, trade_actions_log_perfect
#         return None, None, None

#     for t in range(num_total_trading_days):
#         current_price_today = price_test_series.iloc[t]
#         current_date = price_test_series.index[t]

#         if pd.isna(current_price_today):
#             if t > 0:
#                 portfolio_values.iloc[t, :] = portfolio_values.iloc[t-1, :]
#                 benchmark_col_loc_nan = portfolio_values.columns.get_loc(strat_benchmark_name)
#                 if daily_rate_hold != 0.0 : 
#                     portfolio_values.iloc[t, benchmark_col_loc_nan] = portfolio_values.iloc[t-1, benchmark_col_loc_nan] * (1 + daily_rate_hold)
#             for strat_name_for_nan_처리 in all_strategy_names_trading:
#                 pos_list, current_trade_stats, current_trade_log = get_strategy_artifacts(strat_name_for_nan_처리)
#                 if pos_list is None: continue
#                 for pos in pos_list[:]:
#                     pos['days_left_to_unwind'] -= 1
#                     if pos['days_left_to_unwind'] <= 0:
#                         pnl_for_stats_on_close = pos.get('accumulated_pnl_for_stats_calc', 0)
#                         if strat_name_for_nan_처리 == strat_perfect_name and 'accumulated_pnl_for_stats_calc' not in pos :
#                              pnl_for_stats_on_close = pos.get('pnl_for_stats',0)
#                         if current_trade_stats: 
#                             current_trade_stats['total'] += 1
#                             if pnl_for_stats_on_close > 1e-6: current_trade_stats['win'] += 1
#                             elif pnl_for_stats_on_close < -1e-6: current_trade_stats['loss'] += 1
#                             else: current_trade_stats['breakeven'] += 1
                        
#                         strat_buy_short_stats = buy_short_stats[strat_name_for_nan_처리]
#                         is_long = pos['direction'] == 1
#                         if is_long:
#                             strat_buy_short_stats['buys_total'] += 1
#                             if pnl_for_stats_on_close > 1e-6: strat_buy_short_stats['buys_wins'] += 1
#                         else: 
#                             strat_buy_short_stats['shorts_total'] += 1
#                             if pnl_for_stats_on_close > 1e-6: strat_buy_short_stats['shorts_wins'] += 1
                        
#                         current_trade_log.append({
#                             "type": "close", "date": current_date, "pnl": pnl_for_stats_on_close,
#                             "original_pos_type": "long" if is_long else "short",
#                             "open_date": pos['open_date'], "entry_price": pos['entry_price'],
#                             "exit_price": np.nan, 
#                             "amount_traded": pos.get('initial_shares', trade_size / pos.get('entry_price_valid_day', 1)) * pos.get('entry_price_valid_day', pos['entry_price'])
#                         })
#                         pos_list.remove(pos)
#             continue 

#         # --- P&L Processing for NON-NAN days (condensed for brevity, logic is the same as your last provided version) ---
#         for strat_name_key in all_strategy_names_trading:
#             pos_list, current_trade_stats, current_trade_log = get_strategy_artifacts(strat_name_key)
#             if pos_list is None: continue # Should not happen for these names

#             col_loc = portfolio_values.columns.get_loc(strat_name_key)
#             current_portfolio_val = portfolio_values.iloc[t-1, col_loc] if t > 0 else initial_portfolio_value
#             daily_pnl = 0
#             for pos in pos_list[:]:
#                 pnl_this_day_for_portion = pos['shares_to_unwind_daily'] * (current_price_today - pos['entry_price']) * pos['direction']
#                 daily_pnl += pnl_this_day_for_portion
#                 pos['accumulated_pnl_for_stats_calc'] += pnl_this_day_for_portion
#                 pos['days_left_to_unwind'] -= 1
#                 if pos['days_left_to_unwind'] <= 0:
#                     pnl_on_close = pos['accumulated_pnl_for_stats_calc']
#                     current_trade_stats['total'] += 1
#                     if pnl_on_close > 1e-6: current_trade_stats['win'] += 1
#                     elif pnl_on_close < -1e-6: current_trade_stats['loss'] += 1
#                     else: current_trade_stats['breakeven'] += 1
                    
#                     strat_buy_short_stats = buy_short_stats[strat_name_key]
#                     is_long = pos['direction'] == 1
#                     if is_long:
#                         strat_buy_short_stats['buys_total'] += 1
#                         if pnl_on_close > 1e-6: strat_buy_short_stats['buys_wins'] += 1
#                     else: 
#                         strat_buy_short_stats['shorts_total'] += 1
#                         if pnl_on_close > 1e-6: strat_buy_short_stats['shorts_wins'] += 1

#                     current_trade_log.append({
#                         "type": "close", "pnl": pnl_on_close,
#                         "original_pos_type": "long" if is_long else "short",
#                         "open_date": pos['open_date'], "date": current_date,
#                         "entry_price": pos['entry_price'], "exit_price": current_price_today,
#                         "amount_traded": pos['initial_shares'] * pos['entry_price']
#                     })
#                     pos_list.remove(pos)
#             portfolio_values.iloc[t, col_loc] = current_portfolio_val + daily_pnl
        
#         # --- Signal Generation & Position Initiation (condensed for brevity, logic is the same as your last provided version) ---
#         entry_price_today = current_price_today 
#         if pd.isna(entry_price_today) or entry_price_today == 0: continue
#         shares_to_trade = trade_size / entry_price_today if entry_price_today > 0 else 0
#         shares_to_unwind_daily_new_trade = shares_to_trade / position_hold_days if position_hold_days > 0 else shares_to_trade
#         if shares_to_trade == 0 : continue

#         def new_pos_dict(direction_val, entry_price_val, shares_unwind_val, initial_shares_val):
#             return {'entry_price': entry_price_val, 'entry_price_valid_day': entry_price_val,
#                     'shares_to_unwind_daily': shares_unwind_val, 'direction': direction_val, 
#                     'days_left_to_unwind': position_hold_days, 'accumulated_pnl_for_stats_calc': 0.0,
#                     'open_date': current_date, 'initial_shares': initial_shares_val}
#         def new_open_trade_log_entry(direction_val, entry_price_val):
#             return {"type": "open", "amount": trade_size, 
#                     "position_type": "long" if direction_val == 1 else "short",
#                     "date": current_date, "price": entry_price_val}

#         # Perfect Foresight Signal
#         if t + position_hold_days < num_total_trading_days:
#             exit_price_window = price_test_series.iloc[t+1 : t+1+position_hold_days]
#             if not (len(exit_price_window) < position_hold_days or exit_price_window.isna().any()):
#                 avg_exit_price = exit_price_window.mean()
#                 if not (pd.isna(avg_exit_price) or avg_exit_price == 0):
#                     actual_return_factor_for_stats = (avg_exit_price / entry_price_today)
#                     direction_perfect, expected_total_pnl_perfect = 0,0
#                     if actual_return_factor_for_stats > (1 + ev_profit_threshold): direction_perfect, expected_total_pnl_perfect = 1, trade_size * (actual_return_factor_for_stats - 1)
#                     elif actual_return_factor_for_stats < (1 - ev_profit_threshold): direction_perfect, expected_total_pnl_perfect = -1, trade_size * (1 - actual_return_factor_for_stats)
#                     if direction_perfect != 0:
#                         pos_data = new_pos_dict(direction_perfect, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade)
#                         pos_data['pnl_for_stats'] = expected_total_pnl_perfect 
#                         open_positions_perfect.append(pos_data)
#                         trade_actions_log_perfect.append(new_open_trade_log_entry(direction_perfect, entry_price_today))
#         # Top-1 Signal
#         top1_pred_idx = np.argmax(y_pred_proba_test[t])
#         if top1_pred_idx < len(idx_to_class_map):
#             top1_pred_label = idx_to_class_map.get(top1_pred_idx)
#             if top1_pred_label is not None and cluster_mean_ev_mapping is not None:
#                 top1_pred_mean_ev = cluster_mean_ev_mapping.get(top1_pred_label, 0)
#                 direction_top1 = 0
#                 if top1_pred_mean_ev > ev_profit_threshold: direction_top1 = 1
#                 elif top1_pred_mean_ev < -ev_profit_threshold: direction_top1 = -1
#                 if direction_top1 != 0:
#                     open_positions_top1.append(new_pos_dict(direction_top1, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade))
#                     trade_actions_log_top1.append(new_open_trade_log_entry(direction_top1, entry_price_today))
#         # Top-2 Consensus Signal
#         if y_pred_proba_test.shape[1] >= 2:
#             sorted_indices_today = np.argsort(y_pred_proba_test[t])[::-1]
#             top1_idx_combo, top2_idx_combo = sorted_indices_today[0], sorted_indices_today[1]
#             if top1_idx_combo < len(idx_to_class_map) and top2_idx_combo < len(idx_to_class_map):
#                 l1c, l2c = idx_to_class_map.get(top1_idx_combo), idx_to_class_map.get(top2_idx_combo)
#                 if l1c is not None and l2c is not None and cluster_mean_ev_mapping is not None:
#                     ev1c, ev2c = cluster_mean_ev_mapping.get(l1c,0), cluster_mean_ev_mapping.get(l2c,0)
#                     cat1c, cat2c = get_ev_category(ev1c, ev_profit_threshold), get_ev_category(ev2c, ev_profit_threshold)
#                     direction_top2_consensus = 0
#                     if cat1c == "Profitable" and cat2c == "Profitable": direction_top2_consensus = 1
#                     elif cat1c == "Loss" and cat2c == "Loss": direction_top2_consensus = -1
#                     if direction_top2_consensus != 0:
#                         open_positions_top2_consensus.append(new_pos_dict(direction_top2_consensus, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade))
#                         trade_actions_log_top2_consensus.append(new_open_trade_log_entry(direction_top2_consensus, entry_price_today))
#         # Cumulative Probability Signals
#         proba_today = y_pred_proba_test[t]
#         sum_positive_proba = np.sum(proba_today[positive_cluster_indices]) if len(positive_cluster_indices) > 0 else 0
#         sum_negative_proba = np.sum(proba_today[negative_cluster_indices]) if len(negative_cluster_indices) > 0 else 0
#         direction_cum90, direction_cum75, direction_cum60 = 0,0,0
#         if sum_positive_proba > 0.90: direction_cum90 = 1
#         elif sum_negative_proba > 0.90: direction_cum90 = -1
#         if direction_cum90 != 0: 
#             open_positions_cum90.append(new_pos_dict(direction_cum90, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade))
#             trade_actions_log_cum90.append(new_open_trade_log_entry(direction_cum90, entry_price_today))
#         if sum_positive_proba > 0.75: direction_cum75 = 1
#         elif sum_negative_proba > 0.75: direction_cum75 = -1
#         if direction_cum75 != 0:
#             open_positions_cum75.append(new_pos_dict(direction_cum75, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade))
#             trade_actions_log_cum75.append(new_open_trade_log_entry(direction_cum75, entry_price_today))
#         if sum_positive_proba > 0.60: direction_cum60 = 1
#         elif sum_negative_proba > 0.60: direction_cum60 = -1
#         if direction_cum60 != 0:
#             open_positions_cum60.append(new_pos_dict(direction_cum60, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade))
#             trade_actions_log_cum60.append(new_open_trade_log_entry(direction_cum60, entry_price_today))

#     # --- Calculate Performance Metrics ---
#     metrics = {}
#     epsilon_std = 1e-9 
#     all_trade_actions = { 
#         strat_top1_name: trade_actions_log_top1, strat_top2_consensus_name: trade_actions_log_top2_consensus,
#         strat_cum90_name: trade_actions_log_cum90, strat_cum75_name: trade_actions_log_cum75,
#         strat_cum60_name: trade_actions_log_cum60, strat_perfect_name: trade_actions_log_perfect
#     }
    
#     # Convert trade action dates to datetime if they are not already (for proper comparison)
#     for strat_key in all_trade_actions:
#         for trade in all_trade_actions[strat_key]:
#             if 'date' in trade and not isinstance(trade['date'], pd.Timestamp):
#                 try: trade['date'] = pd.to_datetime(trade['date'])
#                 except: pass # ignore if conversion fails
#             if 'open_date' in trade and not isinstance(trade['open_date'], pd.Timestamp):
#                 try: trade['open_date'] = pd.to_datetime(trade['open_date'])
#                 except: pass


#     for strategy_col_name in portfolio_values.columns:
#         final_value = portfolio_values[strategy_col_name].iloc[-1] if num_total_trading_days > 0 and not portfolio_values[strategy_col_name].empty else initial_portfolio_value
#         total_return_pct = ((final_value / initial_portfolio_value) - 1) * 100 if initial_portfolio_value > 0 else 0
        
#         if portfolio_values[strategy_col_name].dropna().empty and initial_portfolio_value == 0 : strat_returns = pd.Series(dtype=float)
#         elif portfolio_values[strategy_col_name].dropna().empty and initial_portfolio_value !=0 : strat_returns = pd.Series([0.0] * num_total_trading_days, index=price_test_series.index[:num_total_trading_days])
#         else :
#             temp_series_for_pct_change = pd.Series([initial_portfolio_value] + portfolio_values[strategy_col_name].tolist())
#             strat_returns = temp_series_for_pct_change.pct_change().iloc[1:].fillna(0)
#             strat_returns.index = price_test_series.index[:len(strat_returns)] # Ensure index alignment

#         sharpe_ratio_all_time = np.nan
#         if len(strat_returns) > 1 :
#             std_dev = strat_returns.std()
#             if std_dev > epsilon_std: sharpe_ratio_all_time = (strat_returns.mean() / std_dev) * np.sqrt(periods_per_year if periods_per_year > 0 else 252)
#             elif strat_returns.mean() == 0 and std_dev <= epsilon_std : sharpe_ratio_all_time = 0.0
#         elif len(strat_returns) <=1 and not strat_returns.empty and strat_returns.mean()==0 : sharpe_ratio_all_time = 0.0

#         current_metrics = {
#             'Final Portfolio Value': final_value, 'Total Return (%)': total_return_pct,
#             'Sharpe Ratio (Annualized, All-Time)': sharpe_ratio_all_time,
#             'Avg Daily Return (%)': strat_returns.mean() * 100 if not strat_returns.empty else 0,
#             'Std Dev Daily Return (%)': strat_returns.std() * 100 if len(strat_returns) > 1 else 0,
#         }
        
#         # Rolling Metrics (Sharpe, Long Win Rate, Short Win Rate)
#         # Initialize a list to store dicts for each window's metrics
#         rolling_metrics_details_list = []
#         if not strat_returns.empty and len(strat_returns) >= rolling_window_days and rolling_window_days > 0:
#             # Get the relevant trade log for the current strategy
#             strategy_specific_trades = all_trade_actions.get(strategy_col_name, [])
            
#             for i in range(0, len(strat_returns) - rolling_window_days + 1, rolling_window_days):
#                 window_returns = strat_returns.iloc[i : i + rolling_window_days]
#                 window_start_date = window_returns.index[0]
#                 window_end_date = window_returns.index[-1]
                
#                 # Rolling Sharpe
#                 sharpe_window_val = np.nan
#                 if len(window_returns) >= 2:
#                     std_dev_window = window_returns.std()
#                     mean_window = window_returns.mean()
#                     if std_dev_window > epsilon_std: sharpe_window_val = (mean_window / std_dev_window) * np.sqrt(periods_per_year if periods_per_year > 0 else 252)
#                     elif mean_window == 0 and std_dev_window <= epsilon_std : sharpe_window_val = 0.0
                
#                 # Rolling Long/Short Win Rates for this window
#                 long_wins_window, long_total_window = 0, 0
#                 short_wins_window, short_total_window = 0, 0
                
#                 if strategy_specific_trades: # Only if it's a trading strategy
#                     # Filter trades that CLOSED within this window
#                     # Ensure dates are comparable (Timestamp vs Timestamp)
#                     window_start_date_ts = pd.to_datetime(window_start_date)
#                     window_end_date_ts = pd.to_datetime(window_end_date)

#                     for trade in strategy_specific_trades:
#                         if trade['type'] == 'close':
#                             # Ensure trade['date'] is a Timestamp for comparison
#                             trade_close_date_ts = pd.to_datetime(trade['date'])
#                             if window_start_date_ts <= trade_close_date_ts <= window_end_date_ts:
#                                 is_profit = trade['pnl'] > 1e-6 # Using same threshold as overall stats
#                                 if trade['original_pos_type'] == 'long':
#                                     long_total_window += 1
#                                     if is_profit: long_wins_window += 1
#                                 elif trade['original_pos_type'] == 'short':
#                                     short_total_window += 1
#                                     if is_profit: short_wins_window += 1
                                    
#                 long_win_rate_window = (long_wins_window / long_total_window * 100) if long_total_window > 0 else 0
#                 short_win_rate_window = (short_wins_window / short_total_window * 100) if short_total_window > 0 else 0

#                 rolling_metrics_details_list.append({
#                     "window_start_date": window_start_date.strftime('%Y-%m-%d') if isinstance(window_start_date, pd.Timestamp) else str(window_start_date),
#                     "window_end_date": window_end_date.strftime('%Y-%m-%d') if isinstance(window_end_date, pd.Timestamp) else str(window_end_date),
#                     "sharpe_ratio": sharpe_window_val,
#                     "long_win_rate (%)": long_win_rate_window,
#                     "total_long_trades_in_window": long_total_window,
#                     "short_win_rate (%)": short_win_rate_window,
#                     "total_short_trades_in_window": short_total_window,
#                 })
        
#         current_metrics[f'Rolling {rolling_window_days}-Day Metrics'] = rolling_metrics_details_list
#         # Optional: Calculate average of rolling metrics if desired
#         if rolling_metrics_details_list:
#             current_metrics[f'Avg Rolling {rolling_window_days}-Day Sharpe'] = np.nanmean([m['sharpe_ratio'] for m in rolling_metrics_details_list])
#             current_metrics[f'Avg Rolling {rolling_window_days}-Day Long Win Rate (%)'] = np.nanmean([m['long_win_rate (%)'] for m in rolling_metrics_details_list])
#             current_metrics[f'Avg Rolling {rolling_window_days}-Day Short Win Rate (%)'] = np.nanmean([m['short_win_rate (%)'] for m in rolling_metrics_details_list])


#         trade_stats_map = {
#             strat_top1_name: trade_stats_top1, strat_top2_consensus_name: trade_stats_top2_consensus,
#             strat_cum90_name: trade_stats_cum90, strat_cum75_name: trade_stats_cum75,
#             strat_cum60_name: trade_stats_cum60, strat_perfect_name: trade_stats_perfect
#         }
#         if strategy_col_name in trade_stats_map:
#             stats = trade_stats_map[strategy_col_name]
#             current_metrics.update({ 
#                 'Total Trades': stats['total'], 'Win Rate (%)': (stats['win'] / stats['total'] * 100) if stats['total'] > 0 else 0,
#                 'Loss Rate (%)': (stats['loss'] / stats['total'] * 100) if stats['total'] > 0 else 0,
#                 'Breakeven Rate (%)': (stats['breakeven'] / stats['total'] * 100) if stats['total'] > 0 else 0,
#             })
#             if strategy_col_name in buy_short_stats: 
#                 specific_buy_short_stats = buy_short_stats[strategy_col_name]
#                 current_metrics['Buy Wins (%)'] = (specific_buy_short_stats['buys_wins'] / specific_buy_short_stats['buys_total'] * 100) if specific_buy_short_stats['buys_total'] > 0 else 0
#                 current_metrics['Total Buy Trades'] = specific_buy_short_stats['buys_total']
#                 current_metrics['Short Wins (%)'] = (specific_buy_short_stats['shorts_wins'] / specific_buy_short_stats['shorts_total'] * 100) if specific_buy_short_stats['shorts_total'] > 0 else 0
#                 current_metrics['Total Short Trades'] = specific_buy_short_stats['shorts_total']
#         metrics[strategy_col_name] = current_metrics
            
#     return portfolio_values, metrics, all_trade_actions


def run_trading_simulation(
    price_test_series: pd.Series,
    y_pred_proba_test: np.ndarray,
    cluster_mean_ev_mapping: pd.Series,
    model_classes_ordered: list,
    initial_portfolio_value: float = 100000.0,
    trade_size: float = 3000.0,
    hold_apr: float = 0.05,
    ev_profit_threshold: float = 0.001,
    position_hold_days: int = 30,
    model_name: str = "Model Strategy",
    periods_per_year: int = 252,
    rolling_window_days: int = 62 # Parameter for existing rolling metrics
) -> tuple[pd.DataFrame, dict, dict]:

    num_total_trading_days = len(price_test_series)
    if num_total_trading_days == 0: return pd.DataFrame(), {}, {}
    if len(y_pred_proba_test) != num_total_trading_days:
        num_total_trading_days = min(num_total_trading_days, len(y_pred_proba_test))
        print("num_total_trading_days not same as y_pred_proba_test length")
        if num_total_trading_days == 0: 
            print("error, price_test_series empty. returning null")
            return pd.DataFrame(), {}, {}
        price_test_series = price_test_series.iloc[:num_total_trading_days]
        y_pred_proba_test = y_pred_proba_test[:num_total_trading_days]

    # Define strategy names
    strat_top1_name = f'{model_name} (Top-1)'
    strat_top2_consensus_name = f'{model_name} (Top-2 Consensus)'
    strat_cum90_name = f'{model_name} (Cum@90)'
    strat_cum75_name = f'{model_name} (Cum@75)'
    strat_cum60_name = f'{model_name} (Cum@60)'
    strat_perfect_name = 'Perfect Foresight'
    strat_benchmark_name = 'Hold Benchmark'
    
    all_strategy_names_trading = [
        strat_top1_name, strat_top2_consensus_name, strat_cum90_name,
        strat_cum75_name, strat_cum60_name, strat_perfect_name
    ]
    all_portfolio_columns = all_strategy_names_trading + [strat_benchmark_name]

    portfolio_values = pd.DataFrame(
        index=price_test_series.index,
        columns=all_portfolio_columns
    )
    portfolio_values.iloc[:, :] = initial_portfolio_value
    
    daily_rate_hold = 0.0 
    if num_total_trading_days > 0 and periods_per_year > 0 and hold_apr is not None:
        daily_rate_hold = (1.0 + hold_apr)**(1.0 / periods_per_year) - 1.0
        current_hold_val = initial_portfolio_value
        benchmark_col_loc = portfolio_values.columns.get_loc(strat_benchmark_name)
        for t_idx in range(num_total_trading_days):
            current_hold_val *= (1 + daily_rate_hold)
            portfolio_values.iloc[t_idx, benchmark_col_loc] = current_hold_val
    else:
        portfolio_values[strat_benchmark_name] = initial_portfolio_value

    open_positions_top1 = []
    open_positions_top2_consensus = []
    open_positions_cum90 = []
    open_positions_cum75 = []
    open_positions_cum60 = []
    open_positions_perfect = []

    trade_stats_top1 = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}
    trade_stats_top2_consensus = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}
    trade_stats_cum90 = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}
    trade_stats_cum75 = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}
    trade_stats_cum60 = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}
    trade_stats_perfect = {'win': 0, 'loss': 0, 'breakeven': 0, 'total': 0}

    trade_actions_log_top1 = []
    trade_actions_log_top2_consensus = []
    trade_actions_log_cum90 = []
    trade_actions_log_cum75 = []
    trade_actions_log_cum60 = []
    trade_actions_log_perfect = []

    buy_short_stats = {name: {"buys_wins": 0, "buys_total": 0, "shorts_wins": 0, "shorts_total": 0} for name in all_strategy_names_trading}

    idx_to_class_map = {i: label for i, label in enumerate(model_classes_ordered)}
    if y_pred_proba_test.shape[1] != len(model_classes_ordered):
        print(f"CRITICAL WARNING for {model_name}: y_pred_proba_test columns ({y_pred_proba_test.shape[1]}) "
              f"!= model_classes_ordered length ({len(model_classes_ordered)}). Signal generation flawed.")

    positive_cluster_indices = []
    negative_cluster_indices = []
    if cluster_mean_ev_mapping is not None:
        for i, class_label in idx_to_class_map.items():
            mean_ev = cluster_mean_ev_mapping.get(class_label, 0)
            if mean_ev > ev_profit_threshold:
                positive_cluster_indices.append(i)
            elif mean_ev < -ev_profit_threshold:
                negative_cluster_indices.append(i)
    
    def get_strategy_artifacts(strat_name_key):
        if strat_name_key == strat_top1_name: return open_positions_top1, trade_stats_top1, trade_actions_log_top1
        if strat_name_key == strat_top2_consensus_name: return open_positions_top2_consensus, trade_stats_top2_consensus, trade_actions_log_top2_consensus
        if strat_name_key == strat_cum90_name: return open_positions_cum90, trade_stats_cum90, trade_actions_log_cum90
        if strat_name_key == strat_cum75_name: return open_positions_cum75, trade_stats_cum75, trade_actions_log_cum75
        if strat_name_key == strat_cum60_name: return open_positions_cum60, trade_stats_cum60, trade_actions_log_cum60
        if strat_name_key == strat_perfect_name: return open_positions_perfect, trade_stats_perfect, trade_actions_log_perfect
        return None, None, None

    for t in range(num_total_trading_days):
        current_price_today = price_test_series.iloc[t]
        current_date = price_test_series.index[t]

        if pd.isna(current_price_today):
            if t > 0:
                portfolio_values.iloc[t, :] = portfolio_values.iloc[t-1, :]
                benchmark_col_loc_nan = portfolio_values.columns.get_loc(strat_benchmark_name)
                if daily_rate_hold != 0.0 : 
                    portfolio_values.iloc[t, benchmark_col_loc_nan] = portfolio_values.iloc[t-1, benchmark_col_loc_nan] * (1 + daily_rate_hold)
            for strat_name_for_nan_처리 in all_strategy_names_trading:
                pos_list, current_trade_stats, current_trade_log = get_strategy_artifacts(strat_name_for_nan_처리)
                if pos_list is None: continue
                for pos in pos_list[:]:
                    pos['days_left_to_unwind'] -= 1
                    if pos['days_left_to_unwind'] <= 0:
                        pnl_for_stats_on_close = pos.get('accumulated_pnl_for_stats_calc', 0)
                        if strat_name_for_nan_처리 == strat_perfect_name and 'accumulated_pnl_for_stats_calc' not in pos :
                                pnl_for_stats_on_close = pos.get('pnl_for_stats',0)
                        if current_trade_stats: 
                            current_trade_stats['total'] += 1
                            if pnl_for_stats_on_close > 1e-6: current_trade_stats['win'] += 1
                            elif pnl_for_stats_on_close < -1e-6: current_trade_stats['loss'] += 1
                            else: current_trade_stats['breakeven'] += 1
                        
                        strat_buy_short_stats = buy_short_stats[strat_name_for_nan_처리]
                        is_long = pos['direction'] == 1
                        if is_long:
                            strat_buy_short_stats['buys_total'] += 1
                            if pnl_for_stats_on_close > 1e-6: strat_buy_short_stats['buys_wins'] += 1
                        else: 
                            strat_buy_short_stats['shorts_total'] += 1
                            if pnl_for_stats_on_close > 1e-6: strat_buy_short_stats['shorts_wins'] += 1
                        
                        current_trade_log.append({
                            "type": "close", "date": current_date, "pnl": pnl_for_stats_on_close,
                            "original_pos_type": "long" if is_long else "short",
                            "open_date": pos['open_date'], "entry_price": pos['entry_price'],
                            "exit_price": np.nan, 
                            "amount_traded": pos.get('initial_shares', trade_size / pos.get('entry_price_valid_day', 1)) * pos.get('entry_price_valid_day', pos['entry_price'])
                        })
                        pos_list.remove(pos)
            continue 

        # --- P&L Processing for NON-NAN days ---
        for strat_name_key in all_strategy_names_trading:
            pos_list, current_trade_stats, current_trade_log = get_strategy_artifacts(strat_name_key)
            if pos_list is None: continue 

            col_loc = portfolio_values.columns.get_loc(strat_name_key)
            current_portfolio_val = portfolio_values.iloc[t-1, col_loc] if t > 0 else initial_portfolio_value
            daily_pnl = 0
            for pos in pos_list[:]:
                pnl_this_day_for_portion = pos['shares_to_unwind_daily'] * (current_price_today - pos['entry_price']) * pos['direction']
                daily_pnl += pnl_this_day_for_portion
                pos['accumulated_pnl_for_stats_calc'] += pnl_this_day_for_portion
                pos['days_left_to_unwind'] -= 1
                if pos['days_left_to_unwind'] <= 0:
                    pnl_on_close = pos['accumulated_pnl_for_stats_calc']
                    current_trade_stats['total'] += 1
                    if pnl_on_close > 1e-6: current_trade_stats['win'] += 1
                    elif pnl_on_close < -1e-6: current_trade_stats['loss'] += 1
                    else: current_trade_stats['breakeven'] += 1
                    
                    strat_buy_short_stats = buy_short_stats[strat_name_key]
                    is_long = pos['direction'] == 1
                    if is_long:
                        strat_buy_short_stats['buys_total'] += 1
                        if pnl_on_close > 1e-6: strat_buy_short_stats['buys_wins'] += 1
                    else: 
                        strat_buy_short_stats['shorts_total'] += 1
                        if pnl_on_close > 1e-6: strat_buy_short_stats['shorts_wins'] += 1

                    current_trade_log.append({
                        "type": "close", "pnl": pnl_on_close,
                        "original_pos_type": "long" if is_long else "short",
                        "open_date": pos['open_date'], "date": current_date,
                        "entry_price": pos['entry_price'], "exit_price": current_price_today,
                        "amount_traded": pos['initial_shares'] * pos['entry_price']
                    })
                    pos_list.remove(pos)
            portfolio_values.iloc[t, col_loc] = current_portfolio_val + daily_pnl
        
        # --- Signal Generation & Position Initiation ---
        entry_price_today = current_price_today 
        if pd.isna(entry_price_today) or entry_price_today == 0: continue
        shares_to_trade = trade_size / entry_price_today if entry_price_today > 0 else 0
        shares_to_unwind_daily_new_trade = shares_to_trade / position_hold_days if position_hold_days > 0 else shares_to_trade
        if shares_to_trade == 0 : continue

        def new_pos_dict(direction_val, entry_price_val, shares_unwind_val, initial_shares_val):
            return {'entry_price': entry_price_val, 'entry_price_valid_day': entry_price_val,
                    'shares_to_unwind_daily': shares_unwind_val, 'direction': direction_val, 
                    'days_left_to_unwind': position_hold_days, 'accumulated_pnl_for_stats_calc': 0.0,
                    'open_date': current_date, 'initial_shares': initial_shares_val}
        def new_open_trade_log_entry(direction_val, entry_price_val):
            return {"type": "open", "amount": trade_size, 
                    "position_type": "long" if direction_val == 1 else "short",
                    "date": current_date, "price": entry_price_val}

        # Perfect Foresight Signal
        if t + position_hold_days < num_total_trading_days:
            exit_price_window = price_test_series.iloc[t+1 : t+1+position_hold_days]
            if not (len(exit_price_window) < position_hold_days or exit_price_window.isna().any()):
                avg_exit_price = exit_price_window.mean()
                if not (pd.isna(avg_exit_price) or avg_exit_price == 0):
                    actual_return_factor_for_stats = (avg_exit_price / entry_price_today)
                    direction_perfect, expected_total_pnl_perfect = 0,0
                    if actual_return_factor_for_stats > (1 + ev_profit_threshold): direction_perfect, expected_total_pnl_perfect = 1, trade_size * (actual_return_factor_for_stats - 1)
                    elif actual_return_factor_for_stats < (1 - ev_profit_threshold): direction_perfect, expected_total_pnl_perfect = -1, trade_size * (1 - actual_return_factor_for_stats)
                    if direction_perfect != 0:
                        pos_data = new_pos_dict(direction_perfect, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade)
                        pos_data['pnl_for_stats'] = expected_total_pnl_perfect 
                        open_positions_perfect.append(pos_data)
                        trade_actions_log_perfect.append(new_open_trade_log_entry(direction_perfect, entry_price_today))
        # Top-1 Signal
        top1_pred_idx = np.argmax(y_pred_proba_test[t])
        if top1_pred_idx < len(idx_to_class_map): # Check index bounds
            top1_pred_label = idx_to_class_map.get(top1_pred_idx)
            if top1_pred_label is not None and cluster_mean_ev_mapping is not None:
                top1_pred_mean_ev = cluster_mean_ev_mapping.get(top1_pred_label, 0)
                direction_top1 = 0
                if top1_pred_mean_ev > ev_profit_threshold: direction_top1 = 1
                elif top1_pred_mean_ev < -ev_profit_threshold: direction_top1 = -1
                if direction_top1 != 0:
                    open_positions_top1.append(new_pos_dict(direction_top1, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade))
                    trade_actions_log_top1.append(new_open_trade_log_entry(direction_top1, entry_price_today))
        
        # Top-2 Consensus Signal
        if y_pred_proba_test.shape[1] >= 2:
            sorted_indices_today = np.argsort(y_pred_proba_test[t])[::-1]
            top1_idx_combo, top2_idx_combo = sorted_indices_today[0], sorted_indices_today[1]
            if top1_idx_combo < len(idx_to_class_map) and top2_idx_combo < len(idx_to_class_map): # Check index bounds
                l1c, l2c = idx_to_class_map.get(top1_idx_combo), idx_to_class_map.get(top2_idx_combo)
                if l1c is not None and l2c is not None and cluster_mean_ev_mapping is not None:
                    ev1c, ev2c = cluster_mean_ev_mapping.get(l1c,0), cluster_mean_ev_mapping.get(l2c,0)
                    cat1c, cat2c = get_ev_category(ev1c, ev_profit_threshold), get_ev_category(ev2c, ev_profit_threshold)
                    direction_top2_consensus = 0
                    if cat1c == "Profitable" and cat2c == "Profitable": direction_top2_consensus = 1
                    elif cat1c == "Loss" and cat2c == "Loss": direction_top2_consensus = -1
                    if direction_top2_consensus != 0:
                        open_positions_top2_consensus.append(new_pos_dict(direction_top2_consensus, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade))
                        trade_actions_log_top2_consensus.append(new_open_trade_log_entry(direction_top2_consensus, entry_price_today))

        # Cumulative Probability Signals
        proba_today = y_pred_proba_test[t]
        sum_positive_proba = np.sum(proba_today[positive_cluster_indices]) if len(positive_cluster_indices) > 0 else 0
        sum_negative_proba = np.sum(proba_today[negative_cluster_indices]) if len(negative_cluster_indices) > 0 else 0
        direction_cum90, direction_cum75, direction_cum60 = 0,0,0
        if sum_positive_proba > 0.90: direction_cum90 = 1
        elif sum_negative_proba > 0.90: direction_cum90 = -1
        if direction_cum90 != 0: 
            open_positions_cum90.append(new_pos_dict(direction_cum90, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade))
            trade_actions_log_cum90.append(new_open_trade_log_entry(direction_cum90, entry_price_today))
        if sum_positive_proba > 0.75: direction_cum75 = 1
        elif sum_negative_proba > 0.75: direction_cum75 = -1
        if direction_cum75 != 0:
            open_positions_cum75.append(new_pos_dict(direction_cum75, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade))
            trade_actions_log_cum75.append(new_open_trade_log_entry(direction_cum75, entry_price_today))
        if sum_positive_proba > 0.60: direction_cum60 = 1
        elif sum_negative_proba > 0.60: direction_cum60 = -1
        if direction_cum60 != 0:
            open_positions_cum60.append(new_pos_dict(direction_cum60, entry_price_today, shares_to_unwind_daily_new_trade, shares_to_trade))
            trade_actions_log_cum60.append(new_open_trade_log_entry(direction_cum60, entry_price_today))

    # --- Calculate Performance Metrics ---
    metrics = {}
    epsilon_std = 1e-9 
    all_trade_actions = { 
        strat_top1_name: trade_actions_log_top1, strat_top2_consensus_name: trade_actions_log_top2_consensus,
        strat_cum90_name: trade_actions_log_cum90, strat_cum75_name: trade_actions_log_cum75,
        strat_cum60_name: trade_actions_log_cum60, strat_perfect_name: trade_actions_log_perfect
    }
    
    # Convert trade action dates to datetime if they are not already (for proper comparison)
    for strat_key in all_trade_actions:
        for trade in all_trade_actions[strat_key]:
            if 'date' in trade and not isinstance(trade['date'], pd.Timestamp):
                try: trade['date'] = pd.to_datetime(trade['date'])
                except: pass 
            if 'open_date' in trade and not isinstance(trade['open_date'], pd.Timestamp):
                try: trade['open_date'] = pd.to_datetime(trade['open_date'])
                except: pass

    for strategy_col_name in portfolio_values.columns:
        final_value = portfolio_values[strategy_col_name].iloc[-1] if num_total_trading_days > 0 and not portfolio_values[strategy_col_name].empty else initial_portfolio_value
        total_return_pct = ((final_value / initial_portfolio_value) - 1) * 100 if initial_portfolio_value != 0 else 0 # Avoid division by zero
        
        if portfolio_values[strategy_col_name].dropna().empty and initial_portfolio_value == 0 : strat_returns = pd.Series(dtype=float)
        elif portfolio_values[strategy_col_name].dropna().empty and initial_portfolio_value !=0 : strat_returns = pd.Series([0.0] * num_total_trading_days, index=price_test_series.index[:num_total_trading_days])
        else :
            # Prepend initial portfolio value for pct_change calculation
            # Ensure the series for pct_change starts with the correct initial value for this strategy
            # For benchmark, it starts with initial_portfolio_value and grows by daily_rate_hold
            # For trading strategies, it starts with initial_portfolio_value and changes by daily_pnl
            temp_series_for_pct_change = pd.Series([initial_portfolio_value] + portfolio_values[strategy_col_name].tolist())
            strat_returns = temp_series_for_pct_change.pct_change().iloc[1:].fillna(0)
            if num_total_trading_days > 0 and len(strat_returns) == num_total_trading_days:
                 strat_returns.index = price_test_series.index[:len(strat_returns)] # Ensure index alignment
            elif num_total_trading_days > 0 : # Mismatch, try to align
                 strat_returns = pd.Series(strat_returns.values, index=price_test_series.index[:len(strat_returns)])


        sharpe_ratio_all_time = np.nan
        if len(strat_returns) > 1 :
            std_dev = strat_returns.std()
            if std_dev > epsilon_std: sharpe_ratio_all_time = (strat_returns.mean() / std_dev) * np.sqrt(periods_per_year if periods_per_year > 0 else 252)
            elif strat_returns.mean() == 0 and std_dev <= epsilon_std : sharpe_ratio_all_time = 0.0
        elif len(strat_returns) == 1 and not strat_returns.empty and strat_returns.iloc[0]==0 : sharpe_ratio_all_time = 0.0


        current_metrics = {
            'Final Portfolio Value': final_value, 'Total Return (%)': total_return_pct,
            'Sharpe Ratio (Annualized, All-Time)': sharpe_ratio_all_time,
            'Avg Daily Return (%)': strat_returns.mean() * 100 if not strat_returns.empty else 0,
            'Std Dev Daily Return (%)': strat_returns.std() * 100 if len(strat_returns) > 1 else 0,
        }
        #print(current_metrics)
        # --- START: New Metrics for last N trading days (10, 30, 60) ---
        metrics_periods_last_n = [10, 30, 60]
        if num_total_trading_days > 0: 
            for n_days_lookback in metrics_periods_last_n:
                # --- Sharpe Ratio Last N Days ---
                sharpe_key = f"sharpe_ratio_last_{n_days_lookback}_trading_days"
                returns_slice_for_sharpe = pd.Series(dtype=float) # Default to empty
                if len(strat_returns) >= n_days_lookback: # Sufficient data for full lookback
                    returns_slice_for_sharpe = strat_returns.iloc[-n_days_lookback:]
                elif len(strat_returns) > 1: # Partial data, but more than 1 day
                    returns_slice_for_sharpe = strat_returns.iloc[:] # Use all available
                
                if len(returns_slice_for_sharpe) > 1:
                    std_dev_slice = returns_slice_for_sharpe.std()
                    mean_slice = returns_slice_for_sharpe.mean()
                    if std_dev_slice > epsilon_std:
                        current_metrics[sharpe_key] = (mean_slice / std_dev_slice) * np.sqrt(periods_per_year if periods_per_year > 0 else 252)
                    elif mean_slice == 0 and std_dev_slice <= epsilon_std:
                        current_metrics[sharpe_key] = 0.0
                    else: 
                        current_metrics[sharpe_key] = np.nan 
                elif len(returns_slice_for_sharpe) == 1 and not returns_slice_for_sharpe.empty and returns_slice_for_sharpe.iloc[0] == 0.0:
                    current_metrics[sharpe_key] = 0.0
                else: 
                    current_metrics[sharpe_key] = np.nan

                # --- Win Rate Last N Days ---
                win_rate_key = f"win_rate_last_{n_days_lookback}_trading_days"
                if strategy_col_name in all_trade_actions:
                    strategy_specific_trades = all_trade_actions[strategy_col_name]
                    wins_ndays, total_trades_ndays = 0, 0
                    
                    end_date_of_simulation = price_test_series.index[-1]
                    start_date_idx_for_window = max(0, num_total_trading_days - n_days_lookback)
                    start_date_of_window = price_test_series.index[start_date_idx_for_window]

                    for trade in strategy_specific_trades:
                        if trade['type'] == 'close':
                            trade_close_date = trade.get('date') 
                            if not isinstance(trade_close_date, pd.Timestamp):
                                try: trade_close_date = pd.to_datetime(trade_close_date)
                                except: continue # Skip if date is unparseable
                            
                            # Basic timezone handling: if index is aware, try to make trade_date aware, if index is naive, make trade_date naive.
                            # This is a simplified approach; robust timezone handling can be complex.
                            if price_test_series.index.tz is not None: # Index is TZ aware
                                if trade_close_date.tz is None: # Trade date is naive
                                    try: trade_close_date = trade_close_date.tz_localize(price_test_series.index.tz)
                                    except Exception: pass # If localization fails, proceed with caution
                                elif trade_close_date.tz != price_test_series.index.tz: # Both aware, different TZs
                                    try: trade_close_date = trade_close_date.tz_convert(price_test_series.index.tz)
                                    except Exception: pass
                            else: # Index is TZ naive
                                if trade_close_date.tz is not None: # Trade date is aware
                                     try: trade_close_date = trade_close_date.tz_localize(None) # Make naive
                                     except Exception: pass
                            
                            if pd.isna(trade_close_date): continue

                            if start_date_of_window <= trade_close_date <= end_date_of_simulation:
                                total_trades_ndays += 1
                                if trade.get('pnl', 0) > 1e-6: 
                                    wins_ndays += 1
                    current_metrics[win_rate_key] = (wins_ndays / total_trades_ndays * 100) if total_trades_ndays > 0 else 0.0
                else: 
                    current_metrics[win_rate_key] = np.nan

                # --- Running Total Returns Last N Days ---
                running_total_returns_key = f"running_total_returns_last_{n_days_lookback}_trading_days"
                portfolio_series_for_strat = portfolio_values[strategy_col_name]
                
                val_end_period = portfolio_series_for_strat.iloc[-1]
                val_start_for_calc = initial_portfolio_value # Default for short histories

                if num_total_trading_days > n_days_lookback:
                    # Value N days prior to the start of the N-day interval ending today is at index -(N+1)
                    val_start_for_calc = portfolio_series_for_strat.iloc[-(n_days_lookback + 1)]
                # If num_total_trading_days <= n_days_lookback, val_start_for_calc remains initial_portfolio_value

                if val_start_for_calc is not None and val_start_for_calc != 0:
                    current_metrics[running_total_returns_key] = ((val_end_period / val_start_for_calc) - 1) * 100
                elif val_end_period == val_start_for_calc: 
                    current_metrics[running_total_returns_key] = 0.0
                else: 
                    current_metrics[running_total_returns_key] = np.nan
        else: # num_total_trading_days is 0
            for n_days_lookback in metrics_periods_last_n:
                current_metrics[f"sharpe_ratio_last_{n_days_lookback}_trading_days"] = np.nan
                current_metrics[f"win_rate_last_{n_days_lookback}_trading_days"] = np.nan
                current_metrics[f"running_total_returns_last_{n_days_lookback}_trading_days"] = np.nan
        # --- END: New Metrics for last N trading days ---


        # Rolling Metrics (Sharpe, Long Win Rate, Short Win Rate) - Existing logic
        rolling_metrics_details_list = []
        if not strat_returns.empty and len(strat_returns) >= rolling_window_days and rolling_window_days > 0:
            strategy_specific_trades_for_rolling = all_trade_actions.get(strategy_col_name, [])
            
            for i in range(0, len(strat_returns) - rolling_window_days + 1, rolling_window_days):
                window_returns = strat_returns.iloc[i : i + rolling_window_days]
                window_start_date = window_returns.index[0]
                window_end_date = window_returns.index[-1]
                
                sharpe_window_val = np.nan
                if len(window_returns) >= 2:
                    std_dev_window = window_returns.std()
                    mean_window = window_returns.mean()
                    if std_dev_window > epsilon_std: sharpe_window_val = (mean_window / std_dev_window) * np.sqrt(periods_per_year if periods_per_year > 0 else 252)
                    elif mean_window == 0 and std_dev_window <= epsilon_std : sharpe_window_val = 0.0
                
                long_wins_window, long_total_window = 0, 0
                short_wins_window, short_total_window = 0, 0
                
                if strategy_specific_trades_for_rolling: 
                    window_start_date_ts = pd.to_datetime(window_start_date)
                    window_end_date_ts = pd.to_datetime(window_end_date)

                    for trade in strategy_specific_trades_for_rolling:
                        if trade['type'] == 'close':
                            trade_close_date_ts = pd.to_datetime(trade['date']) # Assumes 'date' is convertible
                            # Simplified timezone handling for rolling metrics, assuming consistency from earlier conversion
                            if window_start_date_ts <= trade_close_date_ts <= window_end_date_ts:
                                is_profit = trade['pnl'] > 1e-6 
                                if trade['original_pos_type'] == 'long':
                                    long_total_window += 1
                                    if is_profit: long_wins_window += 1
                                elif trade['original_pos_type'] == 'short':
                                    short_total_window += 1
                                    if is_profit: short_wins_window += 1
                                    
                long_win_rate_window = (long_wins_window / long_total_window * 100) if long_total_window > 0 else 0
                short_win_rate_window = (short_wins_window / short_total_window * 100) if short_total_window > 0 else 0

                rolling_metrics_details_list.append({
                    "window_start_date": window_start_date.strftime('%Y-%m-%d') if isinstance(window_start_date, pd.Timestamp) else str(window_start_date),
                    "window_end_date": window_end_date.strftime('%Y-%m-%d') if isinstance(window_end_date, pd.Timestamp) else str(window_end_date),
                    "sharpe_ratio": sharpe_window_val,
                    "long_win_rate (%)": long_win_rate_window,
                    "total_long_trades_in_window": long_total_window,
                    "short_win_rate (%)": short_win_rate_window,
                    "total_short_trades_in_window": short_total_window,
                })
        
        current_metrics[f'Rolling {rolling_window_days}-Day Metrics'] = rolling_metrics_details_list
        if rolling_metrics_details_list:
            current_metrics[f'Avg Rolling {rolling_window_days}-Day Sharpe'] = np.nanmean([m['sharpe_ratio'] for m in rolling_metrics_details_list if 'sharpe_ratio' in m])
            current_metrics[f'Avg Rolling {rolling_window_days}-Day Long Win Rate (%)'] = np.nanmean([m['long_win_rate (%)'] for m in rolling_metrics_details_list if 'long_win_rate (%)' in m])
            current_metrics[f'Avg Rolling {rolling_window_days}-Day Short Win Rate (%)'] = np.nanmean([m['short_win_rate (%)'] for m in rolling_metrics_details_list if 'short_win_rate (%)' in m])


        trade_stats_map = {
            strat_top1_name: trade_stats_top1, strat_top2_consensus_name: trade_stats_top2_consensus,
            strat_cum90_name: trade_stats_cum90, strat_cum75_name: trade_stats_cum75,
            strat_cum60_name: trade_stats_cum60, strat_perfect_name: trade_stats_perfect
        }
        if strategy_col_name in trade_stats_map:
            stats = trade_stats_map[strategy_col_name]
            current_metrics.update({ 
                'Total Trades': stats['total'], 'Win Rate (%)': (stats['win'] / stats['total'] * 100) if stats['total'] > 0 else 0,
                'Loss Rate (%)': (stats['loss'] / stats['total'] * 100) if stats['total'] > 0 else 0,
                'Breakeven Rate (%)': (stats['breakeven'] / stats['total'] * 100) if stats['total'] > 0 else 0,
            })
            if strategy_col_name in buy_short_stats: 
                specific_buy_short_stats = buy_short_stats[strategy_col_name]
                current_metrics['Buy Wins (%)'] = (specific_buy_short_stats['buys_wins'] / specific_buy_short_stats['buys_total'] * 100) if specific_buy_short_stats['buys_total'] > 0 else 0
                current_metrics['Total Buy Trades'] = specific_buy_short_stats['buys_total']
                current_metrics['Short Wins (%)'] = (specific_buy_short_stats['shorts_wins'] / specific_buy_short_stats['shorts_total'] * 100) if specific_buy_short_stats['shorts_total'] > 0 else 0
                current_metrics['Total Short Trades'] = specific_buy_short_stats['shorts_total']
        metrics[strategy_col_name] = current_metrics


    return portfolio_values, metrics, all_trade_actions

def plot_trading_actions(
    price_series: pd.Series,
    trade_actions: list,
    strategy_name: str,
    last_n_days: int = 90
):
    if price_series.empty:
        print(f"Price series for {strategy_name} is empty. Skipping trade action plot.")
        return

    plot_data = price_series.copy()
    if not plot_data.index.is_monotonic_increasing:
        plot_data = plot_data.sort_index() # Ensure data is sorted by date for plotting

    if len(plot_data) > last_n_days:
        plot_data = plot_data.iloc[-last_n_days:]

    if plot_data.empty:
        print(f"No data to plot for {strategy_name} in the last {last_n_days} days.")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_data.index, plot_data.values, label=f'{price_series.name if price_series.name else "Price"}', color='dodgerblue', lw=1.5)

    buys_x, buys_y = [], []
    shorts_x, shorts_y = [], []

    min_date_plot = plot_data.index.min()
    max_date_plot = plot_data.index.max()

    for action in trade_actions:
        # Ensure action_date is a Timestamp, especially if it's the primary key for an "open" or "close" event.
        # The 'date' field in our new log means the date the event occurred (open or close).
        action_event_date_str = action.get('date')
        action_event_date = None

        if action_event_date_str is not None:
            try:
                action_event_date = pd.Timestamp(action_event_date_str)
            except Exception as e:
                # print(f"Warning: Could not parse date '{action_event_date_str}' for an action in {strategy_name}. Skipping this action for plotting. Error: {e}")
                continue # Skip if date cannot be parsed
        else:
            # print(f"Warning: Action in {strategy_name} is missing 'date' field. Skipping.")
            continue # Skip if no date

        # We are interested in plotting 'open' signals
        if action.get('type') == 'open':
            # Check if this 'open' action falls within the plotting date range
            if min_date_plot <= action_event_date <= max_date_plot:
                action_price = action.get('price')
                position_type = action.get('position_type')

                if action_price is None or position_type is None:
                    # print(f"Warning: 'open' action on {action_event_date} for {strategy_name} is missing 'price' or 'position_type'. Skipping.")
                    continue

                if position_type == 'long':
                    buys_x.append(action_event_date)
                    buys_y.append(action_price)
                elif position_type == 'short':
                    shorts_x.append(action_event_date)
                    shorts_y.append(action_price)
    
    if buys_x:
        ax.scatter(buys_x, buys_y, color='green', marker='^', s=100, label='Buy Signal (Open Long)', alpha=0.9, edgecolors='k', zorder=5)
    if shorts_x:
        ax.scatter(shorts_x, shorts_y, color='red', marker='v', s=100, label='Short Signal (Open Short)', alpha=0.9, edgecolors='k', zorder=5)

    ax.set_title(f'Trading Actions for {strategy_name} (Last {last_n_days} Days)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.2f}'))

    if isinstance(plot_data.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=30, ha='right')
    
    # Ensure x-axis limits cover the plotted data range
    if not plot_data.empty:
        ax.set_xlim(plot_data.index.min(), plot_data.index.max())

    plt.tight_layout()
    plt.show()



def visualize_strategy_kpis_versus_benchmarks(
    strategy_name: str,
    strategy_metrics: dict, 
    target_apr_pct_value: float,
    model_context_name: str = "",
    rolling_window_days: int = 62 
):
    """
    Plots:
    1. Sharpe ratio dot plot.
    2. APR vs Target bar chart.
    3. Rolling Sharpe Ratio line chart.
    4. Rolling Long & Short Win Rates line chart.

    strategy_metrics should contain keys like 'Sharpe Ratio (Annualized, All-Time)',
    'Calculated APR (%)', and f'Rolling {rolling_window_days}-Day Metrics'.
    target_apr_pct_value should be in percentage units (e.g., 5 for 5%).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 4)) # Changed to 2x2 layout
    fig.suptitle(f'Key Performance Indicators: {strategy_name} ({model_context_name})', fontsize=16, y=0.98)

    # Extract necessary metrics
    strategy_sharpe = strategy_metrics.get('Sharpe Ratio (Annualized, All-Time)', np.nan)
    strategy_apr_pct = strategy_metrics.get('Calculated APR (%)', np.nan) 

    benchmark_metrics = strategy_metrics.get('Hold Benchmark', {})
    hold_sharpe = benchmark_metrics.get('Sharpe Ratio (Annualized, All-Time)', np.nan)
    hold_apr_pct = benchmark_metrics.get('Calculated APR (%)', np.nan)


    # --- Plot 1: Sharpe Ratio Dot Plot (axes[0, 0]) ---
    ax1 = axes[0, 0]
    ax1.set_title('All-Time Strat vs Hold Sharpe Ratio Analysis', fontsize=12)
    x_vals = [0, 1]
    labels = ['Strategy', 'Hold Benchmark']
    colors = ['royalblue', 'royalblue']
    y_vals = [strategy_sharpe, hold_sharpe]
    for i, (x, y, label, color) in enumerate(zip(x_vals, y_vals, labels, colors)):
        if not pd.isna(y):
            ax1.plot(x, y, 'o', color=color, markersize=12, alpha=0.8, label=label)
            ax1.text(x, y, f' {y:.2f}', va='bottom', ha='center', fontsize=10, color=color, fontweight='bold')
        else:
            ax1.text(x, 1.5, 'Sharpe: N/A', ha='center', va='center', fontsize=12, color='grey')
    ax1.axhspan(1, 2, alpha=0.2, color='darkseagreen', label='Typical Sharpe Range (1-2)')
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Sharpe Ratio')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
    current_sharpe_points = [val for val in y_vals if not pd.isna(val)] + [0.8, 2.2]
    min_y_s, max_y_s = min(current_sharpe_points) - 0.5, max(current_sharpe_points) + 0.5
    if abs(max_y_s - min_y_s) < 1: min_y_s -= 0.5; max_y_s += 0.5
    ax1.set_ylim(min_y_s, max_y_s)

    # --- Plot 2: Average APR vs Target APR Bar Chart (axes[0, 1]) ---
    ax2 = axes[0, 1]
    ax2.set_title('Average Annualized APR vs Target', fontsize=12)
    labels_apr = ['Strategy APR', 'Target APR']
    strategy_apr_val_for_plot = 0
    color_strategy_apr = 'lightgray'
    strategy_apr_label = "N/A"
    if not pd.isna(strategy_apr_pct):
        strategy_apr_val_for_plot = strategy_apr_pct
        color_strategy_apr = 'salmon'
        strategy_apr_label = f'{strategy_apr_pct:.2f}%'
    values_for_apr_bars = [strategy_apr_val_for_plot, target_apr_pct_value] # target_apr_pct_value already in %
    colors_for_apr_bars = [color_strategy_apr, 'skyblue']
    bars_apr = ax2.bar(labels_apr, values_for_apr_bars, color=colors_for_apr_bars, width=0.6)
    ax2.set_ylabel('APR (%)')
    for i, bar in enumerate(bars_apr):
        yval = bar.get_height()
        text_label = strategy_apr_label if i == 0 else f'{target_apr_pct_value:.2f}%'
        if i == 0 and pd.isna(strategy_apr_pct):
            ax2.text(bar.get_x() + bar.get_width()/2.0, 0.1 * max(values_for_apr_bars, default=1), 'N/A', ha='center', va='bottom', fontsize=10, color='grey')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(values_for_apr_bars, default=1), text_label,
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.grid(True, axis='y', linestyle=':', alpha=0.7)

    # --- Rolling Metrics Data Extraction ---
    rolling_metrics_key = f'Rolling {rolling_window_days}-Day Metrics'
    rolling_data = strategy_metrics.get(rolling_metrics_key, [])

    # --- Plot 3: Rolling Sharpe Ratio (axes[1, 0]) ---
    ax3 = axes[1, 0]
    ax3.set_title(f'Rolling {rolling_window_days}-Day Sharpe Ratio', fontsize=12)
    if rolling_data:
        dates = [pd.to_datetime(item['window_end_date']) for item in rolling_data]
        sharpes = [item['sharpe_ratio'] for item in rolling_data]
        ax3.plot(dates, sharpes, marker='o', linestyle='-', color='purple', label='Rolling Sharpe')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, linestyle=':', alpha=0.7)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No rolling Sharpe data', ha='center', va='center', transform=ax3.transAxes, color='grey')

    # --- Plot 4: Rolling Long & Short Win Rates (axes[1, 1]) ---
    ax4 = axes[1, 1]
    ax4.set_title(f'Rolling {rolling_window_days}-Day Win Rates', fontsize=12)
    if rolling_data:
        dates = [pd.to_datetime(item['window_end_date']) for item in rolling_data] # Re-calc or use from above
        long_win_rates = [item['long_win_rate (%)'] for item in rolling_data]
        short_win_rates = [item['short_win_rate (%)'] for item in rolling_data]

        ax4.plot(dates, long_win_rates, marker='^', linestyle='--', color='green', label='Long Win Rate (%)')
        ax4.plot(dates, short_win_rates, marker='v', linestyle='--', color='red', label='Short Win Rate (%)')

        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_ylim(0, 105) # Win rates are percentages
        ax4.grid(True, linestyle=':', alpha=0.7)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No rolling win rate data', ha='center', va='center', transform=ax4.transAxes, color='grey')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for suptitle
    plt.show()


def evaluate_trading(
    y_pred_proba_test,
    price_test_series,
    cluster_mean_ev_mapping,
    model_classes_ordered,
    model_name="Unnamed Strategy",
    initial_portfolio_value=100_000,
    trade_size=1000,
    hold_apr=0.02,
    ev_profit_threshold=0.0,
    position_hold_days=5,
    periods_per_year=252
):
    """
    Runs trading simulation and returns trading metrics dict.
    """
    portfolio_df, trading_metrics, trade_actions = run_trading_simulation(
        price_test_series=price_test_series,
        y_pred_proba_test=y_pred_proba_test,
        cluster_mean_ev_mapping=cluster_mean_ev_mapping,
        model_classes_ordered=model_classes_ordered,
        initial_portfolio_value=initial_portfolio_value,
        trade_size=trade_size,
        hold_apr=hold_apr,
        ev_profit_threshold=ev_profit_threshold,
        position_hold_days=position_hold_days,
        model_name=model_name,
        periods_per_year=periods_per_year,
    )

    return {
        "portfolio_df": portfolio_df,
        "trade_actions": trade_actions,
        "trading_metrics": trading_metrics
    }