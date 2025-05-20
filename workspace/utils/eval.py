import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, r2_score

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
    metric_func = r2_score if is_regression else accuracy_score
    metric_name = "R²" if is_regression else "Accuracy"

    # Initial model
    rf = Model(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    score_before = metric_func(y_val, y_pred)
    print(f"✅ RandomForest {metric_name} (original): {score_before:.4f}")

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
