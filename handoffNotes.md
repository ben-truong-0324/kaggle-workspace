# 🧠 Active Project Context

This project is a modular ML experimentation environment built for structured datasets (e.g. Kaggle wine quality), supporting both classical and deep learning models with rich ETL tracking, model logging (MLflow & W&B), and sweep experimentation.

---

## ✅ Current Work Focus

- mlflow needs sqlite storage for runs
- xgboost also needs y labels normalized
- Building a flexible model training loop that supports:
  - `sklearn` models (trees, boosting, linear)
  - `PyTorch` models (MLPs, CNNs, LSTMs)
- Expanding hyperparameter sweeps with rich model configurations
- Improving interpretability via:
  - Cluster-to-target diagnostics
  - Per-class performance breakdowns
  - Feature and target EDA

---

## 📝 Recent Changes

- Added `model_configs` sweep dictionary with support for:
  - Decision Trees, Random Forest, Logistic Regression
  - SVM, KNN, Naive Bayes
  - XGBoost and LightGBM
  - Neural nets with variable depth, dropout, batch norm, and activation
- Upgraded `get_nn_model()` to handle `model_type` ("mlp", "cnn", "lstm") with multi-layer MLPs planned
- Created `conditionally_encode_labels()` and task inference to automate classification/regression logic
- Improved `evaluate_model()` to report per-class or binned regression metrics and log to W&B/MLflow
- Added EDA tools for target distribution, feature skew, and cluster-label alignment

---

## 🔜 Next Steps

- Add `reload_etl()` and `replay_train()` utilities to rerun past versions
- Refactor training loop into reusable `trainer.py` or a `TrainingSession` class
- Add CV folds, ensembling, or model stacking support
- Improve model saving for deployment (pickle/ONNX or TorchScript)
- Enable CLI-based or Jupyter-configurable sweeps

---

## 🧩 Active Decisions & Considerations

- One **model per MLflow/W&B run** — avoid mixing artifacts
- Save only **metadata and source** in ETL; keep data in memory
- Use `t1`, `v1`, etc., for incremental versioning (simple & readable)
- Sklearn for rapid prototyping; PyTorch for neural flexibility
- Evaluate with task-specific metrics (not just accuracy or MAE)

---

## 🧱 Important Patterns & Preferences

- **Modular functions** for each pipeline step (ETL, model, train, eval)
- **Clear abstraction boundaries** (`models.py`, `train.py`, `etl.py`)
- **Metric-first logging** per epoch (W&B & MLflow)
- **Manual control of versions**, not date-based or auto-randomized
- Local model artifact logging using standardized names (`model_{version}.pt`)

---

## 💡 Learnings & Project Insights

- Class imbalance must be handled explicitly — per-class metrics help
- Neural nets benefit from normalization, dropout, and tuning activations
- Clustering + EDA surfaces model weaknesses early
- Label encoding should be conditional and reversible
- Building a general-purpose model runner takes discipline — but pays off in reusability and reproducibility

