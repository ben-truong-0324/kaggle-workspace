# 🧠 Active Project Context

## ✅ Current Work Focus

- Building a **versioned, modular machine learning experimentation workflow** using:
  - Custom ETL versioning (`etl.py`)
  - Model training abstraction (`train.py`, `models.py`)
  - Integrated **MLflow** and **Weights & Biases (W&B)** tracking
- Automating sweeps across multiple model types and hyperparameters
- Supporting both **scikit-learn** and **PyTorch (NN)** pipelines

---

## 📝 Recent Changes

- Added `run_etl()` with:
  - ETL versioning via `etl_generated_store.json`
  - Code-based ETL tracking (via `inspect.getsource`)
- Created `get_next_train_script()` to version training runs
- Modularized:
  - `train_sklearn_model()` and `train_nn_model()` for flexibility
  - `get_sklearn_model()` and `get_nn_model()` in `models.py`
- Integrated **MLflow** and **W&B** early in the training loop for:
  - Per-epoch metric logging
  - Full experiment metadata
  - Model artifact logging
- Enabled **custom evaluation metrics** (e.g., F1, R²) for training optimization

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

- Jupyter import caching can cause confusion — `__init__.py` is essential
- Logging early is critical to per-epoch metric tracking
- Custom metrics can be cleanly injected with `eval_metric_fn`
- Artifact tracking helps debugging as much as reproducibility
- Declarative config (`model_configs` dict) simplifies experimentation massively
