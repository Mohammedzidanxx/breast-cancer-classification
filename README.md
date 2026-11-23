# Breast Cancer Tumor Classification

Machine learning project that compares individual classifiers and ensemble methods (Voting, Bagging, Stacking) on the Breast Cancer Wisconsin (Diagnostic) dataset, with an interactive Streamlit app for live predictions.

The final model is a **Stacking ensemble** (Random Forest + Gradient Boosting + SVM as base learners, Logistic Regression as the meta-model), tuned with GridSearchCV using stratified 5-fold cross-validation (scoring: F1).

| Metric | Cross-validation | Held-out test set |
| --- | --- | --- |
| F1-score | 96.75% | 95.00% |
| Accuracy | 97.58% | 96.49% |

## Repository contents

| File | Description |
| --- | --- |
| `Breast_Cancer.ipynb` | Full analysis notebook: EDA, outlier handling, 6 individual classifiers, Voting/Bagging/Stacking experiments, hyperparameter tuning, and final evaluation |
| `app.py` | Interactive Streamlit app (English / Arabic) with live predictions and a comparison of all trained models |
| `train_model.py` | Rebuilds the trained model artifacts from `brca.csv` |
| `brca.csv` | Dataset (569 samples, 30 features) |
| `stacking_model.joblib` | Final tuned stacking model |
| `base_models.joblib` | The 6 individual classifiers used in the app's comparison section |
| `model_meta.json`, `base_models_metrics.json` | Feature order, cross-validation metrics, and per-model metadata |
| `requirements.txt` | Pinned dependencies |
| `run_app.bat` | Double-click launcher for Windows |

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows you can also double-click `run_app.bat`.

## Retrain the models

```bash
python train_model.py
```

## Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic) from the UCI Machine Learning Repository, licensed CC BY 4.0.
