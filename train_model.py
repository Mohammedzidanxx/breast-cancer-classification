# train_model.py
# Reproduces the exact final model from the Breast_Cancer.ipynb notebook:
# - same data prep (drop index column, B/M -> 0/1)
# - same stratified 80/20 split (random_state=42)
# - same base pipelines (RobustScaler inside SVM/KNN/LR)
# - same Stacking Experiment B + GridSearchCV tuning
# - same final model: clone of the tuned estimator, fit on X_train only
#
# Run once BEFORE starting the app:
#     python train_model.py
# It creates:  stacking_model.joblib      (the trained model, preprocessing included)
#              model_meta.json            (feature order, metrics, default input values)
#              base_models.joblib         (the six standalone individual models, for the comparison view)
#              base_models_metrics.json   (their cross-validation metrics on the training data)

import json
import joblib
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "brca.csv"
MODEL_PATH = "stacking_model.joblib"
META_PATH = "model_meta.json"


def main():
    # ---------------- Data (notebook cells 2, 14-17) ----------------
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns=["y"])
    y = df["y"].map({"B": 0, "M": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ---------------- Base models (notebook cells 18-20) ----------------
    lr_model = Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(max_iter=5000)),
    ])
    svm_model = Pipeline([
        ("scaler", RobustScaler()),
        ("model", SVC(probability=True, random_state=42)),
    ])
    knn_model = Pipeline([
        ("scaler", RobustScaler()),
        ("model", KNeighborsClassifier()),
    ])
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---------------- Stacking Experiment B (notebook cell 32) ----------------
    stacking_B = StackingClassifier(
        estimators=[
            ("rf", rf_model),
            ("gb", gb_model),
            ("svm", svm_model),
        ],
        final_estimator=LogisticRegression(max_iter=5000),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method="auto",
        n_jobs=-1,
    )

    # ---------------- Hyperparameter tuning (notebook cell 38) ----------------
    stacking_param_grid = {
        "final_estimator__C": [0.01, 0.1, 1, 10, 100],
        "final_estimator__class_weight": [None, "balanced"],
        "final_estimator__solver": ["lbfgs"],
    }

    stacking_grid = GridSearchCV(
        estimator=stacking_B,
        param_grid=stacking_param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    stacking_grid.fit(X_train, y_train)

    print("Best Parameters:", stacking_grid.best_params_)
    print("Best CV F1-Score:", round(stacking_grid.best_score_, 4))

    # ---------------- Final model (notebook cell 44) ----------------
    final_model = clone(stacking_grid.best_estimator_)
    final_model.fit(X_train, y_train)

    # ---------------- Test evaluation (notebook cells 45-47) ----------------
    y_pred = final_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        # rows = actual [Benign, Malignant], cols = predicted [Benign, Malignant]
        "confusion_matrix": cm.tolist(),
        "test_size": int(len(y_test)),
        "best_params": {k: (str(v) if v is None else v) for k, v in stacking_grid.best_params_.items()},
        "best_cv_f1": round(float(stacking_grid.best_score_), 4),
    }

    print("\nFinal Test Results")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {k}: {metrics[k]}")
    print("  confusion matrix [[TN, FP], [FN, TP]]:", cm.tolist())

    # ---------------- Standalone individual models (for the app's comparison view) --------
    # The notebook's six individual models (cells 18-20), each fitted on X_train and
    # evaluated exactly like notebook cell 21: stratified 5-fold cross-validation on the
    # training data, mean of each metric. The final row is the tuned stacking model,
    # evaluated the same way.
    individual_models = {
        "Logistic Regression": lr_model,
        "KNN": knn_model,
        "SVM": svm_model,
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
    }
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}
    base_metrics = {}
    fitted_models = {}
    for name, bm in individual_models.items():
        scores = cross_validate(bm, X_train, y_train, cv=cv, scoring=scoring)
        base_metrics[name] = {k: round(float(scores[f"test_{k}"].mean()), 4) for k in scoring}
        bm.fit(X_train, y_train)
        fitted_models[name] = bm
        print(f"  {name}: {base_metrics[name]}")

    tuned_scores = cross_validate(
        clone(stacking_grid.best_estimator_), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )
    base_metrics["Stacking"] = {k: round(float(tuned_scores[f"test_{k}"].mean()), 4) for k in scoring}
    print(f"  Stacking (tuned): {base_metrics['Stacking']}")

    joblib.dump(fitted_models, "base_models.joblib")
    with open("base_models_metrics.json", "w") as f:
        json.dump(base_metrics, f, indent=2)

    # ---------------- Save model + metadata ----------------
    joblib.dump(final_model, MODEL_PATH)

    meta = {
        "feature_names": list(X.columns),
        "defaults": {c: float(X_train[c].median()) for c in X.columns},
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "trained_with": "train_model.py - same pipeline as Breast_Cancer.ipynb",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {MODEL_PATH} and {META_PATH}")
    print(f"scikit-learn version used: {sklearn.__version__}")


if __name__ == "__main__":
    main()
