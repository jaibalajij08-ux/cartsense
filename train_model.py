"""
CartSense AI - Model Training and Evaluation Pipeline
-----------------------------------------------------
This script loads the Online Shoppers Purchasing Intention dataset,
preprocesses features, handles class imbalance, trains 3 ML algorithms
(Logistic Regression, Decision Tree, Random Forest), compares their
performance, extracts feature importances, and saves artifacts with Joblib.
"""

import os
import json
import urllib.request
import zipfile
import io
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATASET_PATH = os.path.join(DATA_DIR, 'online_shoppers_intention.csv')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Feature definitions
NUMERICAL_COLS = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'PageValues',
    'SpecialDay'
]

CATEGORICAL_COLS = [
    'Month',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
    'VisitorType',
    'Weekend'
]

TARGET_COL = 'Revenue'


def download_dataset_if_missing():
    """Ensure the UCI dataset is present locally, downloading if missing."""
    if os.path.exists(DATASET_PATH):
        print(f"[OK] Found local dataset at: {DATASET_PATH}")
        return

    print("[INFO] Dataset not found locally. Downloading from UCI Machine Learning Repository...")
    urls = [
        'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip',
        'https://raw.githubusercontent.com/PacktPublishing/Hands-on-Predictive-Analytics-with-Python/master/Chapter06/online_shoppers_intention.csv'
    ]

    downloaded = False
    for url in urls:
        try:
            print(f"[INFO] Fetching from: {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                if url.endswith('.zip'):
                    with zipfile.ZipFile(io.BytesIO(content)) as z:
                        for filename in z.namelist():
                            if filename.endswith('.csv'):
                                with open(DATASET_PATH, 'wb') as f:
                                    f.write(z.read(filename))
                                downloaded = True
                                break
                else:
                    with open(DATASET_PATH, 'wb') as f:
                        f.write(content)
                    downloaded = True
            if downloaded:
                print(f"[OK] Successfully downloaded dataset to {DATASET_PATH}")
                break
        except Exception as err:
            print(f"[WARN] Failed to download from {url}: {err}")

    if not downloaded:
        raise FileNotFoundError("Could not download online_shoppers_intention.csv. Please place it in data/ folder.")


def load_and_inspect_data():
    """Load the dataset and print summary statistics."""
    df = pd.read_csv(DATASET_PATH)
    print(f"\n{'='*60}")
    print(f"DATASET LOADED: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"{'='*60}")
    print(f"Missing values count:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    if df.isnull().sum().sum() == 0:
        print("[OK] Zero missing values detected.")

    rev_counts = df[TARGET_COL].value_counts()
    rev_props = df[TARGET_COL].value_counts(normalize=True) * 100
    print(f"\nClass Distribution (Target: {TARGET_COL}):")
    print(f"  No Purchase (False): {rev_counts.get(False, 0):,} ({rev_props.get(False, 0):.2f}%)")
    print(f"  Purchase (True):    {rev_counts.get(True, 0):,} ({rev_props.get(True, 0):.2f}%)")
    return df


def build_preprocessor():
    """Build scikit-learn ColumnTransformer for numerical scaling and categorical encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ]
    )
    return preprocessor


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train Logistic Regression, Decision Tree, and Random Forest.
    Evaluate metrics and return results dict.
    """
    # Fit preprocessor on training data only to prevent data leakage
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Class imbalance is handled with class_weight='balanced'
    algorithms = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=6,
            min_samples_split=10,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced',
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    }

    metrics_results = {}
    trained_models = {}

    print(f"\n{'='*60}")
    print("TRAINING AND EVALUATING MODELS")
    print(f"{'='*60}")

    for name, model in algorithms.items():
        print(f"\n--> Training {name}...")
        model.fit(X_train_trans, y_train)
        trained_models[name] = model

        # Evaluate on test set
        y_pred = model.predict(X_test_trans)
        y_proba = model.predict_proba(X_test_trans)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, output_dict=True)

        print(f"    Accuracy:  {acc*100:.2f}%")
        print(f"    Precision: {prec*100:.2f}%")
        print(f"    Recall:    {rec*100:.2f}%")
        print(f"    F1-Score:  {f1*100:.2f}%")
        print(f"    ROC-AUC:   {auc:.4f}")

        metrics_results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'classification_report': cr
        }

    return trained_models, metrics_results, preprocessor


def extract_feature_importances(rf_model, preprocessor):
    """
    Extract feature importances from Random Forest model and map back to
    original high-level and detailed one-hot encoded features.
    """
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    all_feature_names = NUMERICAL_COLS + cat_feature_names

    importances = rf_model.feature_importances_

    # Detailed encoded features
    detailed_importance = [
        {'feature': feat, 'importance': float(imp)}
        for feat, imp in zip(all_feature_names, importances)
    ]
    detailed_importance.sort(key=lambda x: x['importance'], reverse=True)

    # Aggregated raw features
    raw_importance = {col: 0.0 for col in NUMERICAL_COLS + CATEGORICAL_COLS}
    for item in detailed_importance:
        feat_name = item['feature']
        matched = False
        for raw_col in CATEGORICAL_COLS:
            if feat_name.startswith(f"{raw_col}_"):
                raw_importance[raw_col] += item['importance']
                matched = True
                break
        if not matched:
            if feat_name in raw_importance:
                raw_importance[feat_name] += item['importance']

    aggregated_list = [
        {'feature': k, 'importance': round(v, 4), 'percentage': round(v * 100, 2)}
        for k, v in raw_importance.items()
    ]
    aggregated_list.sort(key=lambda x: x['importance'], reverse=True)

    return {
        'detailed_top_20': detailed_importance[:20],
        'aggregated': aggregated_list
    }


def main():
    print("=" * 60)
    print("   CartSense AI - Machine Learning Training System")
    print("=" * 60)

    download_dataset_if_missing()
    df = load_and_inspect_data()

    # Prepare features and target
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].astype(int).copy()

    # Convert categorical columns to string for consistent one-hot encoding
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype(str)

    # Train / Test split with stratification to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} sessions | Test set: {X_test.shape[0]} sessions")

    # Build and fit preprocessing pipeline
    preprocessor = build_preprocessor()

    # Train and evaluate all 3 models
    trained_models, metrics_results, fitted_preprocessor = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # Select Best Model based on F1-Score (essential for imbalanced fraud/conversion tasks)
    best_name = max(metrics_results.keys(), key=lambda k: metrics_results[k]['f1_score'])
    best_model = trained_models[best_name]
    print(f"\n{'*'*60}")
    print(f"BEST MODEL SELECTED: {best_name} (F1-Score: {metrics_results[best_name]['f1_score']*100:.2f}%)")
    print(f"{'*'*60}")

    # Extract feature importance from Random Forest
    rf_model = trained_models['Random Forest']
    feature_imp_data = extract_feature_importances(rf_model, fitted_preprocessor)

    # Create full pipeline for best model so it can predict raw DataFrame directly
    best_pipeline = Pipeline([
        ('preprocessor', fitted_preprocessor),
        ('classifier', best_model)
    ])

    # Save models using Joblib
    print("\nSaving trained models and artifacts to 'models/' directory...")
    joblib.dump(fitted_preprocessor, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, 'best_model.joblib'))
    joblib.dump(trained_models['Logistic Regression'], os.path.join(MODELS_DIR, 'logistic_regression.joblib'))
    joblib.dump(trained_models['Decision Tree'], os.path.join(MODELS_DIR, 'decision_tree.joblib'))
    joblib.dump(trained_models['Random Forest'], os.path.join(MODELS_DIR, 'random_forest.joblib'))

    # Save metrics and feature importances as JSON
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics_results, f, indent=4)

    with open(os.path.join(MODELS_DIR, 'feature_importance.json'), 'w') as f:
        json.dump(feature_imp_data, f, indent=4)

    # Save metadata
    metadata = {
        'best_model': best_name,
        'numerical_cols': NUMERICAL_COLS,
        'categorical_cols': CATEGORICAL_COLS,
        'target_col': TARGET_COL,
        'unique_values': {
            col: sorted(df[col].astype(str).unique().tolist())
            for col in CATEGORICAL_COLS
        },
        'feature_defaults': {
            'Administrative': 1,
            'Administrative_Duration': 20.0,
            'Informational': 0,
            'Informational_Duration': 0.0,
            'ProductRelated': 18,
            'ProductRelated_Duration': 600.0,
            'BounceRates': 0.01,
            'ExitRates': 0.03,
            'PageValues': 15.0,
            'SpecialDay': 0.0,
            'Month': 'Nov',
            'OperatingSystems': '2',
            'Browser': '2',
            'Region': '1',
            'TrafficType': '2',
            'VisitorType': 'Returning_Visitor',
            'Weekend': 'False'
        }
    }
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    # Save a small test evaluation sample for live confusion matrix rendering
    joblib.dump((X_test, y_test), os.path.join(MODELS_DIR, 'test_sample.joblib'))

    print("[SUCCESS] All models, pipelines, and evaluation metrics saved successfully!")


if __name__ == '__main__':
    main()
