"""
CartSense AI - Machine Learning Pipeline Service
------------------------------------------------
Handles loading trained models, running live predictions, computing
probabilities, generating actionable e-commerce recommendations,
batch scoring for uploaded datasets, and threshold sensitivity analysis.
"""

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_models_dir() -> str:
    """Resolve models directory across various deployment directory layouts."""
    candidates = [
        os.path.join(BASE_DIR, 'models'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
        os.path.join(os.path.dirname(os.path.abspath(__file__))),
        os.path.join(os.getcwd(), 'models'),
        os.path.join(os.getcwd()),
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, 'best_model.joblib')):
            return p
    return candidates[0]

MODELS_DIR = get_models_dir()


def ensure_models_trained():
    """Ensure trained models and evaluation artifacts exist, triggering training if needed."""
    best_model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')

    if not (os.path.exists(best_model_path) and os.path.exists(metrics_path)):
        from train_model import main as run_train
        run_train()


@st.cache_resource(show_spinner=False)
def load_all_models_and_artifacts():
    """Load cached pre-trained models, preprocessor, and metric reports."""
    ensure_models_trained()

    best_pipeline = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    lr_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.joblib'))
    dt_model = joblib.load(os.path.join(MODELS_DIR, 'decision_tree.joblib'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))

    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'r') as f:
        metrics = json.load(f)

    with open(os.path.join(MODELS_DIR, 'feature_importance.json'), 'r') as f:
        feature_importance = json.load(f)

    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    test_sample = None
    test_sample_path = os.path.join(MODELS_DIR, 'test_sample.joblib')
    if os.path.exists(test_sample_path):
        test_sample = joblib.load(test_sample_path)

    return {
        'best_pipeline': best_pipeline,
        'preprocessor': preprocessor,
        'models': {
            'Logistic Regression': lr_model,
            'Decision Tree': dt_model,
            'Random Forest': rf_model
        },
        'metrics': metrics,
        'feature_importance': feature_importance,
        'metadata': metadata,
        'test_sample': test_sample
    }


def predict_session(input_data: dict, artifacts: dict) -> dict:
    """
    Generate purchase intention prediction and probability for a shopping session.
    Explains the session's key contributing factors and produces actionable marketing actions.
    """
    pipeline = artifacts['best_pipeline']
    metadata = artifacts['metadata']

    df_input = pd.DataFrame([input_data])

    for col in metadata['categorical_cols']:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str)

    for col in metadata['numerical_cols']:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0.0)

    pred_class = int(pipeline.predict(df_input)[0])
    probabilities = pipeline.predict_proba(df_input)[0]
    purchase_prob = float(probabilities[1])
    non_purchase_prob = float(probabilities[0])

    is_likely = bool(pred_class == 1)

    margin = abs(purchase_prob - 0.5) * 2
    if margin >= 0.7:
        confidence = "Very High"
    elif margin >= 0.4:
        confidence = "High"
    elif margin >= 0.2:
        confidence = "Moderate"
    else:
        confidence = "Borderline"

    # Analyze feature influences
    explanations = []
    page_values = float(input_data.get('PageValues', 0.0))
    exit_rates = float(input_data.get('ExitRates', 0.0))
    bounce_rates = float(input_data.get('BounceRates', 0.0))
    product_duration = float(input_data.get('ProductRelated_Duration', 0.0))
    visitor_type = str(input_data.get('VisitorType', ''))
    month = str(input_data.get('Month', ''))

    if page_values > 15.0:
        explanations.append({
            'feature': 'Page Values',
            'value': f"{page_values:.2f}",
            'impact': 'Strongly Positive',
            'direction': 'positive',
            'desc': 'High page value confirms the visitor interacted with critical high-intent product or cart pages.'
        })
    elif page_values > 0.0:
        explanations.append({
            'feature': 'Page Values',
            'value': f"{page_values:.2f}",
            'impact': 'Positive',
            'direction': 'positive',
            'desc': 'Positive page value confirms the visitor touched pages that historically generate sales.'
        })
    else:
        explanations.append({
            'feature': 'Page Values',
            'value': "0.00",
            'impact': 'Negative',
            'direction': 'negative',
            'desc': 'Zero page value signifies no interaction with high-converting pages during this visit.'
        })

    if exit_rates > 0.07:
        explanations.append({
            'feature': 'Exit Rate',
            'value': f"{exit_rates*100:.1f}%",
            'impact': 'High Friction',
            'direction': 'negative',
            'desc': 'Elevated exit rate shows the visitor is likely to abandon the site without completing checkout.'
        })
    elif exit_rates < 0.03:
        explanations.append({
            'feature': 'Exit Rate',
            'value': f"{exit_rates*100:.1f}%",
            'impact': 'Low Friction',
            'direction': 'positive',
            'desc': 'Low exit rate indicates persistent exploration and ongoing site engagement.'
        })

    if bounce_rates > 0.05:
        explanations.append({
            'feature': 'Bounce Rate',
            'value': f"{bounce_rates*100:.1f}%",
            'impact': 'High Bounce',
            'direction': 'negative',
            'desc': 'High bounce rate reflects rapid disengagement from visited landing pages.'
        })
    else:
        explanations.append({
            'feature': 'Bounce Rate',
            'value': f"{bounce_rates*100:.1f}%",
            'impact': 'Active Engagement',
            'direction': 'positive',
            'desc': 'Low bounce rate indicates deep site exploration across multiple pages.'
        })

    if product_duration > 1200:
        explanations.append({
            'feature': 'Product Duration',
            'value': f"{product_duration/60:.1f} mins",
            'impact': 'Deep Consideration',
            'direction': 'positive',
            'desc': 'Long product dwell time indicates thorough inspection and comparison.'
        })

    if visitor_type == 'New_Visitor':
        explanations.append({
            'feature': 'Visitor Type',
            'value': 'New Visitor',
            'impact': 'High Conversion Velocity',
            'direction': 'positive',
            'desc': 'New visitors historically exhibit a 24.9% conversion rate once deeply engaged in catalog.'
        })

    # Generate Actionable E-Commerce Recommendations
    recommendations = []
    if is_likely:
        recommendations.append({
            'title': 'Deploy Express 1-Click Checkout',
            'desc': 'Fast-track buyer to payment with Apple Pay / Google Pay / Shop Pay to minimize drop-off.',
            'icon': '⚡'
        })
        recommendations.append({
            'title': 'Display Cart Reservation Timer',
            'desc': 'Show a friendly "Cart reserved for 15:00 minutes" prompt to incentivize immediate completion.',
            'icon': '⏳'
        })
        recommendations.append({
            'title': 'Offer One-Click Complementary Add-on',
            'desc': 'Present a personalized discounted cross-sell (e.g. warranty, accessories) right before payment.',
            'icon': '💎'
        })
    else:
        if exit_rates > 0.05:
            recommendations.append({
                'title': 'Trigger Exit-Intent Retention Offer',
                'desc': 'Display a gentle 10% discount modal or free shipping coupon when the cursor moves towards the close tab.',
                'icon': '🎁'
            })
        if bounce_rates > 0.04:
            recommendations.append({
                'title': 'Surface Social Proof & Best-Seller Carousel',
                'desc': 'Show "Top-Rated Items in Your City" to capture attention before visitor bounces.',
                'icon': '⭐'
            })
        recommendations.append({
            'title': 'Enable Proactive Live Chat Assistance',
            'desc': 'Prompt: "Need help choosing the right size or delivery date? We are here to help!"',
            'icon': '💬'
        })

    return {
        'is_likely': is_likely,
        'prediction_label': 'PURCHASE LIKELY' if is_likely else 'PURCHASE UNLIKELY',
        'purchase_probability': purchase_prob,
        'non_purchase_probability': non_purchase_prob,
        'confidence': confidence,
        'explanations': explanations,
        'recommendations': recommendations
    }


def predict_batch(df_upload: pd.DataFrame, artifacts: dict) -> tuple[pd.DataFrame, dict]:
    """
    Score a batch of customer sessions uploaded via CSV.
    Returns the augmented dataframe with predictions and a summary metrics dictionary.
    """
    pipeline = artifacts['best_pipeline']
    metadata = artifacts['metadata']

    df_clean = df_upload.copy()

    # Verify and format required columns
    for col in metadata['categorical_cols']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
        else:
            df_clean[col] = metadata['feature_defaults'].get(col, 'Unknown')

    for col in metadata['numerical_cols']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
        else:
            df_clean[col] = metadata['feature_defaults'].get(col, 0.0)

    # Run predictions
    preds = pipeline.predict(df_clean)
    probas = pipeline.predict_proba(df_clean)[:, 1]

    df_result = df_upload.copy()
    df_result['Predicted_Revenue'] = [bool(p == 1) for p in preds]
    df_result['Prediction_Label'] = ['Purchase' if p == 1 else 'No Purchase' for p in preds]
    df_result['Purchase_Probability'] = (probas * 100).round(1)

    total = len(df_result)
    buyers = int(sum(preds == 1))
    non_buyers = int(sum(preds == 0))
    conv_rate = (buyers / total * 100) if total > 0 else 0.0

    summary = {
        'total_scored': total,
        'predicted_buyers': buyers,
        'predicted_non_buyers': non_buyers,
        'projected_conversion_rate': round(conv_rate, 2),
        'avg_buyer_probability': round(float(np.mean(probas[preds == 1]) * 100), 1) if buyers > 0 else 0.0
    }

    return df_result, summary


def recompute_metrics_at_threshold(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Dynamically recompute classification metrics at any probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm
    }
