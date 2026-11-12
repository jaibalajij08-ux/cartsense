"""
CartSense AI - Data Loading and Analytics Service
-------------------------------------------------
Handles data retrieval, caching, filtering, and statistical aggregations
for the Online Shoppers Purchasing Intention dataset.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'online_shoppers_intention.csv')


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load and cache the online shoppers intention dataset."""
    if not os.path.exists(DATASET_PATH):
        # Fallback to trigger training script download if running standalone
        from train_model import download_dataset_if_missing
        download_dataset_if_missing()

    df = pd.read_csv(DATASET_PATH)

    # Convert Weekend and Revenue to readable string/boolean representations
    # Ensure types are consistent
    df['Revenue_Label'] = df['Revenue'].map({True: 'Purchase', False: 'No Purchase'})
    df['Weekend_Label'] = df['Weekend'].map({True: 'Weekend', False: 'Weekday'})

    # Month ordering for chronological charts
    month_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['Month_Clean'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)

    return df


def get_overview_metrics(df: pd.DataFrame) -> dict:
    """Compute top-level KPI metrics from actual dataset."""
    total_sessions = len(df)
    purchase_sessions = int((df['Revenue'] == True).sum())
    non_purchase_sessions = int((df['Revenue'] == False).sum())
    conversion_rate = (purchase_sessions / total_sessions * 100) if total_sessions > 0 else 0.0

    avg_page_values = float(df['PageValues'].mean())
    avg_bounce_rate = float(df['BounceRates'].mean() * 100)
    avg_exit_rate = float(df['ExitRates'].mean() * 100)
    avg_product_duration = float(df['ProductRelated_Duration'].mean())

    return {
        'total_sessions': total_sessions,
        'purchase_sessions': purchase_sessions,
        'non_purchase_sessions': non_purchase_sessions,
        'conversion_rate': round(conversion_rate, 2),
        'avg_page_values': round(avg_page_values, 2),
        'avg_bounce_rate': round(avg_bounce_rate, 2),
        'avg_exit_rate': round(avg_exit_rate, 2),
        'avg_product_duration': round(avg_product_duration, 1)
    }


def get_monthly_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sessions and conversion rate by month."""
    grouped = df.groupby('Month_Clean', observed=False).agg(
        total_sessions=('Revenue', 'count'),
        purchases=('Revenue', lambda x: (x == True).sum()),
        avg_page_values=('PageValues', 'mean')
    ).reset_index()

    grouped['conversion_rate'] = (grouped['purchases'] / grouped['total_sessions'] * 100).fillna(0).round(2)
    return grouped


def get_visitor_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by visitor type."""
    grouped = df.groupby('VisitorType').agg(
        sessions=('Revenue', 'count'),
        purchases=('Revenue', lambda x: (x == True).sum()),
        avg_page_values=('PageValues', 'mean'),
        avg_bounce_rate=('BounceRates', lambda x: x.mean() * 100),
        avg_exit_rate=('ExitRates', lambda x: x.mean() * 100),
        avg_duration=('ProductRelated_Duration', 'mean')
    ).reset_index()

    grouped['conversion_rate'] = (grouped['purchases'] / grouped['sessions'] * 100).round(2)
    grouped['avg_page_values'] = grouped['avg_page_values'].round(2)
    grouped['avg_bounce_rate'] = grouped['avg_bounce_rate'].round(2)
    grouped['avg_exit_rate'] = grouped['avg_exit_rate'].round(2)
    grouped['avg_duration'] = grouped['avg_duration'].round(1)
    return grouped


def get_weekend_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by Weekend vs Weekday."""
    grouped = df.groupby('Weekend_Label').agg(
        sessions=('Revenue', 'count'),
        purchases=('Revenue', lambda x: (x == True).sum()),
        avg_page_values=('PageValues', 'mean'),
        avg_duration=('ProductRelated_Duration', 'mean')
    ).reset_index()

    grouped['conversion_rate'] = (grouped['purchases'] / grouped['sessions'] * 100).round(2)
    grouped['avg_page_values'] = grouped['avg_page_values'].round(2)
    return grouped


def get_traffic_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by TrafficType."""
    grouped = df.groupby('TrafficType').agg(
        sessions=('Revenue', 'count'),
        purchases=('Revenue', lambda x: (x == True).sum()),
        avg_bounce_rate=('BounceRates', lambda x: x.mean() * 100),
        avg_page_values=('PageValues', 'mean')
    ).reset_index()

    grouped['conversion_rate'] = (grouped['purchases'] / grouped['sessions'] * 100).round(2)
    grouped = grouped.sort_values(by='sessions', ascending=False)
    return grouped
