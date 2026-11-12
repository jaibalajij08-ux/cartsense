"""
CartSense AI - Enterprise Automated Test Suite
----------------------------------------------
Verifies that all components, datasets, models, batch scoring,
radar profiling, ROC curves, and visualizations execute cleanly.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    load_dataset,
    get_overview_metrics,
    get_monthly_analysis,
    get_visitor_analysis,
    get_weekend_analysis,
    get_traffic_analysis
)
from src.ml_pipeline import (
    load_all_models_and_artifacts,
    predict_session,
    predict_batch,
    recompute_metrics_at_threshold
)
import src.visualizations as viz


def test_data_loader():
    print("[TEST] Running Data Loader test...")
    df = load_dataset()
    assert df is not None, "Dataset failed to load!"
    assert len(df) == 12330, f"Expected 12,330 rows, got {len(df)}"
    assert 'Revenue' in df.columns, "'Revenue' column missing"

    metrics = get_overview_metrics(df)
    assert metrics['total_sessions'] == 12330
    assert metrics['purchase_sessions'] == 1908
    assert metrics['non_purchase_sessions'] == 10422
    assert metrics['conversion_rate'] > 15.0 and metrics['conversion_rate'] < 16.0

    df_monthly = get_monthly_analysis(df)
    assert len(df_monthly) == 10

    df_visitor = get_visitor_analysis(df)
    assert len(df_visitor) == 3

    df_weekend = get_weekend_analysis(df)
    assert len(df_weekend) == 2

    df_traffic = get_traffic_analysis(df)
    assert len(df_traffic) == 20
    print("[PASS] Data loader & aggregations verified.")
    return df


def test_ml_pipeline():
    print("[TEST] Running ML Pipeline & Inference tests...")
    artifacts = load_all_models_and_artifacts()
    assert 'best_pipeline' in artifacts
    assert 'metrics' in artifacts
    assert 'feature_importance' in artifacts

    # 1. Single session prediction
    high_intent_input = {
        'Administrative': 4,
        'Administrative_Duration': 95.0,
        'Informational': 2,
        'Informational_Duration': 45.0,
        'ProductRelated': 35,
        'ProductRelated_Duration': 1420.0,
        'BounceRates': 0.002,
        'ExitRates': 0.015,
        'PageValues': 38.5,
        'SpecialDay': 0.0,
        'Month': 'Nov',
        'OperatingSystems': '2',
        'Browser': '2',
        'Region': '1',
        'TrafficType': '2',
        'VisitorType': 'Returning_Visitor',
        'Weekend': 'False'
    }
    result_high = predict_session(high_intent_input, artifacts)
    assert result_high['is_likely'] == True
    assert len(result_high['recommendations']) > 0
    print(f"  High-Intent Probability: {result_high['purchase_probability']*100:.1f}% -> {result_high['prediction_label']}")

    # 2. Batch prediction test
    df_sample = pd.DataFrame([high_intent_input, high_intent_input])
    df_scored, batch_summary = predict_batch(df_sample, artifacts)
    assert len(df_scored) == 2
    assert batch_summary['total_scored'] == 2
    assert batch_summary['predicted_buyers'] == 2
    print(f"  Batch Scoring: Scored {batch_summary['total_scored']} sessions successfully.")

    # 3. Threshold recomputation test
    y_dummy = np.array([0, 1, 0, 1])
    p_dummy = np.array([0.1, 0.9, 0.4, 0.6])
    thresh_res = recompute_metrics_at_threshold(y_dummy, p_dummy, threshold=0.5)
    assert thresh_res['accuracy'] == 1.0
    print("  Threshold sensitivity recomputation verified.")

    print("[PASS] ML Pipeline & real-app features verified.")
    return artifacts


def test_visualizations(df, artifacts):
    print("[TEST] Running Visualizations test...")
    f1 = viz.plot_purchase_distribution(df)
    assert f1 is not None

    df_monthly = get_monthly_analysis(df)
    f2 = viz.plot_monthly_trend(df_monthly)
    assert f2 is not None

    f3 = viz.plot_pagevalues_distribution(df)
    assert f3 is not None

    f4 = viz.plot_probability_gauge(0.85)
    assert f4 is not None

    # New Radar Chart
    f5 = viz.plot_session_radar({'ProductRelated_Duration': 1400, 'PageValues': 25, 'ExitRates': 0.02, 'BounceRates': 0.01, 'ProductRelated': 30, 'Administrative': 4})
    assert f5 is not None

    # New Sunburst Chart
    f6 = viz.plot_visitor_sunburst(df)
    assert f6 is not None

    df_visitor = get_visitor_analysis(df)
    f7 = viz.plot_visitor_type_comparison(df_visitor)
    assert f7 is not None

    f8 = viz.plot_bounce_vs_exit(df)
    assert f8 is not None

    f9 = viz.plot_duration_distribution(df)
    assert f9 is not None

    df_weekend = get_weekend_analysis(df)
    f10 = viz.plot_weekend_behavior(df_weekend)
    assert f10 is not None

    df_traffic = get_traffic_analysis(df)
    f11 = viz.plot_traffic_source_performance(df_traffic)
    assert f11 is not None

    f12 = viz.plot_model_comparison_bar(artifacts['metrics'])
    assert f12 is not None

    # New Multi-Model ROC Curve
    f13 = viz.plot_multi_model_roc(artifacts['test_sample'], artifacts['models'], artifacts['preprocessor'])
    assert f13 is not None

    # Threshold Confusion Matrix
    f14 = viz.plot_threshold_confusion_matrix([[1500, 200], [50, 716]], 'Random Forest', threshold=0.5)
    assert f14 is not None

    f15 = viz.plot_feature_importance_bar(artifacts['feature_importance']['aggregated'])
    assert f15 is not None

    print("[PASS] All 15 visualization generators executed and verified.")


if __name__ == '__main__':
    print("=" * 60)
    print("   Running CartSense AI Enterprise Test Suite")
    print("=" * 60)
    df = test_data_loader()
    artifacts = test_ml_pipeline()
    test_visualizations(df, artifacts)
    print("=" * 60)
    print("   ALL TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 60)
