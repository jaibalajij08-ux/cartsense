"""
CartSense AI - Ultra-Vibrant Interactive Visualizations Engine
--------------------------------------------------------------
Provides Plotly chart generators styled with multi-color neon palettes,
radar benchmark comparisons, sunburst visitor breakdowns, interactive
ROC curves, and dynamic threshold confusion matrices.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# Vibrant Color Constants
THEME_BG = "rgba(10, 16, 32, 0)"
THEME_CARD = "rgba(15, 23, 42, 0.7)"
THEME_GRID = "rgba(51, 65, 85, 0.4)"

# Bright Neon & Jewel Tone Palette
COLOR_CYAN = "#00f2fe"
COLOR_ELECTRIC_BLUE = "#3b82f6"
COLOR_PURPLE = "#a855f7"
COLOR_EMERALD = "#10b981"
COLOR_CORAL = "#f43f5e"
COLOR_AMBER = "#f59e0b"
COLOR_LIME = "#84cc16"
COLOR_PINK = "#ec4899"

FONT_FAMILY = "Plus Jakarta Sans, -apple-system, sans-serif"


def apply_base_layout(fig, height=390, show_legend=True):
    """Apply consistent styling across all Plotly charts."""
    fig.update_layout(
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        font=dict(color="#f1f5f9", family=FONT_FAMILY, size=12),
        margin=dict(l=35, r=35, t=55, b=45),
        height=height,
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(71, 85, 105, 0.5)",
            borderwidth=1,
            font=dict(size=11, color="#cbd5e1")
        ),
        xaxis=dict(
            gridcolor=THEME_GRID,
            zerolinecolor=THEME_GRID,
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor=THEME_GRID,
            zerolinecolor=THEME_GRID,
            tickfont=dict(color="#94a3b8")
        )
    )
    return fig


# ---------------- PAGE 1: OVERVIEW DASHBOARD ---------------- #

def plot_purchase_distribution(df: pd.DataFrame):
    """Vibrant multi-color Donut chart with glowing segments."""
    counts = df['Revenue_Label'].value_counts().reset_index()
    counts.columns = ['Outcome', 'Count']

    # Gradient-like neon colors
    colors = [COLOR_CORAL, COLOR_EMERALD] if counts.iloc[0]['Outcome'] == 'No Purchase' else [COLOR_EMERALD, COLOR_CORAL]

    fig = go.Figure(data=[
        go.Pie(
            labels=counts['Outcome'],
            values=counts['Count'],
            hole=0.64,
            marker=dict(
                colors=colors,
                line=dict(color='#060b18', width=3)
            ),
            textinfo='label+percent',
            textfont=dict(size=13, color='#ffffff', family=FONT_FAMILY),
            hoverinfo='label+value+percent'
        )
    ])

    total = len(df)
    fig.add_annotation(
        text=f"<b style='font-size:24px;color:#ffffff;'>{total:,}</b><br><span style='font-size:12px;color:#94a3b8;'>Total Sessions</span>",
        x=0.5, y=0.5,
        font=dict(family=FONT_FAMILY),
        showarrow=False
    )
    fig.update_layout(title="<b>Session Outcome Imbalance Distribution</b>")
    return apply_base_layout(fig, height=360)


def plot_monthly_trend(df_monthly: pd.DataFrame):
    """Vibrant combo chart with cyan bars and glowing emerald conversion line."""
    fig = go.Figure()

    # Bar chart for Total Sessions with gradient fill
    fig.add_trace(go.Bar(
        x=df_monthly['Month_Clean'],
        y=df_monthly['total_sessions'],
        name='Total Sessions',
        marker=dict(
            color='rgba(14, 165, 233, 0.8)',
            line=dict(color=COLOR_CYAN, width=2)
        ),
        yaxis='y1'
    ))

    # Line chart for Conversion Rate with neon emerald glow
    fig.add_trace(go.Scatter(
        x=df_monthly['Month_Clean'],
        y=df_monthly['conversion_rate'],
        name='Conversion Rate (%)',
        mode='lines+markers',
        line=dict(color=COLOR_EMERALD, width=4, shape='spline'),
        marker=dict(size=10, color='#34d399', symbol='diamond', line=dict(color='#ffffff', width=2)),
        yaxis='y2'
    ))

    fig.update_layout(
        title="<b>Monthly Traffic Volume vs Conversion Rate Velocity</b>",
        yaxis=dict(title=dict(text="Total Sessions", font=dict(color=COLOR_CYAN))),
        yaxis2=dict(
            title=dict(text="Conversion Rate (%)", font=dict(color=COLOR_EMERALD)),
            overlaying='y',
            side='right',
            gridcolor='rgba(0,0,0,0)',
            range=[0, max(df_monthly['conversion_rate']) * 1.35]
        )
    )
    return apply_base_layout(fig, height=380)


def plot_pagevalues_distribution(df: pd.DataFrame):
    """Multi-colored violin/box plot of Google Page Values."""
    df_pv = df[df['PageValues'] > 0]
    fig = px.box(
        df_pv,
        x='Revenue_Label',
        y='PageValues',
        color='Revenue_Label',
        color_discrete_map={'Purchase': COLOR_EMERALD, 'No Purchase': COLOR_CORAL},
        points='outliers',
        title="<b>Spread of Google Page Values (Non-Zero Sessions)</b>",
        labels={'Revenue_Label': 'Session Outcome', 'PageValues': 'Google Page Value'}
    )
    return apply_base_layout(fig, height=360, show_legend=False)


# ---------------- PAGE 2: LIVE PURCHASE PREDICTION ---------------- #

def plot_probability_gauge(probability: float):
    """High-energy radial gauge with colorful threshold steps."""
    pct = probability * 100
    bar_color = COLOR_EMERALD if probability >= 0.5 else COLOR_CORAL

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 46, 'color': '#ffffff', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': bar_color, 'thickness': 0.35},
            'bgcolor': "#1e293b",
            'borderwidth': 1,
            'bordercolor': "rgba(71, 85, 105, 0.6)",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(244, 63, 94, 0.25)'},
                {'range': [35, 60], 'color': 'rgba(245, 158, 11, 0.25)'},
                {'range': [60, 100], 'color': 'rgba(16, 185, 129, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.85,
                'value': 50
            }
        }
    ))
    fig.update_layout(
        title={'text': "<b>Purchase Probability Index</b>", 'x': 0.5, 'xanchor': 'center'},
        margin=dict(l=25, r=25, t=55, b=20),
        height=290
    )
    return apply_base_layout(fig, height=290, show_legend=False)


def plot_session_radar(user_input: dict):
    """
    Interactive 6-axis Radar / Spider chart comparing current session
    against typical benchmark Buyer and typical Window Browser.
    """
    categories = [
        'Product Dwell Time',
        'Google Page Value',
        'Cart Readiness (Low Exit)',
        'Low Bounce (Engagement)',
        'Catalog Exploration Depth',
        'Administrative Visits'
    ]

    # Normalize metrics to 0-100 scale
    val_dwell = min(100.0, (float(user_input.get('ProductRelated_Duration', 0.0)) / 1800.0) * 100)
    val_pv = min(100.0, (float(user_input.get('PageValues', 0.0)) / 30.0) * 100)
    val_exit = max(0.0, 100.0 - (float(user_input.get('ExitRates', 0.0)) / 0.10) * 100)
    val_bounce = max(0.0, 100.0 - (float(user_input.get('BounceRates', 0.0)) / 0.10) * 100)
    val_prod = min(100.0, (float(user_input.get('ProductRelated', 0.0)) / 40.0) * 100)
    val_admin = min(100.0, (float(user_input.get('Administrative', 0.0)) / 6.0) * 100)

    user_values = [val_dwell, val_pv, val_exit, val_bounce, val_prod, val_admin]
    buyer_benchmark = [75, 80, 85, 90, 70, 60]
    browser_benchmark = [20, 5, 30, 40, 15, 10]

    fig = go.Figure()

    # Benchmark: Typical Buyer (Emerald)
    fig.add_trace(go.Scatterpolar(
        r=buyer_benchmark,
        theta=categories,
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.15)',
        line=dict(color=COLOR_EMERALD, width=2, dash='dot'),
        name='Avg Buyer Benchmark'
    ))

    # Benchmark: Window Browser (Coral)
    fig.add_trace(go.Scatterpolar(
        r=browser_benchmark,
        theta=categories,
        fill='toself',
        fillcolor='rgba(244, 63, 94, 0.1)',
        line=dict(color=COLOR_CORAL, width=2, dash='dot'),
        name='Casual Browser Benchmark'
    ))

    # Active User Session (Glowing Cyan)
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 242, 254, 0.35)',
        line=dict(color=COLOR_CYAN, width=3),
        name='Active User Session'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(15, 23, 42, 0.6)',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=THEME_GRID, tickfont=dict(color="#64748b", size=9)),
            angularaxis=dict(gridcolor=THEME_GRID, tickfont=dict(color="#e2e8f0", size=10, family=FONT_FAMILY))
        ),
        title="<b>Session Behaviour Radar vs Industry Benchmarks</b>",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return apply_base_layout(fig, height=360)


# ---------------- PAGE 3: CUSTOMER ANALYTICS ---------------- #

def plot_visitor_sunburst(df: pd.DataFrame):
    """Vibrant Sunburst chart displaying hierarchy: VisitorType -> Weekend -> Revenue."""
    sample = df.sample(min(4000, len(df)), random_state=42)
    fig = px.sunburst(
        sample,
        path=['VisitorType', 'Weekend_Label', 'Revenue_Label'],
        color='Revenue_Label',
        color_discrete_map={'Purchase': COLOR_EMERALD, 'No Purchase': COLOR_PURPLE, '(?)': '#64748b'},
        title="<b>Funnel Flow: Visitor Type &bull; Day Type &bull; Purchase Outcome</b>"
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return apply_base_layout(fig, height=380, show_legend=False)


def plot_visitor_type_comparison(df_visitor: pd.DataFrame):
    """Bar & line chart comparing conversion across visitor types."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_visitor['VisitorType'],
        y=df_visitor['sessions'],
        name='Total Sessions',
        marker=dict(color='rgba(99, 102, 241, 0.8)', line=dict(color=COLOR_PURPLE, width=1.5)),
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=df_visitor['VisitorType'],
        y=df_visitor['conversion_rate'],
        name='Conversion Rate (%)',
        mode='lines+markers',
        line=dict(color=COLOR_AMBER, width=3),
        marker=dict(size=11, color=COLOR_AMBER, symbol='star'),
        yaxis='y2'
    ))
    fig.update_layout(
        title="<b>Visitor Volume vs Conversion Rate by Audience Category</b>",
        yaxis=dict(title=dict(text="Total Sessions", font=dict(color=COLOR_PURPLE))),
        yaxis2=dict(
            title=dict(text="Conversion Rate (%)", font=dict(color=COLOR_AMBER)),
            overlaying='y',
            side='right',
            gridcolor='rgba(0,0,0,0)',
            range=[0, max(df_visitor['conversion_rate']) * 1.35]
        )
    )
    return apply_base_layout(fig, height=360)


def plot_bounce_vs_exit(df: pd.DataFrame):
    """Vibrant scatter plot with marginal distributions."""
    sample_df = df.sample(min(1500, len(df)), random_state=42)
    fig = px.scatter(
        sample_df,
        x='BounceRates',
        y='ExitRates',
        color='Revenue_Label',
        color_discrete_map={'Purchase': COLOR_EMERALD, 'No Purchase': 'rgba(244, 63, 94, 0.5)'},
        opacity=0.8,
        title="<b>Friction Mapping: Bounce Rate vs Exit Rate</b>",
        labels={'BounceRates': 'Bounce Rate', 'ExitRates': 'Exit Rate'}
    )
    return apply_base_layout(fig, height=360)


def plot_duration_distribution(df: pd.DataFrame):
    """Multi-color histogram of product page duration."""
    clipped = df.copy()
    clipped['Duration_Clipped'] = np.clip(clipped['ProductRelated_Duration'] / 60, 0, 60)

    fig = px.histogram(
        clipped,
        x='Duration_Clipped',
        color='Revenue_Label',
        barmode='overlay',
        nbins=45,
        color_discrete_map={'Purchase': COLOR_EMERALD, 'No Purchase': COLOR_CORAL},
        opacity=0.7,
        title="<b>Product Page Dwell Time Distribution (Minutes, Capped at 60m)</b>",
        labels={'Duration_Clipped': 'Dwell Time on Product Pages (mins)'}
    )
    return apply_base_layout(fig, height=360)


def plot_traffic_source_performance(df_traffic: pd.DataFrame):
    """Horizontal bar chart with Turbo gradient ranking channels."""
    top_traffic = df_traffic[df_traffic['sessions'] >= 40].sort_values(by='conversion_rate', ascending=True)
    top_traffic['Traffic_Label'] = top_traffic['TrafficType'].apply(lambda x: f"Channel #{x}")

    fig = px.bar(
        top_traffic,
        x='conversion_rate',
        y='Traffic_Label',
        orientation='h',
        color='conversion_rate',
        color_continuous_scale=[[0, '#3b82f6'], [0.5, '#a855f7'], [1, '#10b981']],
        title="<b>Traffic Channels Ranked by Conversion Performance</b>",
        labels={'conversion_rate': 'Conversion Rate (%)', 'Traffic_Label': 'Acquisition Channel'},
        text=top_traffic['conversion_rate'].apply(lambda x: f"{x:.1f}%")
    )
    fig.update_traces(textposition='outside', textfont=dict(color='#cbd5e1'))
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis=dict(range=[0, max(top_traffic['conversion_rate']) * 1.25]))
    return apply_base_layout(fig, height=390, show_legend=False)


def plot_weekend_behavior(df_weekend: pd.DataFrame):
    """Vibrant grouped bar & line chart for Weekend vs Weekday analysis."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_weekend['Weekend_Label'],
        y=df_weekend['sessions'],
        name='Sessions',
        marker=dict(color='rgba(168, 85, 247, 0.75)', line=dict(color=COLOR_PURPLE, width=2)),
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=df_weekend['Weekend_Label'],
        y=df_weekend['conversion_rate'],
        name='Conversion Rate (%)',
        mode='lines+markers',
        line=dict(color=COLOR_EMERALD, width=3.5),
        marker=dict(size=12, color=COLOR_EMERALD, symbol='diamond'),
        yaxis='y2'
    ))
    fig.update_layout(
        title="<b>Weekend vs Weekday Shopping Intent & Conversion Lift</b>",
        yaxis=dict(title=dict(text="Total Sessions", font=dict(color=COLOR_PURPLE))),
        yaxis2=dict(
            title=dict(text="Conversion Rate (%)", font=dict(color=COLOR_EMERALD)),
            overlaying='y',
            side='right',
            gridcolor='rgba(0,0,0,0)',
            range=[0, max(df_weekend['conversion_rate']) * 1.45]
        )
    )
    return apply_base_layout(fig, height=360)


# ---------------- PAGE 4: MACHINE LEARNING MODELS ---------------- #

def plot_model_comparison_bar(metrics_dict: dict):
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1 for the 3 algorithms."""
    models = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    labels = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Balanced F1-Score']
    colors = [COLOR_CYAN, COLOR_PURPLE, COLOR_AMBER, COLOR_EMERALD]

    fig = go.Figure()
    for metric_key, label, color in zip(metric_names, labels, colors):
        values = [round(metrics_dict[m][metric_key] * 100, 2) for m in models]
        fig.add_trace(go.Bar(
            name=label,
            x=models,
            y=values,
            marker=dict(color=color),
            text=[f"{v:.1f}%" for v in values],
            textposition='auto',
            textfont=dict(color='#ffffff', size=11, family=FONT_FAMILY)
        ))

    fig.update_layout(
        barmode='group',
        title="<b>Comparative Algorithm Performance Benchmark (Holdout Test Set)</b>",
        yaxis=dict(title="Score (%)", range=[0, 105])
    )
    return apply_base_layout(fig, height=390)


def plot_multi_model_roc(test_sample, models_dict, preprocessor):
    """Plot multi-model ROC Curves with shaded areas and AUC comparisons."""
    if test_sample is None:
        return None

    X_test, y_test = test_sample
    X_test_trans = preprocessor.transform(X_test)

    colors = {
        'Logistic Regression': COLOR_CYAN,
        'Decision Tree': COLOR_PURPLE,
        'Random Forest': COLOR_EMERALD
    }

    fig = go.Figure()

    # Diagonal random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='#64748b', dash='dash', width=1.5),
        name='Random Chance (AUC = 0.50)'
    ))

    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test_trans)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            line=dict(color=colors.get(name, COLOR_CYAN), width=3),
            name=f'{name} (AUC = {roc_auc:.4f})'
        ))

    fig.update_layout(
        title="<b>Receiver Operating Characteristic (ROC) Multi-Model Curves</b>",
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate (Recall / Sensitivity)", range=[-0.02, 1.05])
    )
    return apply_base_layout(fig, height=380)


def plot_threshold_confusion_matrix(cm_data: list, model_name: str, threshold: float = 0.5):
    """Vibrant interactive confusion matrix heatmap."""
    z = cm_data
    x = ['Predicted: No Purchase', 'Predicted: Purchase']
    y = ['Actual: No Purchase', 'Actual: Purchase']

    total = sum(sum(row) for row in z)
    annotations = []
    labels_map = [
        ['True Negative (TN)', 'False Positive (FP)'],
        ['False Negative (FN)', 'True Positive (TP)']
    ]

    for i in range(2):
        for j in range(2):
            count = z[i][j]
            pct = (count / total) * 100
            label = labels_map[i][j]
            color = "#ffffff" if (i == j and count > 100) else "#cbd5e1"
            annotations.append(dict(
                x=x[j], y=y[i],
                text=f"<b>{count:,}</b><br><span style='font-size:11px;color:#cbd5e1;'>{label} ({pct:.1f}%)</span>",
                showarrow=False,
                font=dict(color=color, size=13, family=FONT_FAMILY)
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[
            [0.0, '#09132c'],
            [0.5, '#1e3a8a'],
            [1.0, '#0284c7']
        ],
        showscale=False
    ))

    fig.update_layout(
        title=f"<b>Confusion Matrix Heatmap ({model_name} @ Threshold {threshold:.2f})</b>",
        annotations=annotations,
        yaxis=dict(autorange='reversed')
    )
    return apply_base_layout(fig, height=360, show_legend=False)


# ---------------- PAGE 5: FEATURE IMPORTANCE ---------------- #

def plot_feature_importance_bar(feat_data: list, title="Feature Importance Analysis"):
    """Horizontal bar chart with neon electric cyan gradient."""
    df_feat = pd.DataFrame(feat_data).sort_values(by='importance', ascending=True)

    fig = px.bar(
        df_feat,
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale=[[0, '#1e3a8a'], [0.5, '#0284c7'], [1, '#00f2fe']],
        title=f"<b>{title}</b>",
        labels={'importance': 'Gini Importance Score', 'feature': 'Session Metric Node'},
        text=df_feat['importance'].apply(lambda x: f"{x*100:.1f}%")
    )
    fig.update_traces(textposition='outside', textfont=dict(color='#cbd5e1', size=11, family=FONT_FAMILY))
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis=dict(range=[0, max(df_feat['importance']) * 1.2]))
    return apply_base_layout(fig, height=450, show_legend=False)
