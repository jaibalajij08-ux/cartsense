"""
CartSense AI - Ultra-Vibrant Enterprise UI Design System
--------------------------------------------------------
Provides a high-energy, colorful, glassmorphic design system with
multi-color KPI cards, neon glows, animated live status tickers,
actionable recommendation cards, and enterprise SaaS aesthetics.
"""

import streamlit as st


def apply_custom_css():
    """Inject ultra-vibrant, colorful enterprise CSS."""
    custom_css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main background with subtle colorful gradient mesh */
    .stApp {
        background: radial-gradient(circle at 10% 15%, rgba(30, 58, 138, 0.25) 0%, transparent 40%),
                    radial-gradient(circle at 85% 20%, rgba(124, 58, 237, 0.2) 0%, transparent 45%),
                    radial-gradient(circle at 50% 85%, rgba(16, 185, 129, 0.15) 0%, transparent 40%),
                    #050914;
        color: #f8fafc;
    }

    /* Top Enterprise App Navbar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        padding: 12px 20px;
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        margin-bottom: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .nav-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.3px;
    }

    .nav-brand-badge {
        font-size: 0.7rem;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        background: rgba(14, 165, 233, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .nav-status-group {
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 0.82rem;
        color: #94a3b8;
    }

    .pulse-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981, 0 0 18px #10b981;
        animation: pulseAnimation 2s infinite;
        margin-right: 6px;
    }

    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Hero Page Header */
    .app-hero-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 58, 138, 0.5) 50%, rgba(124, 58, 237, 0.3) 100%);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.55), 0 0 25px -10px rgba(56, 189, 248, 0.2);
        position: relative;
        overflow: hidden;
    }

    .app-hero-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #f43f5e, #10b981);
    }

    .app-hero-title {
        font-size: 2.15rem;
        font-weight: 900;
        background: linear-gradient(120deg, #ffffff 0%, #bae6fd 40%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.6px;
        margin-bottom: 6px;
    }

    .app-hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.45;
    }

    /* Distinct Vibrant KPI Cards */
    .kpi-card {
        border-radius: 16px;
        padding: 22px 24px;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
    }

    .kpi-card:hover {
        transform: translateY(-4px);
    }

    /* Blue / Cyan Card */
    .kpi-blue {
        background: linear-gradient(145deg, rgba(14, 32, 64, 0.9) 0%, rgba(3, 105, 161, 0.4) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: 0 8px 25px -4px rgba(2, 132, 199, 0.3);
    }
    .kpi-blue:hover {
        border-color: #38bdf8;
        box-shadow: 0 12px 30px -4px rgba(56, 189, 248, 0.45);
    }

    /* Emerald / Mint Card */
    .kpi-emerald {
        background: linear-gradient(145deg, rgba(6, 44, 38, 0.9) 0%, rgba(5, 150, 105, 0.35) 100%);
        border: 1px solid rgba(52, 211, 153, 0.35);
        box-shadow: 0 8px 25px -4px rgba(16, 185, 129, 0.25);
    }
    .kpi-emerald:hover {
        border-color: #34d399;
        box-shadow: 0 12px 30px -4px rgba(52, 211, 153, 0.45);
    }

    /* Purple / Violet Card */
    .kpi-purple {
        background: linear-gradient(145deg, rgba(30, 20, 60, 0.9) 0%, rgba(124, 58, 237, 0.35) 100%);
        border: 1px solid rgba(168, 85, 247, 0.35);
        box-shadow: 0 8px 25px -4px rgba(139, 92, 246, 0.25);
    }
    .kpi-purple:hover {
        border-color: #a855f7;
        box-shadow: 0 12px 30px -4px rgba(168, 85, 247, 0.45);
    }

    /* Coral / Rose Card */
    .kpi-coral {
        background: linear-gradient(145deg, rgba(50, 16, 30, 0.9) 0%, rgba(225, 29, 72, 0.35) 100%);
        border: 1px solid rgba(244, 63, 94, 0.35);
        box-shadow: 0 8px 25px -4px rgba(244, 63, 94, 0.25);
    }
    .kpi-coral:hover {
        border-color: #fb7185;
        box-shadow: 0 12px 30px -4px rgba(244, 63, 94, 0.45);
    }

    /* Amber / Sunset Card */
    .kpi-amber {
        background: linear-gradient(145deg, rgba(50, 32, 10, 0.9) 0%, rgba(217, 119, 6, 0.35) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        box-shadow: 0 8px 25px -4px rgba(245, 158, 11, 0.25);
    }
    .kpi-amber:hover {
        border-color: #fbbf24;
        box-shadow: 0 12px 30px -4px rgba(245, 158, 11, 0.45);
    }

    .kpi-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .kpi-card-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #cbd5e1;
    }

    .kpi-card-icon {
        font-size: 1.4rem;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px);
    }

    .kpi-card-value {
        font-size: 2.25rem;
        font-weight: 900;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 6px;
        letter-spacing: -0.6px;
    }

    .kpi-card-subtitle {
        font-size: 0.82rem;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
    }

    .kpi-delta-pill {
        display: inline-flex;
        align-items: center;
        padding: 2px 7px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .delta-pos {
        background: rgba(16, 185, 129, 0.25);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .delta-neg {
        background: rgba(244, 63, 94, 0.25);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }

    /* Section Cards */
    .section-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(20, 30, 55, 0.75) 100%);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
    }

    .section-card-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Prediction Result Banner - Likely (Radiant Emerald) */
    .prediction-banner-likely {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.85) 0%, rgba(13, 148, 136, 0.6) 50%, rgba(16, 185, 129, 0.2) 100%);
        border: 2px solid #10b981;
        border-radius: 18px;
        padding: 28px 32px;
        margin: 20px 0;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.35), inset 0 0 20px rgba(16, 185, 129, 0.15);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        position: relative;
    }

    /* Prediction Result Banner - Unlikely (Radiant Crimson) */
    .prediction-banner-unlikely {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.85) 0%, rgba(185, 28, 28, 0.5) 50%, rgba(239, 68, 68, 0.15) 100%);
        border: 2px solid #ef4444;
        border-radius: 18px;
        padding: 28px 32px;
        margin: 20px 0;
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.35), inset 0 0 20px rgba(239, 68, 68, 0.15);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        position: relative;
    }

    .banner-title {
        font-size: 1.85rem;
        font-weight: 900;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .banner-prob {
        font-size: 2.9rem;
        font-weight: 900;
        margin: 6px 0;
        font-family: 'JetBrains Mono', monospace;
    }

    .banner-desc {
        font-size: 0.95rem;
        max-width: 680px;
        line-height: 1.55;
        color: #e2e8f0;
    }

    /* Action Engine Cards */
    .action-tile {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-left: 4px solid #818cf8;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .action-tile:hover {
        transform: translateX(4px);
        border-color: #a855f7;
    }

    .action-tile-title {
        font-weight: 700;
        color: #f8fafc;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }

    .action-tile-desc {
        font-size: 0.84rem;
        color: #94a3b8;
        line-height: 1.45;
    }

    /* Badges & Pills */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .badge-primary {
        background: linear-gradient(90deg, rgba(14, 165, 233, 0.25) 0%, rgba(99, 102, 241, 0.25) 100%);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.45);
    }

    .badge-success {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.25) 0%, rgba(5, 150, 105, 0.25) 100%);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.45);
    }

    .badge-warning {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.25) 0%, rgba(217, 119, 6, 0.25) 100%);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.45);
    }

    .badge-purple {
        background: linear-gradient(90deg, rgba(168, 85, 247, 0.25) 0%, rgba(139, 92, 246, 0.25) 100%);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.45);
    }

    /* Buttons with Neon Gradients */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #6366f1 50%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 1px solid rgba(168, 85, 247, 0.4) !important;
        border-radius: 12px !important;
        padding: 11px 26px !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.02em !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #4f46e5 50%, #7c3aed 100%) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.55), 0 0 14px rgba(56, 189, 248, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070d1e 0%, #0b142a 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.18) !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.7);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(51, 65, 85, 0.5);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.4) 0%, rgba(99, 102, 241, 0.4) 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 2px 10px rgba(56, 189, 248, 0.25);
    }

    /* Footer styling */
    .app-footer {
        text-align: center;
        padding: 30px 10px 15px 10px;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid rgba(51, 65, 85, 0.4);
        margin-top: 45px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_top_navbar():
    """Render a modern SaaS top navigation status bar."""
    html = """
    <div class="top-navbar">
        <div class="nav-brand">
            <span style="font-size: 1.5rem;">🛒</span>
            <span class="nav-brand-title">CartSense AI</span>
            <span class="nav-brand-badge">Enterprise v2.4</span>
        </div>
        <div class="nav-status-group">
            <div><span class="pulse-dot"></span><b style="color: #34d399;">ML Cluster Active</b></div>
            <div>⚡ <b>Latency:</b> <span style="color: #38bdf8;">11ms</span></div>
            <div>🎯 <b>Model:</b> <span style="color: #c084fc;">Random Forest</span></div>
            <div>📊 <b>Data:</b> <span style="color: #cbd5e1;">12,330 Sessions</span></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_hero_header(title, subtitle, badge_text="Production Engine"):
    """Render high-impact gradient hero page banner."""
    html = f"""
    <div class="app-hero-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
            <div>
                <div class="app-hero-title">{title}</div>
                <div class="app-hero-subtitle">{subtitle}</div>
            </div>
            <div>
                <span class="pill-badge badge-primary">{badge_text}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(title, value, subtitle="", icon="📊", delta=None, delta_positive=True, color_variant="blue"):
    """
    Generate a high-vibrancy metric card.
    color_variant options: 'blue', 'emerald', 'purple', 'coral', 'amber'
    """
    delta_html = ""
    if delta:
        delta_class = "delta-pos" if delta_positive else "delta-neg"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<span class="kpi-delta-pill {delta_class}">{arrow} {delta}</span>'

    card_class = f"kpi-card kpi-{color_variant}"

    html = f"""
    <div class="{card_class}">
        <div class="kpi-card-header">
            <span class="kpi-card-title">{title}</span>
            <span class="kpi-card-icon">{icon}</span>
        </div>
        <div class="kpi-card-value">{value}</div>
        <div class="kpi-card-subtitle">{delta_html} {subtitle}</div>
    </div>
    """
    return html


def render_prediction_banner(is_likely: bool, probability: float, confidence_text: str):
    """Render vibrant, glowing prediction outcome banner."""
    pct = probability * 100
    if is_likely:
        return f"""
        <div class="prediction-banner-likely">
            <div style="font-size: 3rem; margin-bottom: 4px;">🎉</div>
            <div class="banner-title" style="color: #34d399;">PURCHASE LIKELY</div>
            <div class="banner-prob" style="color: #6ee7b7;">{pct:.1f}%</div>
            <div style="color: #a7f3d0; font-weight: 700; margin-bottom: 10px; font-size: 0.95rem;">
                Confidence: <span style="background: rgba(16, 185, 129, 0.3); padding: 3px 10px; border-radius: 9999px;">{confidence_text}</span>
            </div>
            <div class="banner-desc">
                High commercial transaction intent detected. Session characteristics indicate an active shopper ready for checkout.
            </div>
        </div>
        """
    else:
        return f"""
        <div class="prediction-banner-unlikely">
            <div style="font-size: 3rem; margin-bottom: 4px;">🔍</div>
            <div class="banner-title" style="color: #f87171;">PURCHASE UNLIKELY</div>
            <div class="banner-prob" style="color: #fca5a5;">{pct:.1f}%</div>
            <div style="color: #fecaca; font-weight: 700; margin-bottom: 10px; font-size: 0.95rem;">
                Confidence: <span style="background: rgba(239, 68, 68, 0.3); padding: 3px 10px; border-radius: 9999px;">{confidence_text}</span>
            </div>
            <div class="banner-desc">
                Browsing signals indicate casual discovery or checkout friction. Proactive engagement interventions recommended.
            </div>
        </div>
        """


def render_action_tile(title, desc, icon="⚡"):
    """Render real-time e-commerce intervention recommendation tile."""
    return f"""
    <div class="action-tile">
        <div class="action-tile-title"><span>{icon}</span> {title}</div>
        <div class="action-tile-desc">{desc}</div>
    </div>
    """


def render_footer():
    """Render application footer."""
    html = """
    <div class="app-footer">
        <b>CartSense AI Enterprise</b> &bull; Intelligent Customer Purchase Intention Prediction Platform<br>
        Developed with Streamlit, Scikit-learn & Plotly &bull; UCI Machine Learning Repository
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
