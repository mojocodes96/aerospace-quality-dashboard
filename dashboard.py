"""
dashboard.py
============
Aerospace Quality Analytics — Foundry-inspired dashboard.
Run with: py -m streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Quality Analytics | Foundry",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS THEME
# =============================================================================

st.markdown("""
<style>
.stApp { background-color: #2d2d2d; }

[data-testid="stSidebar"] {
    background-color: #0a1628 !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d6e8 !important; }
[data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }

.block-container {
    padding-top: 0 !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

.foundry-header {
    background: #0a1628;
    padding: 10px 20px;
    margin: -1rem -1.5rem 1.2rem -1.5rem;
    display: flex;
    align-items: center;
    border-bottom: 2px solid #0055cc;
}
.foundry-header .app-name {
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
}
.foundry-header .breadcrumb { font-size: 11px; color: #6b8aad; }
.foundry-header .header-right {
    margin-left: auto;
    font-size: 11px;
    color: #6b8aad;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 8px;
    margin-bottom: 12px;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-top: 3px solid #0055cc;
    border-radius: 4px;
    padding: 12px 14px;
}
.kpi-card.warn    { border-top-color: #d97706; }
.kpi-card.danger  { border-top-color: #dc2626; }
.kpi-card.success { border-top-color: #059669; }
.kpi-label {
    font-size: 10px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.kpi-value { font-size: 22px; font-weight: 700; color: #111827; line-height: 1.1; }
.kpi-sub   { font-size: 10px; color: #9ca3af; margin-top: 3px; }

.section-header {
    font-size: 11px;
    font-weight: 700;
    color: #c0c0c0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 6px 0 8px 0;
    border-bottom: 1px solid #4a4a4a;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# PLOTLY_LAYOUT — only keys that are NEVER overridden per chart
PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, sans-serif", size=11, color="#2d2d2d"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    data_dir = "data"
    if not os.path.exists(data_dir):
        st.error("No data folder found. Run pipeline.py first.")
        st.stop()
    return {
        "kpis":        pd.read_csv(f"{data_dir}/kpis.csv"),
        "defects":     pd.read_csv(f"{data_dir}/defects.csv",
                           parse_dates=["inspection_date", "run_date"]),
        "trend":       pd.read_csv(f"{data_dir}/trend.csv",
                           parse_dates=["year_month_dt"]),
        "suppliers":   pd.read_csv(f"{data_dir}/suppliers.csv"),
        "pareto":      pd.read_csv(f"{data_dir}/pareto_defect.csv"),
        "anomalies":   pd.read_csv(f"{data_dir}/anomaly_scores.csv"),
        "predictions": pd.read_csv(f"{data_dir}/predictions.csv"),
        "rca_factors": pd.read_csv(f"{data_dir}/rca_factors.csv"),
        "rca_summary": pd.read_csv(f"{data_dir}/rca_summary.csv"),
    }

data        = load_data()
kpis        = data["kpis"].iloc[0]
defects     = data["defects"]
trend       = data["trend"]
suppliers   = data["suppliers"]
predictions = data["predictions"]
rca_summary = data["rca_summary"]
rca_factors = data["rca_factors"]
anomalies   = data["anomalies"]


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### 🛡 Quality Analytics")
    st.markdown(
        "<div style='font-size:10px;color:#4a6a8a;margin-top:-8px'>"
        "Mock Quality Dashboard</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#8aa4c0;"
        "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px'>"
        "Filters</div>",
        unsafe_allow_html=True,
    )

    selected_line  = st.selectbox("Production line",
        ["All lines"]      + sorted(defects["production_line"].unique().tolist()))
    selected_shift = st.selectbox("Shift",
        ["All shifts"]     + sorted(defects["shift"].unique().tolist()))
    selected_sev   = st.selectbox("Severity",
        ["All severities"] + sorted(defects["severity"].unique().tolist()))
    selected_part  = st.selectbox("Part number",
        ["All parts"]      + sorted(defects["part_number"].unique().tolist()))

    st.divider()
    st.markdown(
        "<div style='font-size:10px;color:#4a6a8a'>ML Models</div>"
        "<div style='font-size:10px;color:#6b8aad'>Random Forest · AUC 0.622</div>"
        "<div style='font-size:10px;color:#6b8aad'>XGBoost · AUC 0.611</div>"
        "<div style='font-size:10px;color:#6b8aad'>Isolation Forest · 18%</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div style='font-size:10px;color:#4a6a8a'>"
        "18 months · simulated aerospace data</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# FILTER LOGIC
# =============================================================================

mask = pd.Series([True] * len(defects))
if selected_line  != "All lines":      mask &= (defects["production_line"] == selected_line)
if selected_shift != "All shifts":     mask &= (defects["shift"]           == selected_shift)
if selected_sev   != "All severities": mask &= (defects["severity"]        == selected_sev)
if selected_part  != "All parts":      mask &= (defects["part_number"]     == selected_part)

df = defects[mask].copy()


# =============================================================================
# HEADER BAR
# =============================================================================

st.markdown(f"""
<div class="foundry-header">
    <div>
        <div class="app-name">🛡 Aerospace Quality Analytics</div>
        <div class="breadcrumb">Defense Hardware · Manufacturing Quality · Live pipeline</div>
    </div>
    <div class="header-right">
        {len(df):,} defects &nbsp;|&nbsp;
        Line: {selected_line} &nbsp;|&nbsp;
        Shift: {selected_shift} &nbsp;|&nbsp;
        Severity: {selected_sev}
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# KPI CARDS
# =============================================================================

pass_rate     = kpis["overall_pass_rate"]
defect_cost   = kpis["total_defect_cost"]
critical_ct   = int(kpis["critical_defect_count"])
open_cas      = int(kpis["open_corrective_actions"])
yield_pct     = kpis["overall_yield_pct"]
total_defects = int(kpis["total_defects"])

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card {'success' if pass_rate >= 85 else 'warn'}">
        <div class="kpi-label">Overall pass rate</div>
        <div class="kpi-value">{pass_rate}%</div>
        <div class="kpi-sub">Target: 85%</div>
    </div>
    <div class="kpi-card danger">
        <div class="kpi-label">Total defect cost</div>
        <div class="kpi-value">${defect_cost:,.0f}</div>
        <div class="kpi-sub">{total_defects} defects total</div>
    </div>
    <div class="kpi-card {'danger' if critical_ct > 0 else 'success'}">
        <div class="kpi-label">Critical defects</div>
        <div class="kpi-value">{critical_ct}</div>
        <div class="kpi-sub">Requires immediate action</div>
    </div>
    <div class="kpi-card {'warn' if open_cas > 5 else 'success'}">
        <div class="kpi-label">Open corrective actions</div>
        <div class="kpi-value">{open_cas}</div>
        <div class="kpi-sub">Awaiting closure</div>
    </div>
    <div class="kpi-card {'success' if yield_pct >= 95 else 'warn'}">
        <div class="kpi-label">Production yield</div>
        <div class="kpi-value">{yield_pct}%</div>
        <div class="kpi-sub">Target: 95%</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# QUALITY TRENDS
# =============================================================================

st.markdown('<div class="section-header">Quality trends</div>', unsafe_allow_html=True)

col_trend, col_cost = st.columns([3, 2])

with col_trend:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend["year_month_dt"], y=trend["pass_rate_pct"],
        mode="lines+markers", name="Monthly",
        line=dict(color="#9ca3af", width=1.2), marker=dict(size=3),
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend["year_month_dt"], y=trend["pass_rate_rolling_3m"],
        mode="lines", name="3-month avg",
        line=dict(color="#0055cc", width=2),
    ))
    fig_trend.add_hline(
        y=85, line_dash="dot", line_color="#dc2626",
        annotation_text="85% target", annotation_font_size=10,
    )
    fig_trend.update_layout(
        **PLOTLY_LAYOUT,
        title="Pass rate over time (%)",
        height=240,
        yaxis_range=[50, 100],
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
        xaxis=dict(gridcolor="#f0f2f5", linecolor="#dde3ec", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f0f2f5", linecolor="#dde3ec", tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_cost:
    fig_cost = px.bar(
        trend, x="year_month_dt", y="defect_cost",
        title="Monthly defect cost ($)",
        color_discrete_sequence=["#dc2626"],
    )
    fig_cost.update_layout(
        **PLOTLY_LAYOUT,
        height=240,
        showlegend=False,
        margin=dict(l=8, r=8, t=36, b=8),
        xaxis=dict(title="", gridcolor="#f0f2f5", linecolor="#dde3ec", tickfont=dict(size=10)),
        yaxis=dict(title="Cost ($)", gridcolor="#f0f2f5", linecolor="#dde3ec", tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_cost, use_container_width=True)


# =============================================================================
# FAILURE BREAKDOWN
# =============================================================================

st.markdown('<div class="section-header">Failure analysis</div>', unsafe_allow_html=True)

line_counts  = df.groupby("production_line").size().reset_index(name="defect_count")
shift_counts = df.groupby("shift").size().reset_index(name="defect_count")
sev_counts   = df["severity"].value_counts().reset_index()
sev_counts.columns = ["severity", "count"]

col1, col2, col3 = st.columns(3)

with col1:
    fig_line = px.bar(
        line_counts.sort_values("defect_count"),
        x="defect_count", y="production_line", orientation="h",
        title="Defects by production line",
        color="defect_count",
        color_continuous_scale=["#059669", "#d97706", "#dc2626"],
        text="defect_count",
    )
    fig_line.update_traces(textposition="outside")
    fig_line.update_layout(
        **PLOTLY_LAYOUT,
        height=240,
        coloraxis_showscale=False,
        margin=dict(l=8, r=50, t=36, b=8),
        xaxis=dict(title="Defect count", gridcolor="#f0f2f5", tickfont=dict(size=10)),
        yaxis=dict(title="", gridcolor="#f0f2f5", tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    fig_shift = px.bar(
        shift_counts.sort_values("defect_count"),
        x="defect_count", y="shift", orientation="h",
        title="Defects by shift",
        color="defect_count",
        color_continuous_scale=["#059669", "#d97706", "#dc2626"],
        text="defect_count",
    )
    fig_shift.update_traces(textposition="outside")
    fig_shift.update_layout(
        **PLOTLY_LAYOUT,
        height=240,
        coloraxis_showscale=False,
        margin=dict(l=8, r=50, t=36, b=8),
        xaxis=dict(title="Defect count", gridcolor="#f0f2f5", tickfont=dict(size=10)),
        yaxis=dict(title="", gridcolor="#f0f2f5", tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_shift, use_container_width=True)

with col3:
    fig_sev = px.pie(
        sev_counts, names="severity", values="count",
        title="Severity distribution", hole=0.55,
        color="severity",
        color_discrete_map={
            "Critical": "#dc2626",
            "Major":    "#d97706",
            "Minor":    "#059669",
        },
    )
    fig_sev.update_layout(
        **PLOTLY_LAYOUT,
        height=240,
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
    )
    st.plotly_chart(fig_sev, use_container_width=True)


# =============================================================================
# PARETO CHART
# =============================================================================

st.markdown('<div class="section-header">Pareto analysis — defect types</div>', unsafe_allow_html=True)

pareto_filtered = (
    df.groupby("defect_type")
    .agg(count=("defect_id", "count"), total_cost=("cost_impact", "sum"))
    .reset_index()
    .sort_values("count", ascending=False)
)

if len(pareto_filtered) > 0:
    pareto_filtered["cumulative_pct"] = (
        pareto_filtered["count"].cumsum() / pareto_filtered["count"].sum() * 100
    ).round(1)

    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(
        x=pareto_filtered["defect_type"],
        y=pareto_filtered["count"],
        name="Count", marker_color="#0055cc", yaxis="y",
    ))
    fig_pareto.add_trace(go.Scatter(
        x=pareto_filtered["defect_type"],
        y=pareto_filtered["cumulative_pct"],
        name="Cumulative %", mode="lines+markers",
        line=dict(color="#dc2626", width=1.8),
        marker=dict(size=5), yaxis="y2",
    ))
    fig_pareto.add_hline(
        y=80, line_dash="dot", line_color="#9ca3af",
        yref="y2", annotation_text="80%", annotation_font_size=10,
    )
    fig_pareto.update_layout(
        **PLOTLY_LAYOUT,
        title="Which defect types drive the most volume?",
        height=300,
        margin=dict(l=8, r=60, t=36, b=80),
        xaxis=dict(tickangle=-30, gridcolor="#f0f2f5", tickfont=dict(size=10)),
        yaxis=dict(title="Count", side="left", gridcolor="#f0f2f5"),
        yaxis2=dict(
            title="Cumulative %", side="right",
            overlaying="y", range=[0, 110], ticksuffix="%",
        ),
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
    )
    st.plotly_chart(fig_pareto, use_container_width=True)
else:
    st.info("No defects match the current filters.")


# =============================================================================
# ML INSIGHTS
# =============================================================================

st.markdown('<div class="section-header">Machine learning insights</div>', unsafe_allow_html=True)

tab_rca, tab_pred, tab_anom = st.tabs([
    "Root cause analysis (XGBoost)",
    "Defect risk prediction (Random Forest)",
    "Anomaly detection (Isolation Forest)",
])

with tab_rca:
    col_l, col_r = st.columns(2)

    with col_l:
        fig_rca = px.bar(
            rca_summary.sort_values("total_gain_pct"),
            x="total_gain_pct", y="category", orientation="h",
            title="Risk contribution by category",
            color="total_gain_pct",
            color_continuous_scale=["#bfdbfe", "#0055cc", "#0a1628"],
            text="total_gain_pct",
        )
        fig_rca.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_rca.update_layout(
            **PLOTLY_LAYOUT,
            height=280,
            coloraxis_showscale=False,
            margin=dict(l=8, r=60, t=36, b=8),
            xaxis=dict(title="% of model gain", gridcolor="#f0f2f5", tickfont=dict(size=10)),
            yaxis=dict(title="", gridcolor="#f0f2f5", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_rca, use_container_width=True)

    with col_r:
        fig_feat = px.bar(
            rca_factors.head(15).sort_values("gain_pct"),
            x="gain_pct", y="feature", orientation="h",
            title="Top 15 individual risk factors",
            color="category", text="gain_pct",
        )
        fig_feat.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_feat.update_layout(
            **PLOTLY_LAYOUT,
            height=280,
            margin=dict(l=8, r=60, t=36, b=8),
            xaxis=dict(title="% of model gain", gridcolor="#f0f2f5", tickfont=dict(size=10)),
            yaxis=dict(title="", gridcolor="#f0f2f5", tickfont=dict(size=10)),
            legend=dict(orientation="h", y=-0.3, font=dict(size=9)),
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    st.info(
        "**Finding:** Production Line and Shift are the top controllable risk factors. "
        "Line-Charlie and Night shift appear consistently as primary failure drivers. "
        "**Recommended action:** Targeted process audit of Line-Charlie Night shift operations."
    )

with tab_pred:
    col_l, col_r = st.columns(2)

    with col_l:
        risk_counts = predictions["risk_tier"].value_counts().reset_index()
        risk_counts.columns = ["risk_tier", "count"]
        risk_counts = risk_counts.sort_values(
            "risk_tier",
            key=lambda x: x.map({"High": 0, "Medium": 1, "Low": 2}),
        )
        fig_risk = px.bar(
            risk_counts, x="risk_tier", y="count",
            title="Inspections by predicted risk tier",
            color="risk_tier",
            color_discrete_map={
                "High": "#dc2626", "Medium": "#d97706", "Low": "#059669",
            },
            text="count",
        )
        fig_risk.update_traces(textposition="outside")
        fig_risk.update_layout(
            **PLOTLY_LAYOUT,
            height=260,
            showlegend=False,
            margin=dict(l=8, r=8, t=36, b=8),
            xaxis=dict(title="Risk tier", gridcolor="#f0f2f5", tickfont=dict(size=10)),
            yaxis=dict(title="Count", gridcolor="#f0f2f5", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_r:
        fig_hist = px.histogram(
            predictions, x="fail_probability", nbins=25,
            title="Distribution of predicted fail probability",
            color_discrete_sequence=["#0055cc"],
        )
        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            height=260,
            margin=dict(l=8, r=8, t=36, b=8),
            xaxis=dict(title="Predicted fail probability", gridcolor="#f0f2f5", tickfont=dict(size=10)),
            yaxis=dict(title="Count", gridcolor="#f0f2f5", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with tab_anom:
    col_l, col_r = st.columns(2)

    with col_l:
        fig_anom = px.histogram(
            anomalies, x="anomaly_score_pct",
            color=anomalies["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
            nbins=30,
            title="Anomaly score distribution",
            color_discrete_map={"Normal": "#bfdbfe", "Anomaly": "#dc2626"},
            barmode="overlay",
            labels={"color": "Classification"},
        )
        fig_anom.update_layout(
            **PLOTLY_LAYOUT,
            height=260,
            margin=dict(l=8, r=8, t=36, b=8),
            xaxis=dict(title="Anomaly score (higher = more unusual)", gridcolor="#f0f2f5", tickfont=dict(size=10)),
            yaxis=dict(title="Count", gridcolor="#f0f2f5", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_anom, use_container_width=True)

    with col_r:
        tp = int(((anomalies["is_anomaly"]==1) & (anomalies["target_fail"]==1)).sum())
        fp = int(((anomalies["is_anomaly"]==1) & (anomalies["target_fail"]==0)).sum())
        fn = int(((anomalies["is_anomaly"]==0) & (anomalies["target_fail"]==1)).sum())
        tn = int(((anomalies["is_anomaly"]==0) & (anomalies["target_fail"]==0)).sum())
        recall    = round(100 * tp / (tp + fn), 1) if (tp + fn) > 0 else 0
        precision = round(100 * tp / (tp + fp), 1) if (tp + fp) > 0 else 0

        st.markdown("**Model performance vs actual failures**")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("True positives",  tp, help="Actual failures correctly flagged")
            st.metric("False negatives", fn, help="Missed failures")
        with m2:
            st.metric("True negatives",  tn, help="Normal inspections correctly cleared")
            st.metric("False positives", fp, help="False alarms")
        st.markdown(
            f"- **Recall:** {recall}% of actual failures caught\n"
            f"- **Precision:** {precision}% of flagged items were real failures\n"
            f"- **Model:** Isolation Forest · contamination=0.18 · unsupervised"
        )


# =============================================================================
# SUPPLIER SCORECARD
# =============================================================================

st.markdown('<div class="section-header">Supplier quality scorecard</div>', unsafe_allow_html=True)

def highlight_supplier_rows(row):
    rate = row["fail_rate_pct"]
    if rate > 20:
        return ["background-color: #fee2e2; color: #991b1b"] * len(row)
    elif rate > 12:
        return ["background-color: #fef3c7; color: #92400e"] * len(row)
    return [""] * len(row)

sup_display = suppliers[[
    "supplier_name", "tier", "country", "approved_status",
    "total_inspections", "fail_rate_pct",
    "total_defects", "total_defect_cost", "avg_visual_score",
]].sort_values("fail_rate_pct", ascending=False).reset_index(drop=True)
sup_display.index += 1

st.dataframe(
    sup_display.style.apply(highlight_supplier_rows, axis=1),
    use_container_width=True,
    height=300,
)


# =============================================================================
# DEFECT DETAIL LOG
# =============================================================================

st.markdown(
    f'<div class="section-header">Defect log — {len(df):,} records</div>',
    unsafe_allow_html=True,
)

detail_cols = [
    "defect_id", "defect_type", "severity", "disposition", "cost_impact",
    "production_line", "shift", "part_number",
    "supplier_name", "inspection_type", "inspection_date",
]
st.dataframe(
    df[detail_cols].sort_values("cost_impact", ascending=False).reset_index(drop=True),
    use_container_width=True,
    height=380,
)

st.markdown("""
<div style='font-size:10px;color:#9ca3af;text-align:center;padding:16px 0 8px'>
    Aerospace Quality Analytics &nbsp;·&nbsp;
    Python · Pandas · scikit-learn · XGBoost · Plotly · Streamlit &nbsp;·&nbsp;
    Foundry Workshop-inspired layout &nbsp;·&nbsp; Portfolio project
</div>
""", unsafe_allow_html=True)