import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sentinel NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_FILE = os.path.join(
    BASE_DIR,
    "logs",
    "realtime_detection.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

METRICS_FILE = os.path.join(
    RESULTS_DIR,
    "FINAL_VALIDATION_REPORT.txt"
)

SHAP_FILE = os.path.join(
    RESULTS_DIR,
    "shap_feature_importance.csv"
)

ATTACK_METRICS_FILE = os.path.join(
    RESULTS_DIR,
    "attack_classification_metrics.txt"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 20% 0%,
                rgba(20, 80, 120, 0.16),
                transparent 30%
            ),
            #070b12;
        color: #e8eef7;
    }

    [data-testid="stSidebar"] {
        background: #0a0f17;
        border-right: 1px solid #1b2635;
    }

    [data-testid="stSidebar"] * {
        color: #d8e2ef;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* ---------- HEADER ---------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 4px;
    }

    .brand-icon {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        background: linear-gradient(135deg, #16263a, #0d1725);
        border: 1px solid #263b52;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }

    .brand-title {
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .brand-subtitle {
        color: #7f91a8;
        font-size: 13px;
        margin-top: -3px;
    }

    /* ---------- STATUS ---------- */

    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border-radius: 20px;
        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.25);
        color: #4ade80;
        font-size: 12px;
        font-weight: 700;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(34,197,94,0.8);
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 18px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 12px;
        color: #edf4fc;
    }

    .section-caption {
        color: #718198;
        font-size: 12px;
        margin-top: -7px;
        margin-bottom: 15px;
    }

    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        background: linear-gradient(
            145deg,
            rgba(19,29,43,0.95),
            rgba(11,18,28,0.95)
        );

        border: 1px solid #1d2b3d;
        border-radius: 15px;
        padding: 19px;
        min-height: 125px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.16);
    }

    .metric-label {
        color: #718198;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 650;
    }

    .metric-value {
        color: #f1f6fc;
        font-size: 30px;
        font-weight: 800;
        margin-top: 9px;
    }

    .metric-small {
        color: #718198;
        font-size: 11px;
        margin-top: 5px;
    }

    .metric-green {
        color: #4ade80;
    }

    .metric-red {
        color: #fb7185;
    }

    .metric-yellow {
        color: #facc15;
    }

    /* ---------- PANELS ---------- */

    .panel {
        background: rgba(11,18,28,0.78);
        border: 1px solid #1b293b;
        border-radius: 15px;
        padding: 18px;
    }

    /* ---------- ALERT ---------- */

    .alert-high {
        border-left: 4px solid #ef4444;
        background: rgba(239,68,68,0.07);
        padding: 12px 15px;
        border-radius: 7px;
        margin-bottom: 8px;
    }

    .alert-medium {
        border-left: 4px solid #f59e0b;
        background: rgba(245,158,11,0.07);
        padding: 12px 15px;
        border-radius: 7px;
        margin-bottom: 8px;
    }

    .alert-low {
        border-left: 4px solid #22c55e;
        background: rgba(34,197,94,0.07);
        padding: 12px 15px;
        border-radius: 7px;
        margin-bottom: 8px;
    }

    .alert-title {
        font-weight: 700;
        font-size: 13px;
    }

    .alert-text {
        color: #8493a8;
        font-size: 11px;
        margin-top: 3px;
    }

    /* ---------- SIDEBAR ---------- */

    .sidebar-logo {
        text-align: center;
        padding: 10px 0 25px 0;
    }

    .sidebar-logo-icon {
        font-size: 42px;
    }

    .sidebar-logo-title {
        font-size: 20px;
        font-weight: 800;
        margin-top: 4px;
    }

    .sidebar-logo-subtitle {
        font-size: 11px;
        color: #65758a;
    }

    .sidebar-info {
        background: #0e1622;
        border: 1px solid #1b293a;
        border-radius: 12px;
        padding: 12px;
        font-size: 11px;
        color: #75869b;
        line-height: 1.6;
    }

    /* ---------- TABLE ---------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #1b293b;
        border-radius: 10px;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 9px;
        border: 1px solid #26384e;
        background: #101a28;
        color: #dce7f4;
        font-weight: 650;
    }

    .stButton > button:hover {
        border-color: #3b82f6;
        color: #ffffff;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #53657b;
        font-size: 11px;
        padding-top: 30px;
        border-top: 1px solid #172334;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

def normalize_columns(df):
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )
    return df


@st.cache_data(ttl=2)
def load_logs():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(LOG_FILE)

        if df.empty:
            return df

        df = normalize_columns(df)

        # Normalize common column names
        rename_map = {
            "Time": "Timestamp",
            "Date_Time": "Timestamp",
            "Src_IP": "Source_IP",
            "Dst_IP": "Destination_IP",
            "Src_Port": "Source_Port",
            "Dst_Port": "Destination_Port",
            "Attack_Probability": "Probability",
            "Prob": "Probability",
        }

        df.rename(
            columns={
                k: v for k, v in rename_map.items()
                if k in df.columns
            },
            inplace=True,
        )

        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"],
                errors="coerce"
            )

        if "Probability" in df.columns:
            df["Probability"] = pd.to_numeric(
                df["Probability"],
                errors="coerce"
            ).fillna(0)

        if "Prediction" in df.columns:
            df["Prediction"] = (
                df["Prediction"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

        if "Attack_Type" in df.columns:
            df["Attack_Type"] = (
                df["Attack_Type"]
                .astype(str)
                .str.strip()
            )

        return df

    except Exception as e:
        st.error(f"Unable to read detection log: {e}")
        return pd.DataFrame()


def read_text(path):
    if not os.path.exists(path):
        return "Not available."

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return "Not available."


def model_exists(filename):
    return os.path.exists(os.path.join(MODELS_DIR, filename))


def get_attack_count(df):
    if df.empty or "Prediction" not in df.columns:
        return 0

    return int(
        (df["Prediction"].str.upper() == "ATTACK").sum()
    )


def get_benign_count(df):
    if df.empty or "Prediction" not in df.columns:
        return 0

    return int(
        (df["Prediction"].str.upper() == "BENIGN").sum()
    )


def probability_value(value):
    try:
        value = float(value)

        if value <= 1:
            return value * 100

        return value

    except Exception:
        return 0


# ============================================================
# LOAD DATA
# ============================================================

df = load_logs()

total_flows = len(df)
attack_count = get_attack_count(df)
benign_count = get_benign_count(df)

if total_flows > 0:
    attack_rate = (attack_count / total_flows) * 100
else:
    attack_rate = 0

if not df.empty and "Probability" in df.columns:
    max_probability = df["Probability"].apply(
        probability_value
    ).max()
else:
    max_probability = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🛡️</div>
            <div class="sidebar-logo-title">SENTINEL NIDS</div>
            <div class="sidebar-logo-subtitle">
                AI Network Security Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Live Detection",
            "Threat Alerts",
            "Analytics",
            "AI Explainability",
            "Model Performance",
            "System Status",
        ],
    )

    st.markdown("---")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    auto_refresh = st.toggle(
        "Auto Refresh",
        value=False
    )

    refresh_seconds = st.selectbox(
        "Refresh Interval",
        [2, 5, 10, 30],
        index=1,
        disabled=not auto_refresh
    )

    st.markdown("---")

    st.markdown(
        f"""
        <div class="sidebar-info">
        <b>Detection Engine</b><br>
        Random Forest<br><br>

        <b>Binary Detection</b><br>
        BENIGN / ATTACK<br><br>

        <b>Classification</b><br>
        Multi-class attack detection<br><br>

        <b>Live Capture</b><br>
        Scapy + Npcap
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [7, 2],
    vertical_alignment="center"
)

with header_left:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <div>
                <div class="brand-title">
                    Sentinel NIDS
                </div>
                <div class="brand-subtitle">
                    AI-Powered Network Intrusion Detection & Threat Intelligence
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:

    st.markdown(
        """
        <div style="text-align:right;">
            <div class="status-online">
                <span class="status-dot"></span>
                DETECTION SYSTEM ONLINE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Security Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Real-time network activity monitored by the AI detection engine'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Flows</div>
                <div class="metric-value">{total_flows:,}</div>
                <div class="metric-small">
                    Processed network flows
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Benign Traffic</div>
                <div class="metric-value metric-green">
                    {benign_count:,}
                </div>
                <div class="metric-small">
                    Normal connections
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Detected Attacks</div>
                <div class="metric-value metric-red">
                    {attack_count:,}
                </div>
                <div class="metric-small">
                    Malicious connections
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        if max_probability >= 80:
            risk = "HIGH"
            risk_class = "metric-red"
        elif max_probability >= 50:
            risk = "MEDIUM"
            risk_class = "metric-yellow"
        else:
            risk = "LOW"
            risk_class = "metric-green"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Current Risk</div>
                <div class="metric-value {risk_class}">
                    {risk}
                </div>
                <div class="metric-small">
                    Max probability: {max_probability:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    col1, col2 = st.columns([1.5, 1])

    with col1:

        st.markdown(
            '<div class="section-title">Live Attack Probability</div>',
            unsafe_allow_html=True
        )

        if not df.empty and "Probability" in df.columns:

            chart_df = df.copy()

            chart_df["Probability (%)"] = chart_df[
                "Probability"
            ].apply(probability_value)

            chart_df = chart_df.tail(100)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df["Probability (%)"],
                    mode="lines",
                    name="Attack Probability",
                    line=dict(width=2),
                    fill="tozeroy",
                )
            )

            fig.add_hline(
                y=50,
                line_dash="dash",
                annotation_text="Detection threshold"
            )

            fig.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=15, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa9bc"),
                xaxis=dict(
                    title="Recent flows",
                    gridcolor="#172334"
                ),
                yaxis=dict(
                    title="Probability (%)",
                    range=[0, 100],
                    gridcolor="#172334"
                ),
                legend=dict(
                    bgcolor="rgba(0,0,0,0)"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("Waiting for live traffic data...")

    with col2:

        st.markdown(
            '<div class="section-title">Traffic Classification</div>',
            unsafe_allow_html=True
        )

        if total_flows > 0:

            pie_df = pd.DataFrame({
                "Type": ["Benign", "Attack"],
                "Count": [benign_count, attack_count]
            })

            fig = px.pie(
                pie_df,
                names="Type",
                values="Count",
                hole=0.62,
            )

            fig.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=15, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa9bc"),
                legend=dict(
                    bgcolor="rgba(0,0,0,0)"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("No detection data available.")

    # --------------------------------------------------------
    # RECENT TRAFFIC
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Recent Network Activity</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        display_cols = [
            c for c in [
                "Timestamp",
                "Source_IP",
                "Destination_IP",
                "Source_Port",
                "Destination_Port",
                "Protocol",
                "Prediction",
                "Attack_Type",
                "Probability",
            ]
            if c in df.columns
        ]

        recent = df.tail(12)[display_cols].copy()

        if "Probability" in recent.columns:
            recent["Probability"] = recent[
                "Probability"
            ].apply(
                lambda x: f"{probability_value(x):.2f}%"
            )

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No traffic has been logged yet.")


# ============================================================
# LIVE DETECTION
# ============================================================

elif page == "Live Detection":

    st.markdown(
        '<div class="section-title">🔴 Live Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Real-time network flow inspection'
        '</div>',
        unsafe_allow_html=True
    )

    if df.empty:

        st.warning(
            "No live traffic detected yet. "
            "Start realtime_detection.py."
        )

    else:

        live = df.tail(50).copy()

        if "Probability" in live.columns:
            live["Risk"] = live[
                "Probability"
            ].apply(probability_value)

        display_cols = [
            c for c in [
                "Timestamp",
                "Source_IP",
                "Destination_IP",
                "Source_Port",
                "Destination_Port",
                "Protocol",
                "Packets",
                "Bytes",
                "Prediction",
                "Attack_Type",
                "Probability",
                "Risk",
            ]
            if c in live.columns
        ]

        st.dataframe(
            live[display_cols].sort_values(
                "Timestamp",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True,
            height=620,
        )


# ============================================================
# THREAT ALERTS
# ============================================================

elif page == "Threat Alerts":

    st.markdown(
        '<div class="section-title">🚨 Threat Alerts</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Detected malicious network activity'
        '</div>',
        unsafe_allow_html=True
    )

    if df.empty:

        st.success("No traffic available.")

    else:

        attacks = df[
            df["Prediction"].astype(str).str.upper()
            == "ATTACK"
        ].copy()

        if attacks.empty:

            st.success(
                "🟢 No active threats detected."
            )

        else:

            st.error(
                f"⚠️ {len(attacks):,} malicious flows detected"
            )

            for _, row in attacks.tail(20).iloc[::-1].iterrows():

                attack_type = row.get(
                    "Attack_Type",
                    "Unknown"
                )

                prob = probability_value(
                    row.get("Probability", 0)
                )

                source = row.get(
                    "Source_IP",
                    "Unknown"
                )

                destination = row.get(
                    "Destination_IP",
                    "Unknown"
                )

                timestamp = row.get(
                    "Timestamp",
                    ""
                )

                if prob >= 80:
                    alert_class = "alert-high"
                    severity = "HIGH"
                elif prob >= 50:
                    alert_class = "alert-medium"
                    severity = "MEDIUM"
                else:
                    alert_class = "alert-low"
                    severity = "LOW"

                st.markdown(
                    f"""
                    <div class="{alert_class}">
                        <div class="alert-title">
                            🔴 {severity} — {attack_type}
                        </div>
                        <div class="alert-text">
                            {timestamp}
                            &nbsp; | &nbsp;
                            {source} → {destination}
                            &nbsp; | &nbsp;
                            Confidence: {prob:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="section-title">📊 Security Analytics</div>',
        unsafe_allow_html=True
    )

    if df.empty:

        st.info("Not enough data for analytics.")

    else:

        col1, col2 = st.columns(2)

        # Attack type
        with col1:

            st.markdown(
                '<div class="section-title">'
                'Attack Distribution'
                '</div>',
                unsafe_allow_html=True
            )

            if "Attack_Type" in df.columns:

                attack_df = df[
                    df["Prediction"] == "ATTACK"
                ]

                if not attack_df.empty:

                    counts = (
                        attack_df[
                            "Attack_Type"
                        ]
                        .value_counts()
                        .reset_index()
                    )

                    counts.columns = [
                        "Attack Type",
                        "Count"
                    ]

                    fig = px.bar(
                        counts,
                        x="Count",
                        y="Attack Type",
                        orientation="h",
                    )

                    fig.update_layout(
                        height=400,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa9bc"),
                        margin=dict(
                            l=10,
                            r=10,
                            t=10,
                            b=10
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:
                    st.info(
                        "No attack classifications available."
                    )

        # Protocol
        with col2:

            st.markdown(
                '<div class="section-title">'
                'Protocol Distribution'
                '</div>',
                unsafe_allow_html=True
            )

            if "Protocol" in df.columns:

                protocol_df = (
                    df["Protocol"]
                    .value_counts()
                    .reset_index()
                )

                protocol_df.columns = [
                    "Protocol",
                    "Count"
                ]

                fig = px.pie(
                    protocol_df,
                    names="Protocol",
                    values="Count",
                    hole=0.55,
                )

                fig.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa9bc"),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # Probability
        st.markdown(
            '<div class="section-title">'
            'Probability Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        if "Probability" in df.columns:

            prob_df = df.copy()

            prob_df["Probability (%)"] = (
                prob_df["Probability"]
                .apply(probability_value)
            )

            fig = px.histogram(
                prob_df,
                x="Probability (%)",
                nbins=30,
            )

            fig.update_layout(
                height=330,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa9bc"),
                xaxis=dict(
                    gridcolor="#172334"
                ),
                yaxis=dict(
                    gridcolor="#172334"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# AI EXPLAINABILITY
# ============================================================

elif page == "AI Explainability":

    st.markdown(
        '<div class="section-title">'
        '🤖 AI Explainability'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Understanding which network features influence the model'
        '</div>',
        unsafe_allow_html=True
    )

    if os.path.exists(SHAP_FILE):

        try:

            shap_df = pd.read_csv(SHAP_FILE)

            shap_df.columns = [
                str(c).strip()
                for c in shap_df.columns
            ]

            st.markdown(
                '<div class="section-title">'
                'Top Influential Features'
                '</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                shap_df.head(20),
                use_container_width=True,
                hide_index=True,
            )

            numeric_cols = shap_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_cols:

                value_col = numeric_cols[-1]

                chart = shap_df.head(20).copy()

                feature_col = [
                    c for c in chart.columns
                    if "feature" in c.lower()
                ]

                if feature_col:

                    feature_col = feature_col[0]

                    fig = px.bar(
                        chart.sort_values(
                            value_col
                        ),
                        x=value_col,
                        y=feature_col,
                        orientation="h",
                    )

                    fig.update_layout(
                        height=550,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa9bc"),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        except Exception as e:

            st.warning(
                f"Could not load SHAP results: {e}"
            )

    else:

        st.info(
            "SHAP results are not available yet. "
            "Run shap_explainability.py first."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">'
        '🧠 Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Offline validation results of the trained detection models'
        '</div>',
        unsafe_allow_html=True
    )

    report = read_text(METRICS_FILE)

    if report != "Not available.":

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.code(
            report,
            language="text"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "Final validation report not found."
        )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Validation Visualizations'
        '</div>',
        unsafe_allow_html=True
    )

    plot_col1, plot_col2 = st.columns(2)

    confusion_path = os.path.join(
        RESULTS_DIR,
        "plots",
        "confusion_matrix_random_forest.png"
    )

    roc_path = os.path.join(
        RESULTS_DIR,
        "plots",
        "roc_curve_random_forest.png"
    )

    with plot_col1:

        if os.path.exists(confusion_path):

            st.image(
                confusion_path,
                caption="Random Forest Confusion Matrix",
                use_container_width=True
            )

        else:

            st.info(
                "Confusion matrix image unavailable."
            )

    with plot_col2:

        if os.path.exists(roc_path):

            st.image(
                roc_path,
                caption="Random Forest ROC Curve",
                use_container_width=True
            )

        else:

            st.info(
                "ROC curve image unavailable."
            )

    # Multiclass
    st.markdown(
        '<div class="section-title">'
        'Multiclass Attack Classification'
        '</div>',
        unsafe_allow_html=True
    )

    multiclass_report = read_text(
        ATTACK_METRICS_FILE
    )

    st.code(
        multiclass_report,
        language="text"
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

elif page == "System Status":

    st.markdown(
        '<div class="section-title">'
        '⚙️ System Status'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Health of the NIDS detection components'
        '</div>',
        unsafe_allow_html=True
    )

    components = {
        "Detection Log": (
            os.path.exists(LOG_FILE),
            LOG_FILE
        ),
        "Binary Model": (
            model_exists("best_model.pkl"),
            "models/best_model.pkl"
        ),
        "Binary Scaler": (
            os.path.exists(
                os.path.join(
                    BASE_DIR,
                    "data",
                    "processed",
                    "scaler.pkl"
                )
            ),
            "data/processed/scaler.pkl"
        ),
        "Multiclass Model": (
            model_exists("attack_classifier.pkl"),
            "models/attack_classifier.pkl"
        ),
        "Multiclass Scaler": (
            model_exists("multiclass_scaler.pkl"),
            "models/multiclass_scaler.pkl"
        ),
        "Label Encoder": (
            model_exists(
                "attack_label_encoder.pkl"
            ),
            "models/attack_label_encoder.pkl"
        ),
        "Feature Names": (
            model_exists("feature_names.txt"),
            "models/feature_names.txt"
        ),
    }

    for name, (exists, location) in components.items():

        c1, c2, c3 = st.columns(
            [4, 2, 5]
        )

        with c1:
            st.write(f"**{name}**")

        with c2:

            if exists:

                st.markdown(
                    '<span style="color:#4ade80;'
                    'font-weight:700;">● ONLINE</span>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<span style="color:#fb7185;'
                    'font-weight:700;">● MISSING</span>',
                    unsafe_allow_html=True
                )

        with c3:
            st.caption(location)

        st.divider()

    # Log information
    st.markdown(
        '<div class="section-title">'
        'Log Information'
        '</div>',
        unsafe_allow_html=True
    )

    if os.path.exists(LOG_FILE):

        size_kb = os.path.getsize(
            LOG_FILE
        ) / 1024

        modified = time.ctime(
            os.path.getmtime(LOG_FILE)
        )

        a, b, c = st.columns(3)

        a.metric(
            "Log Rows",
            f"{total_flows:,}"
        )

        b.metric(
            "Log Size",
            f"{size_kb:.1f} KB"
        )

        c.metric(
            "Last Updated",
            modified
        )

    else:

        st.warning(
            "Detection log has not been created yet."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ Sentinel NIDS &nbsp;•&nbsp;
        AI-Powered Network Intrusion Detection System
        <br>
        Random Forest • Scapy • SHAP • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# AUTO REFRESH
# ============================================================

if auto_refresh:
    time.sleep(refresh_seconds)
    st.cache_data.clear()
    st.rerun()