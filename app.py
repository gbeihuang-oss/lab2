import streamlit as st
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import APP_TITLE, APP_SUBTITLE, MODULES
from utils.database import init_database
from utils.storage import init_storage

# ─── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Lab - 虚拟实验室",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Initialize backend ──────────────────────────────────────────────────────
init_database()
init_storage()

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography & Base ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}
h1, h2, h3 {
    font-family: 'Times New Roman', 'Noto Serif SC', serif;
}

/* ── App background ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    min-width: 220px;
}
[data-testid="stSidebar"] * {
    color: #e8eaf6 !important;
}
[data-testid="stSidebar"] .stRadio > label {
    color: #b0bec5 !important;
    font-size: 13px;
}
[data-testid="stSidebar"] hr {
    border-color: #37474f;
}

/* ── Module header ── */
.module-header {
    font-family: 'Times New Roman', serif;
    font-size: 22px;
    font-weight: bold;
    color: #0d1b2a;
    padding: 8px 0 10px 0;
    border-bottom: 3px solid #1565c0;
    margin-bottom: 18px;
    letter-spacing: 0.5px;
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    text-align: center;
    padding: 16px 8px 8px 8px;
}
.sidebar-logo-title {
    font-family: 'Times New Roman', serif;
    font-size: 26px;
    font-weight: bold;
    color: #e3f2fd;
    letter-spacing: 3px;
}
.sidebar-logo-sub {
    font-size: 11px;
    color: #90a4ae;
    margin-top: 2px;
}

/* ── Nav item highlight ── */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    border-radius: 4px;
    padding: 4px 10px;
    transition: background 0.2s;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #f0f4f8;
    border: 1px solid #dce1e7;
    border-radius: 8px;
    padding: 8px 12px;
}

/* ── Step cards in M7 ── */
.step-card {
    border: 1.5px solid #dce1e7;
    border-radius: 8px;
    padding: 14px;
    margin: 6px 0;
    background: #fafbfc;
    border-left: 5px solid #1565c0;
    transition: box-shadow 0.2s;
}
.step-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.step-card-active {
    border-left: 5px solid #2e7d32;
    background: #f1f8f1;
}
.step-title {
    font-family: 'Times New Roman', serif;
    font-size: 15px;
    font-weight: bold;
    color: #0d1b2a;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 6px;
    overflow: hidden;
}

/* ── Chat ── */
.stChatMessage {
    border-radius: 8px;
}

#/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
#header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-title">LAB</div>
        <div class="sidebar-logo-sub">虚拟实验室平台 v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<span style="font-size:12px; color:#90a4ae;">功能模块导航</span>',
        unsafe_allow_html=True
    )

    selected_module = st.radio(
        "",
        list(MODULES.keys()),
        index=0,
        key="nav_radio"
    )

    st.markdown("---")
    st.markdown(
        '<span style="font-size:11px; color:#607d8b;">'
        'Powered by Qwen3-32B (Groq)<br>'
        'Materials Project API<br>'
        'Scikit-learn · Plotly · RDKit'
        '</span>',
        unsafe_allow_html=True
    )

# ─── Module Routing ──────────────────────────────────────────────────────────
module_key = MODULES[selected_module]

if module_key == "m1_assistant":
    from modules.m1_assistant import render
elif module_key == "m2_molecular":
    from modules.m2_molecular import render
elif module_key == "m3_visualization":
    from modules.m3_visualization import render
elif module_key == "m4_image_analysis":
    from modules.m4_image_analysis import render
elif module_key == "m5_optimization":
    from modules.m5_optimization import render
elif module_key == "m6_prediction":
    from modules.m6_prediction import render
elif module_key == "m7_workflow":
    from modules.m7_workflow import render
elif module_key == "m8_database":
    from modules.m8_database import render
else:
    def render():
        st.error(f"模块 {module_key} 未找到")

render()
