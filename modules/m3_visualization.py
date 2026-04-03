import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from utils.database import save_record
from utils.storage import save_file

# ----------- Publication-quality Plotly theme (Nature/Science style) ----------
NATURE_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    margin=dict(l=80, r=40, t=60, b=80),
    legend=dict(
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        font=dict(family="Times New Roman, serif", size=13),
    ),
)

def nature_axis(title: str, log_scale: bool = False):
    return dict(
        title=dict(text=title, font=dict(family="Times New Roman, serif", size=15)),
        showgrid=False,
        zeroline=False,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        tickwidth=2,
        ticklen=6,
        mirror=True,
        showline=True,
        type="log" if log_scale else "linear",
    )

COLOR_PALETTE = [
    "#000000", "#1f77b4", "#d62728", "#2ca02c",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

ANALYSIS_TYPES = {
    "DSC - 差示扫描量热": {
        "x_candidates": ["temperature", "temp", "t", "温度"],
        "y_candidates": ["heat_flow", "heatflow", "dsc", "热流", "热流量"],
        "x_label": "Temperature (°C)",
        "y_label": "Heat Flow (mW/mg)",
    },
    "TGA - 热重分析": {
        "x_candidates": ["temperature", "temp", "t", "温度"],
        "y_candidates": ["mass", "weight", "tga", "质量", "重量", "mass_%"],
        "x_label": "Temperature (°C)",
        "y_label": "Mass (%)",
    },
    "XRD - X射线衍射": {
        "x_candidates": ["2theta", "angle", "2_theta", "theta", "角度"],
        "y_candidates": ["intensity", "counts", "强度"],
        "x_label": "2θ (°)",
        "y_label": "Intensity (a.u.)",
    },
    "应力-应变曲线": {
        "x_candidates": ["strain", "应变", "displacement", "extension"],
        "y_candidates": ["stress", "应力", "force", "load"],
        "x_label": "Strain (%)",
        "y_label": "Stress (MPa)",
    },
    "电化学 (CV/EIS)": {
        "x_candidates": ["voltage", "potential", "v", "电压", "电位"],
        "y_candidates": ["current", "i", "电流"],
        "x_label": "Voltage (V)",
        "y_label": "Current (mA)",
    },
    "自定义": {
        "x_candidates": [],
        "y_candidates": [],
        "x_label": "X 轴",
        "y_label": "Y 轴",
    },
}

def auto_detect_columns(df: pd.DataFrame, candidates: list) -> str:
    """Auto-detect most likely column from candidate keywords."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # Fallback: first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols[0] if numeric_cols else df.columns[0]

def build_figure(df: pd.DataFrame, x_col: str, y_cols: list, analysis_type: str,
                 x_label: str, y_label: str, title: str,
                 x_range=None, y_range=None, line_width: float = 2.0,
                 show_markers: bool = False) -> go.Figure:

    fig = go.Figure()
    mode = "lines+markers" if show_markers else "lines"

    for i, y_col in enumerate(y_cols):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        trace_kwargs = dict(
            x=df[x_col],
            y=df[y_col],
            name=y_col,
            mode=mode,
            line=dict(color=color, width=line_width),
        )
        if show_markers:
            trace_kwargs["marker"] = dict(size=5, color=color, symbol="circle")
        fig.add_trace(go.Scatter(**trace_kwargs))

    layout_update = dict(
        **NATURE_LAYOUT,
        title=dict(
            text=title,
            font=dict(family="Times New Roman, serif", size=16, color="black"),
            x=0.5, xanchor="center",
        ),
        xaxis=nature_axis(x_label),
        yaxis=nature_axis(y_label),
        width=800, height=520,
    )

    if x_range and x_range[0] < x_range[1]:
        layout_update["xaxis"]["range"] = list(x_range)
    if y_range and y_range[0] < y_range[1]:
        layout_update["yaxis"]["range"] = list(y_range)

    fig.update_layout(**layout_update)
    return fig

def render():
    st.markdown('<div class="module-header">M3 - 实验数据可视化</div>', unsafe_allow_html=True)
    st.caption("科研级绘图 (Nature/Science 风格)，支持 DSC、TGA、XRD、应力-应变等多种实验数据")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**数据上传**")
        uploaded = st.file_uploader(
            "上传数据文件", type=["csv", "xlsx", "xls", "txt"],
            help="支持 CSV / Excel / 制表符分隔 TXT"
        )
        if uploaded:
            st.success(f"已加载: {uploaded.name}")

    if uploaded is None:
        st.info("请在侧边栏上传实验数据文件 (CSV 或 Excel)")
        with st.expander("示例数据格式"):
            sample = pd.DataFrame({
                "Temperature": np.linspace(25, 600, 100),
                "Heat_Flow": -np.exp(-((np.linspace(25, 600, 100) - 200) / 50) ** 2) * 3.5,
                "Mass_%": 100 - np.cumsum(np.random.exponential(0.05, 100)).clip(0, 30),
            })
            st.dataframe(sample.head(10))
            st.download_button(
                "下载示例 CSV",
                sample.to_csv(index=False).encode(),
                "sample_dsc_tga.csv",
                "text/csv"
            )
        return

    # Load data
    try:
        if uploaded.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        elif uploaded.name.endswith(".txt"):
            df = pd.read_csv(uploaded, sep=None, engine="python")
        else:
            df = pd.read_csv(uploaded)
        df = df.dropna(how="all").reset_index(drop=True)
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("文件中数值列不足 2 列，无法绘图")
        return

    st.markdown("**数据预览**")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"共 {len(df)} 行 x {len(df.columns)} 列")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        analysis_type = st.selectbox("分析类型", list(ANALYSIS_TYPES.keys()))
        atype = ANALYSIS_TYPES[analysis_type]

    # Auto-detect columns
    default_x = auto_detect_columns(df, atype["x_candidates"])
    default_y_idx = [i for i, c in enumerate(numeric_cols) if c != default_x][:2]

    with col2:
        x_col = st.selectbox("X 轴变量", numeric_cols,
                              index=numeric_cols.index(default_x) if default_x in numeric_cols else 0)

    with col3:
        y_cols = st.multiselect("Y 轴变量 (可多选)", numeric_cols,
                                default=[numeric_cols[i] for i in default_y_idx if i < len(numeric_cols)])

    if not y_cols:
        st.warning("请至少选择一个 Y 轴变量")
        return

    # Label and range config
    col4, col5 = st.columns(2)
    with col4:
        x_label = st.text_input("X 轴标签", value=atype["x_label"])
        y_label = st.text_input("Y 轴标签", value=atype["y_label"])
        chart_title = st.text_input("图表标题", value=analysis_type)

    with col5:
        x_min_v = float(df[x_col].min())
        x_max_v = float(df[x_col].max())
        y_all = pd.concat([df[c] for c in y_cols])
        y_min_v = float(y_all.min())
        y_max_v = float(y_all.max())

        x_range = st.slider("X 轴范围", x_min_v, x_max_v, (x_min_v, x_max_v), key="xrange")
        y_range = st.slider("Y 轴范围", y_min_v - abs(y_min_v) * 0.1,
                            y_max_v + abs(y_max_v) * 0.1,
                            (y_min_v, y_max_v), key="yrange")

    col6, col7, col8 = st.columns(3)
    with col6:
        line_width = st.slider("线条宽度", 0.5, 5.0, 2.0, 0.5)
    with col7:
        show_markers = st.checkbox("显示数据点", value=False)
    with col8:
        show_derivative = st.checkbox("显示一阶导数", value=False)

    # Build and display plot
    fig = build_figure(
        df=df, x_col=x_col, y_cols=y_cols,
        analysis_type=analysis_type,
        x_label=x_label, y_label=y_label, title=chart_title,
        x_range=x_range, y_range=y_range,
        line_width=line_width, show_markers=show_markers,
    )

    # Overlay derivative if requested
    if show_derivative and len(y_cols) >= 1:
        for y_col in y_cols[:1]:
            dy = np.gradient(df[y_col].values, df[x_col].values)
            fig.add_trace(go.Scatter(
                x=df[x_col], y=dy,
                name=f"d({y_col})/d({x_col})",
                mode="lines",
                line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                yaxis="y2",
            ))
        fig.update_layout(
            yaxis2=dict(
                **nature_axis("导数"),
                overlaying="y", side="right",
            )
        )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    with st.expander("数据统计"):
        st.dataframe(df[[x_col] + y_cols].describe().round(4), use_container_width=True)

    # Save buttons
    st.markdown("---")
    save_col1, save_col2, save_col3 = st.columns(3)

    with save_col1:
        buf = io.BytesIO()
        fig.write_image(buf, format="svg")
        st.download_button("下载 SVG", buf.getvalue(), f"{chart_title}.svg", "image/svg+xml")

    with save_col2:
        buf2 = io.BytesIO()
        fig.write_image(buf2, format="pdf")
        st.download_button("下载 PDF", buf2.getvalue(), f"{chart_title}.pdf", "application/pdf")

    with save_col3:
        if st.button("保存至数据库"):
            buf3 = io.BytesIO()
            try:
                fig.write_image(buf3, format="png")
                path = save_file(buf3.getvalue(), "plots", f"{chart_title}.png")
                save_record(
                    module="M3",
                    name=chart_title,
                    file_path=str(path),
                    metadata={"x_col": x_col, "y_cols": y_cols, "analysis_type": analysis_type}
                )
                st.success("已保存")
            except Exception as e:
                st.error(f"保存失败: {e}")
