import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from utils.database import save_record
from utils.storage import save_file, save_json

NATURE_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Times New Roman, serif", size=13),
    margin=dict(l=70, r=40, t=50, b=70),
)

def nature_axis(title: str):
    return dict(
        title=dict(text=title, font=dict(family="Times New Roman, serif", size=14)),
        showgrid=False, zeroline=False, linecolor="black",
        linewidth=2, ticks="outside", tickwidth=2, ticklen=5, mirror=True, showline=True,
    )

def render():
    st.markdown('<div class="module-header">M5 - 配方优化</div>', unsafe_allow_html=True)
    st.caption("基于梯度提升机 (GBR) + 交叉验证的高通量配方筛选与优化")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**数据上传**")
        uploaded = st.file_uploader("上传配方数据", type=["csv", "xlsx", "xls"])
        if uploaded:
            st.success(f"已加载: {uploaded.name}")

    if uploaded is None:
        st.info("请在侧边栏上传配方数据文件（CSV / Excel）")
        with st.expander("示例数据格式"):
            np.random.seed(42)
            n = 80
            sample = pd.DataFrame({
                "TiO2_%": np.random.uniform(5, 40, n),
                "SiO2_%": np.random.uniform(10, 50, n),
                "Al2O3_%": np.random.uniform(5, 20, n),
                "烧结温度_C": np.random.uniform(900, 1400, n),
                "保温时间_h": np.random.uniform(1, 6, n),
            })
            sample["致密度_%"] = (
                85
                + 0.1 * sample["TiO2_%"]
                - 0.05 * sample["SiO2_%"]
                + 0.008 * sample["烧结温度_C"]
                + 0.5 * sample["保温时间_h"]
                + np.random.normal(0, 1, n)
            ).clip(70, 100)
            st.dataframe(sample.head(8))
            st.download_button(
                "下载示例 CSV",
                sample.to_csv(index=False).encode(),
                "sample_formula.csv"
            )
        return

    # Load data
    try:
        if uploaded.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
        df = df.dropna(how="all").reset_index(drop=True)
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("数值列不足 2 列")
        return

    st.markdown("**数据预览**")
    st.dataframe(df.head(8), use_container_width=True)
    st.caption(f"共 {len(df)} 行 x {len(df.columns)} 列")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        feature_cols = st.multiselect(
            "选择输入特征 (X)",
            numeric_cols,
            default=numeric_cols[:-1],
        )
    with col2:
        target_col = st.selectbox(
            "选择目标变量 (Y)",
            [c for c in numeric_cols if c not in feature_cols],
        )

    if not feature_cols or not target_col:
        st.warning("请选择至少 1 个特征列和 1 个目标列")
        return

    # Model config
    with st.expander("模型超参数设置", expanded=False):
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            n_estimators = st.number_input("树的数量", 50, 500, 200, 50)
        with mc2:
            learning_rate = st.number_input("学习率", 0.01, 0.5, 0.05, 0.01)
        with mc3:
            max_depth = st.number_input("最大深度", 2, 8, 4, 1)
        with mc4:
            cv_folds = st.number_input("交叉验证折数", 2, 10, 3, 1)

    X = df[feature_cols].values
    y = df[target_col].values

    if st.button("训练模型", type="primary"):
        with st.spinner("正在训练 GradientBoosting 模型并执行交叉验证..."):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = GradientBoostingRegressor(
                n_estimators=int(n_estimators),
                learning_rate=learning_rate,
                max_depth=int(max_depth),
                random_state=42,
            )
            cv_scores = cross_val_score(model, X_scaled, y, cv=int(cv_folds), scoring="r2")
            model.fit(X_scaled, y)

        st.session_state["m5_model"] = model
        st.session_state["m5_scaler"] = scaler
        st.session_state["m5_feature_cols"] = feature_cols
        st.session_state["m5_target_col"] = target_col
        st.session_state["m5_df"] = df

        r2_mean = cv_scores.mean()
        r2_std = cv_scores.std()

        # Metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("平均 R² (CV)", f"{r2_mean:.4f}")
        mc2.metric("R² 标准差", f"{r2_std:.4f}")
        mc3.metric("训练集数量", len(y))

        if r2_mean < 0.5:
            st.warning("模型 R² 较低，建议检查数据质量或增加训练样本")
        elif r2_mean > 0.9:
            st.success("模型拟合效果良好")
        else:
            st.info("模型拟合效果尚可")

        # Feature importance
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"特征": feature_cols, "重要性": importances})
        fi_df = fi_df.sort_values("重要性", ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["重要性"], y=fi_df["特征"],
            orientation="h",
            marker=dict(color="#1f77b4"),
        ))
        fig_fi.update_layout(
            **NATURE_LAYOUT,
            title=dict(text="特征重要性", font=dict(family="Times New Roman, serif", size=15)),
            xaxis=nature_axis("重要性"),
            yaxis=dict(title="", showgrid=False, linecolor="black", linewidth=2),
            height=300 + len(feature_cols) * 20,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Screening section - only if model is trained
    if "m5_model" not in st.session_state:
        return

    model = st.session_state["m5_model"]
    scaler = st.session_state["m5_scaler"]
    f_cols = st.session_state["m5_feature_cols"]
    t_col = st.session_state["m5_target_col"]
    train_df = st.session_state["m5_df"]

    st.markdown("---")
    st.markdown("**高通量筛选配置**")
    st.caption("根据各特征的实际取值范围设定搜索空间，系统将随机采样 5000 组组合进行预测")

    slider_vals = {}
    cols = st.columns(min(len(f_cols), 4))
    for i, fc in enumerate(f_cols):
        col_min = float(train_df[fc].min())
        col_max = float(train_df[fc].max())
        step = (col_max - col_min) / 100.0
        with cols[i % 4]:
            slider_vals[fc] = st.slider(
                fc, col_min, col_max, (col_min, col_max),
                step=step if step > 0 else 0.01,
                key=f"slider_{fc}"
            )

    n_samples = st.select_slider(
        "随机采样数量",
        options=[1000, 2000, 5000, 10000, 20000],
        value=5000
    )
    top_n = st.number_input("返回 Top N 组合", 5, 50, 20)
    optimize_direction = st.radio("优化方向", ["最大化", "最小化"], horizontal=True)

    if st.button("开始高通量筛选", type="primary"):
        with st.spinner(f"正在随机采样 {n_samples} 组配方并预测..."):
            np.random.seed(0)
            sampled = {}
            for fc in f_cols:
                lo, hi = slider_vals[fc]
                sampled[fc] = np.random.uniform(lo, hi, int(n_samples))
            sample_df = pd.DataFrame(sampled)
            X_sample = scaler.transform(sample_df.values)
            preds = model.predict(X_sample)
            sample_df[t_col + "_预测"] = preds

            if optimize_direction == "最大化":
                top_df = sample_df.nlargest(int(top_n), t_col + "_预测").reset_index(drop=True)
            else:
                top_df = sample_df.nsmallest(int(top_n), t_col + "_预测").reset_index(drop=True)
            top_df.index += 1

        st.success(f"筛选完成，共评估 {n_samples} 组配方，返回 Top {top_n}")
        st.session_state["m5_top_df"] = top_df
        st.session_state["m5_all_preds"] = preds
        st.session_state["m5_all_sample"] = sample_df

    if "m5_top_df" not in st.session_state:
        return

    top_df = st.session_state["m5_top_df"]
    all_preds = st.session_state["m5_all_preds"]
    all_sample = st.session_state["m5_all_sample"]

    st.markdown(f"**Top {len(top_df)} 优化配方**")
    st.dataframe(top_df.round(4), use_container_width=True)

    # Distribution plot
    pred_col = t_col + "_预测"
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=all_preds, nbinsx=50, name="全部预测",
        marker_color="#aec6e8", opacity=0.7
    ))
    fig_dist.add_trace(go.Scatter(
        x=top_df[pred_col],
        y=[2] * len(top_df),
        mode="markers",
        name=f"Top {len(top_df)}",
        marker=dict(color="#d62728", size=10, symbol="star")
    ))
    fig_dist.update_layout(
        **NATURE_LAYOUT,
        title=dict(text=f"{pred_col} 预测分布", font=dict(family="Times New Roman, serif", size=15)),
        xaxis=nature_axis(f"{t_col} (预测值)"),
        yaxis=nature_axis("频数"),
        height=380,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Scatter matrix for top results (if >=2 features)
    if len(f_cols) >= 2:
        st.markdown("**Top 配方特征散点图**")
        fig_scatter = px.scatter(
            top_df, x=f_cols[0], y=f_cols[1],
            color=pred_col,
            color_continuous_scale="Blues",
            title=f"Top {len(top_df)} 配方分布",
        )
        fig_scatter.update_layout(**NATURE_LAYOUT, height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Save and export
    st.markdown("---")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.download_button(
            "下载 Top 配方 CSV",
            top_df.to_csv().encode(),
            "top_formulas.csv"
        )
    with ec2:
        if st.button("保存至数据库"):
            path = save_json(
                {
                    "top_df": top_df.to_dict(orient="records"),
                    "features": f_cols,
                    "target": t_col,
                    "n_samples": int(n_samples),
                },
                "recipes",
                "top_formulas.json"
            )
            save_record(
                module="M5",
                name=f"配方优化 - {t_col}",
                file_path=str(path),
                metadata={"target": t_col, "features": f_cols, "top_n": len(top_df)}
            )
            st.success("已保存")

    with ec3:
        if st.button("发送至 M1 研发助手"):
            stats = top_df.describe().round(4).to_string()
            top_str = top_df.head(5).round(4).to_string()
            prompt = f"""以下是材料配方优化结果，请从材料科学原理角度解释这些最优配方的组成规律：

【目标性能】: {t_col} ({optimize_direction})

【Top 5 最优配方】:
{top_str}

【统计特征（Top {len(top_df)}）】:
{stats}

【问题】:
1. 为什么这些特征组合能达到较优的 {t_col}？请从材料科学机理（相图、动力学、微观结构等）角度解释。
2. 哪个特征对性能影响最大？有何文献支撑？
3. 这些配方在实际实验中需要注意哪些工艺细节？
请引用相关学术文献支撑分析。"""

            st.session_state["m5_prompt"] = prompt
            st.success("已发送至 M1 研发助手，请切换到 M1 模块查看")
