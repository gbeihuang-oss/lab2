import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils.database import save_record
from utils.storage import save_json

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

PERFORMANCE_CATEGORIES = {
    "力学性能": ["抗拉强度 (MPa)", "屈服强度 (MPa)", "断裂延伸率 (%)", "硬度 (HV)", "弯曲强度 (MPa)", "断裂韧性 (MPa·m^0.5)"],
    "热学性能": ["热导率 (W/m·K)", "热膨胀系数 (1e-6/K)", "比热容 (J/g·K)", "玻璃化转变温度 (°C)", "熔点 (°C)"],
    "电学性能": ["电导率 (S/m)", "电阻率 (Ω·m)", "介电常数", "介电损耗 (tan δ)", "击穿场强 (kV/mm)"],
    "光学性能": ["折射率", "透过率 (%)", "带隙 (eV)", "反射率 (%)"],
    "其他": ["密度 (g/cm3)", "孔隙率 (%)", "比表面积 (m2/g)", "粒径 D50 (nm)"],
}

MODELS_CONFIG = {
    "梯度提升回归 (GBR)": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "随机森林 (RF)": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    "极端随机树 (ET)": ExtraTreesRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    "神经网络 (MLP)": MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True),
}

def render():
    st.markdown('<div class="module-header">M6 - 性能预测</div>', unsafe_allow_html=True)
    st.caption("上传材料成分/工艺数据，自动训练多模型并预测材料力学、电学或热学性能")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**数据上传**")
        uploaded = st.file_uploader("上传性能数据", type=["csv", "xlsx", "xls"])
        if uploaded:
            st.success(f"已加载: {uploaded.name}")

    if uploaded is None:
        st.info("请在侧边栏上传材料性能数据集（CSV / Excel）")
        with st.expander("示例数据与格式说明"):
            np.random.seed(0)
            n = 120
            sample_df = pd.DataFrame({
                "Cu_%": np.random.uniform(0, 10, n),
                "Ni_%": np.random.uniform(0, 5, n),
                "Cr_%": np.random.uniform(10, 25, n),
                "Mo_%": np.random.uniform(0, 3, n),
                "固溶温度_C": np.random.uniform(1000, 1200, n),
                "时效时间_h": np.random.uniform(0, 12, n),
            })
            sample_df["抗拉强度_MPa"] = (
                500 + 20 * sample_df["Cr_%"] + 15 * sample_df["Ni_%"]
                + 30 * sample_df["Mo_%"] - 0.1 * sample_df["固溶温度_C"]
                + 5 * sample_df["时效时间_h"] + np.random.normal(0, 20, n)
            ).clip(400, 1200)
            st.dataframe(sample_df.head(8))
            st.download_button("下载示例", sample_df.to_csv(index=False).encode(), "sample_performance.csv")
        return

    # Load data
    try:
        df = pd.read_excel(uploaded) if uploaded.name.endswith((".xlsx", ".xls")) else pd.read_csv(uploaded)
        df = df.dropna(how="all").reset_index(drop=True)
    except Exception as e:
        st.error(f"读取失败: {e}")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("**数据预览**")
    st.dataframe(df.head(8), use_container_width=True)
    st.caption(f"共 {len(df)} 行 x {len(df.columns)} 列")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        feature_cols = st.multiselect("输入特征 (X)", numeric_cols, default=numeric_cols[:-1])
    with col2:
        target_cols = st.multiselect(
            "预测目标 (Y，可多选)",
            [c for c in numeric_cols if c not in feature_cols],
            default=[numeric_cols[-1]] if numeric_cols else []
        )

    if not feature_cols or not target_cols:
        st.warning("请选择特征列和目标列")
        return

    col3, col4 = st.columns(2)
    with col3:
        selected_models = st.multiselect(
            "选择预测模型",
            list(MODELS_CONFIG.keys()),
            default=["梯度提升回归 (GBR)", "随机森林 (RF)"]
        )
    with col4:
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.number_input("交叉验证折数", 2, 10, 5)

    if st.button("训练并评估所有模型", type="primary"):
        results = {}
        X = df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, idx_train, idx_test = train_test_split(
            X_scaled, np.arange(len(X)), test_size=test_size, random_state=42
        )

        for target in target_cols:
            y = df[target].values
            y_train = y[idx_train]
            y_test = y[idx_test]
            results[target] = {}

            for model_name in selected_models:
                with st.spinner(f"训练 {model_name} -> {target}..."):
                    m = MODELS_CONFIG[model_name]
                    cv_r2 = cross_val_score(m, X_scaled, y, cv=int(cv_folds), scoring="r2").mean()
                    m.fit(X_train, y_train)
                    y_pred = m.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    results[target][model_name] = {
                        "model": m,
                        "cv_r2": cv_r2,
                        "test_r2": r2,
                        "mae": mae,
                        "rmse": rmse,
                        "y_test": y_test,
                        "y_pred": y_pred,
                    }

        st.session_state["m6_results"] = results
        st.session_state["m6_scaler"] = scaler
        st.session_state["m6_feature_cols"] = feature_cols
        st.session_state["m6_target_cols"] = target_cols
        st.session_state["m6_df"] = df

    if "m6_results" not in st.session_state:
        return

    results = st.session_state["m6_results"]
    f_cols = st.session_state["m6_feature_cols"]
    t_cols = st.session_state["m6_target_cols"]

    st.markdown("---")
    st.markdown("**模型评估结果**")

    # Metrics table
    rows = []
    for target, models in results.items():
        for model_name, metrics in models.items():
            rows.append({
                "目标性能": target,
                "模型": model_name,
                "CV R²": round(metrics["cv_r2"], 4),
                "测试集 R²": round(metrics["test_r2"], 4),
                "MAE": round(metrics["mae"], 4),
                "RMSE": round(metrics["rmse"], 4),
            })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(
        metrics_df.style.background_gradient(subset=["测试集 R²"], cmap="Blues"),
        use_container_width=True
    )

    # Predicted vs Actual plots
    for target in t_cols:
        if target not in results:
            continue
        st.markdown(f"**{target} - 预测值 vs 实际值**")
        fig = go.Figure()
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
        for i, (model_name, metrics) in enumerate(results[target].items()):
            fig.add_trace(go.Scatter(
                x=metrics["y_test"], y=metrics["y_pred"],
                mode="markers", name=model_name,
                marker=dict(size=7, color=colors[i % len(colors)], opacity=0.7)
            ))
        # Perfect prediction line
        all_y = np.concatenate([metrics["y_test"] for metrics in results[target].values()])
        mn, mx = all_y.min(), all_y.max()
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            name="完美预测", line=dict(color="black", width=1.5, dash="dash")
        ))
        fig.update_layout(
            **NATURE_LAYOUT,
            xaxis=nature_axis(f"实际值 ({target})"),
            yaxis=nature_axis(f"预测值 ({target})"),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    # New data prediction
    st.markdown("---")
    st.markdown("**新样本性能预测**")
    st.caption("输入新材料的成分/工艺参数，预测其性能")

    new_vals = {}
    inp_cols = st.columns(min(len(f_cols), 4))
    for i, fc in enumerate(f_cols):
        col_min = float(df[fc].min())
        col_max = float(df[fc].max())
        with inp_cols[i % 4]:
            new_vals[fc] = st.number_input(
                fc,
                min_value=float(col_min * 0.5),
                max_value=float(col_max * 1.5),
                value=float(df[fc].mean()),
                key=f"pred_input_{fc}",
                format="%.4f"
            )

    if st.button("预测新样本性能"):
        scaler = st.session_state["m6_scaler"]
        X_new = np.array([[new_vals[fc] for fc in f_cols]])
        X_new_scaled = scaler.transform(X_new)

        pred_rows = []
        for target in t_cols:
            if target not in results:
                continue
            best_model_name = max(
                results[target].keys(),
                key=lambda k: results[target][k]["test_r2"]
            )
            best_model = results[target][best_model_name]["model"]
            pred_val = best_model.predict(X_new_scaled)[0]
            pred_rows.append({
                "目标性能": target,
                "预测值": round(pred_val, 4),
                "使用模型": best_model_name,
                "模型 R²": round(results[target][best_model_name]["test_r2"], 4),
            })

        pred_df = pd.DataFrame(pred_rows)
        st.markdown("**预测结果**")
        st.dataframe(pred_df, use_container_width=True)

        # Save
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button("下载预测结果", pred_df.to_csv(index=False).encode(), "prediction.csv")
        with ec2:
            if st.button("保存至数据库"):
                path = save_json(
                    {"input": new_vals, "predictions": pred_df.to_dict(orient="records")},
                    "predictions",
                    "performance_prediction.json"
                )
                save_record(
                    module="M6",
                    name=f"性能预测 - {', '.join(t_cols)}",
                    file_path=str(path),
                    metadata={"features": f_cols, "targets": t_cols}
                )
                st.success("已保存")
