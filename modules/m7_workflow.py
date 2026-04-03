import streamlit as st
import json
from datetime import datetime
from openai import OpenAI
from utils.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
from utils.database import save_record, save_workflow
from utils.storage import save_text

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

WORKFLOW_STEPS = [
    {
        "id": "weighing",
        "title": "原料称量",
        "icon_text": "[称量]",
        "description": "按配方比例精确称取各组分原料",
        "params": ["目标质量 (g)", "允许误差 (%)", "称量顺序", "容器类型"],
        "tips": "使用分析天平（精度 0.1 mg），称量前调零，批次间保持一致性",
        "color": "#1f77b4",
    },
    {
        "id": "mixing",
        "title": "混合搅拌",
        "icon_text": "[混合]",
        "description": "将各组分均匀混合，确保成分均一性",
        "params": ["搅拌转速 (rpm)", "混合时间 (min)", "温度控制 (°C)", "搅拌介质"],
        "tips": "湿法混合优先，干法混合需防止粉尘，记录混合均匀性评估结果",
        "color": "#2ca02c",
    },
    {
        "id": "forming",
        "title": "成型压制",
        "icon_text": "[成型]",
        "description": "将混合料压制成所需形状的生坯",
        "params": ["成型压力 (MPa)", "保压时间 (s)", "模具尺寸 (mm)", "润滑剂用量 (%)"],
        "tips": "控制加压速率，避免分层，检查生坯密度和外观",
        "color": "#ff7f0e",
    },
    {
        "id": "sintering",
        "title": "烧结热处理",
        "icon_text": "[烧结]",
        "description": "在受控气氛下进行高温烧结/热处理",
        "params": ["烧结温度 (°C)", "升温速率 (°C/min)", "保温时间 (h)", "气氛环境", "降温速率 (°C/min)"],
        "tips": "预先去除黏结剂（排胶），监控烧结收缩率，气氛控制精确（氧分压 ppm 级）",
        "color": "#d62728",
    },
    {
        "id": "post_processing",
        "title": "后处理加工",
        "icon_text": "[后处理]",
        "description": "机械加工、表面处理或热处理改性",
        "params": ["加工方式", "表面粗糙度要求 (Ra)", "热处理温度 (°C)", "冷却介质"],
        "tips": "避免引入残余应力，表面处理前检查尺寸精度",
        "color": "#9467bd",
    },
    {
        "id": "testing",
        "title": "性能测试",
        "icon_text": "[测试]",
        "description": "按标准方法测试材料力学、电学、热学等性能",
        "params": ["测试标准 (ISO/ASTM)", "测试温度 (°C)", "样品数量 (n)", "测试设备"],
        "tips": "每批次至少 3 个样品，记录环境温湿度，测试前设备标定，结果统计给出平均值 ± 标准差",
        "color": "#8c564b",
    },
]

def generate_protocol_from_formula(formula_data: dict) -> str:
    """Use LLM to generate detailed experimental protocol from M5 optimization results."""
    prompt = f"""基于以下材料配方优化结果，生成一份详细的实验操作手册：

【优化配方数据】:
{json.dumps(formula_data, ensure_ascii=False, indent=2)}

请生成包含以下内容的实验操作指导手册：
1. 实验目的与背景
2. 所需原料清单（纯度要求、供应商建议）
3. 仪器设备清单（型号建议、关键技术参数）
4. 详细实验步骤（每步骤包含具体参数、操作要点、注意事项）
5. 关键过程控制参数及检测方法
6. 质量评价标准（合格/不合格判定）
7. 安全注意事项与废弃物处理
8. 预期结果与可能遇到的问题及解决方案

格式要求：
- 步骤编号清晰
- 关键参数用具体数值表达（温度、时间、压力等）
- 引用相关标准（ISO、ASTM、GB等）
请以专业实验室操作手册的格式输出，使用中文。"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "你是一位材料实验室资深工程师，擅长设计严谨可重复的实验方案。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成失败: {e}"

def render():
    st.markdown('<div class="module-header">M7 - 虚拟实验室工作流</div>', unsafe_allow_html=True)
    st.caption("可视化实验流程卡片 | 连接 M5 配方优化结果自动生成实验操作手册")

    # CSS for step cards
    st.markdown("""
    <style>
    .step-card {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        background: #fafafa;
        border-left: 6px solid #1f77b4;
    }
    .step-card-active {
        border-left: 6px solid #2ca02c;
        background: #f0faf0;
    }
    .step-title {
        font-family: 'Times New Roman', serif;
        font-size: 16px;
        font-weight: bold;
        color: #1a1a2e;
    }
    </style>
    """, unsafe_allow_html=True)

    # Workflow visualization
    st.markdown("**实验工作流程**")
    st.caption("点击各步骤卡片查看详细参数设置")

    if "workflow_active_steps" not in st.session_state:
        st.session_state.workflow_active_steps = set()

    # Render step cards in a grid
    cols = st.columns(3)
    for i, step in enumerate(WORKFLOW_STEPS):
        with cols[i % 3]:
            is_active = step["id"] in st.session_state.workflow_active_steps
            card_class = "step-card step-card-active" if is_active else "step-card"
            status_icon = "[完成]" if is_active else "[待执行]"

            st.markdown(
                f'<div class="{card_class}">'
                f'<div class="step-title">{i+1}. {step["icon_text"]} {step["title"]}</div>'
                f'<div style="font-size:12px; color:#666; margin-top:6px;">{step["description"]}</div>'
                f'<div style="font-size:11px; color:#888; margin-top:4px;">{status_icon}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            with st.expander(f"设置参数 - {step['title']}", expanded=False):
                step_params = {}
                for param in step["params"]:
                    step_params[param] = st.text_input(
                        param, key=f"wf_{step['id']}_{param}",
                        placeholder="输入参数值..."
                    )
                st.info(f"操作提示: {step['tips']}")
                if st.button(f"标记为已完成", key=f"btn_{step['id']}"):
                    st.session_state.workflow_active_steps.add(step["id"])
                    st.rerun()

    # Progress indicator
    completed = len(st.session_state.workflow_active_steps)
    total = len(WORKFLOW_STEPS)
    progress = completed / total
    st.markdown("---")
    st.markdown(f"**实验进度: {completed}/{total} 步骤已完成**")
    st.progress(progress)

    if completed == total:
        st.success("全部实验步骤已完成！请进行最终数据记录与归档。")

    # Reset workflow button
    if st.button("重置工作流"):
        st.session_state.workflow_active_steps = set()
        st.rerun()

    # ---- Auto-generate protocol from M5 ----
    st.markdown("---")
    st.markdown("**实验操作手册生成**")

    has_m5_data = "m5_top_df" in st.session_state

    if has_m5_data:
        top_df = st.session_state["m5_top_df"]
        target_col = st.session_state.get("m5_target_col", "目标性能")
        feature_cols = st.session_state.get("m5_feature_cols", [])

        st.success(
            f"检测到 M5 配方优化结果 "
            f"(目标: {target_col}, Top {len(top_df)} 配方)"
        )

        with st.expander("Top 5 优化配方预览"):
            st.dataframe(top_df.head(5).round(4), use_container_width=True)

        gen_col1, gen_col2 = st.columns(2)
        with gen_col1:
            selected_rank = st.number_input(
                "选择第几号配方生成手册", 1, min(10, len(top_df)), 1
            )
        with gen_col2:
            material_system = st.text_input(
                "材料体系描述",
                placeholder="例: 氧化铝陶瓷、锂电池正极材料..."
            )

        if st.button("生成实验操作手册", type="primary"):
            formula_row = top_df.iloc[selected_rank - 1]
            formula_data = {
                "材料体系": material_system or "未指定",
                "配方编号": int(selected_rank),
                "成分与工艺参数": {
                    col: round(float(formula_row[col]), 4)
                    for col in feature_cols if col in formula_row.index
                },
                f"{target_col}_预测值": round(float(formula_row.get(target_col + "_预测", formula_row.iloc[-1])), 4),
            }

            with st.spinner("AI 正在生成详细实验操作手册..."):
                protocol = generate_protocol_from_formula(formula_data)

            st.session_state["m7_protocol"] = protocol
            st.session_state["m7_formula_data"] = formula_data

    else:
        st.info("尚未检测到 M5 配方优化结果，您也可以手动输入配方参数")

        manual_formula = st.text_area(
            "手动输入配方参数 (JSON 格式)",
            value='{"材料体系": "氧化铝陶瓷", "Al2O3_%": 95.0, "MgO_%": 0.5, "烧结温度_C": 1600, "保温时间_h": 3}',
            height=100
        )

        if st.button("生成实验操作手册 (手动输入)", type="primary"):
            try:
                formula_data = json.loads(manual_formula)
                with st.spinner("AI 正在生成详细实验操作手册..."):
                    protocol = generate_protocol_from_formula(formula_data)
                st.session_state["m7_protocol"] = protocol
                st.session_state["m7_formula_data"] = formula_data
            except json.JSONDecodeError:
                st.error("JSON 格式有误，请检查输入")

    # Display generated protocol
    if "m7_protocol" in st.session_state:
        st.markdown("---")
        st.markdown("**实验操作手册**")
        st.markdown(st.session_state["m7_protocol"])

        save_col1, save_col2 = st.columns(2)
        with save_col1:
            st.download_button(
                "下载操作手册 (.txt)",
                st.session_state["m7_protocol"].encode("utf-8"),
                "experiment_protocol.txt",
                "text/plain"
            )
        with save_col2:
            if st.button("保存至数据库"):
                path = save_text(
                    st.session_state["m7_protocol"],
                    "recipes",
                    "experiment_protocol.txt"
                )
                formula_data = st.session_state.get("m7_formula_data", {})
                steps_data = [
                    {"step": s["title"], "status": s["id"] in st.session_state.workflow_active_steps}
                    for s in WORKFLOW_STEPS
                ]
                save_record(
                    module="M7",
                    name=f"实验手册 - {formula_data.get('材料体系', '未知')}",
                    file_path=str(path),
                    metadata={"formula": formula_data, "steps": steps_data}
                )
                save_workflow(
                    name=f"工作流 - {formula_data.get('材料体系', '未知')}",
                    steps=steps_data,
                )
                st.success("已保存")
