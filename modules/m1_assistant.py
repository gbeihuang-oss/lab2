import streamlit as st
import requests
from openai import OpenAI
from utils.config import (
    GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL,
    MP_API_KEY, MP_BASE_URL
)
from utils.database import save_chat_message, get_chat_history, clear_chat_history

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

SYSTEM_PROMPT = """你是一位材料科学领域的高级研究助手，拥有化学、物理、冶金、聚合物科学等多学科背景。

【核心准则】
1. 所有回答必须严格基于真实的学术期刊（如 Nature、Science、Advanced Materials、Acta Materialia 等）、专业教科书或经过同行评审的公开实验数据。
2. 每个关键论断必须提供具体引用，格式为：作者, 期刊名, 年份, DOI 或卷期页码。
3. 严禁虚构数据、引用不存在的文献或提供未经验证的信息。
4. 若不确定某项数据，必须明确说明"该数据需查阅原始文献确认"。
5. 回答时优先给出定量参数（如具体温度、压力、成分比例、性能数值）。
6. 若用户询问特定材料的性质，你将优先从 Materials Project 数据库获取第一性原理计算结果。

【回答结构】
- 直接回答问题核心
- 给出关键参数与机理
- 提供参考文献
- 指出局限性或注意事项（如适用）

请用中文回答，专业术语可保留英文原文。"""

def query_materials_project(formula: str) -> str:
    """Query Materials Project API for material properties."""
    try:
        url = f"{MP_BASE_URL}/summary/"
        params = {
            "formula": formula,
            "_fields": "material_id,formula_pretty,band_gap,energy_per_atom,density,volume,nsites,symmetry,theoretical",
            "_limit": 5
        }
        headers = {"X-API-KEY": MP_API_KEY}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                lines = [f"[Materials Project 数据库检索结果 - 化学式: {formula}]"]
                for item in data[:3]:
                    lines.append(
                        f"  material_id: {item.get('material_id', 'N/A')} | "
                        f"带隙: {item.get('band_gap', 'N/A')} eV | "
                        f"密度: {item.get('density', 'N/A')} g/cm3 | "
                        f"每原子能量: {item.get('energy_per_atom', 'N/A')} eV/atom | "
                        f"晶系: {item.get('symmetry', {}).get('crystal_system', 'N/A')}"
                    )
                return "\n".join(lines)
    except Exception as e:
        return f"[Materials Project 查询失败: {e}]"
    return ""

def detect_material_query(text: str) -> str:
    """Try to extract chemical formula from user query for MP lookup."""
    import re
    patterns = [
        r'\b([A-Z][a-z]?(?:\d+(?:\.\d+)?)?){2,8}\b',
        r'([A-Z][a-z]?\d*)+(?:O|N|C|S|P|F|Cl|Br|I)\d*',
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(re.findall(pat, text))
    # Simple heuristic: if looks like a formula (mix of letters + digits)
    for token in text.split():
        if re.match(r'^[A-Z][a-zA-Z0-9]{1,15}$', token) and any(c.isdigit() for c in token):
            return token
    return ""

def render():
    st.markdown('<div class="module-header">M1 - 研发助手</div>', unsafe_allow_html=True)
    st.caption("基于 Qwen3-32B 的材料科学专业问答，集成 Materials Project 数据库实时检索")

    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("**助手设置**")
        temperature = st.slider("创造性 (Temperature)", 0.0, 1.0, 0.3, 0.05)
        max_tokens = st.selectbox("最大回复长度", [1024, 2048, 4096, 8192], index=1)
        enable_mp = st.checkbox("启用 Materials Project 检索", value=True)
        if st.button("清空对话历史"):
            clear_chat_history()
            st.session_state.messages = []
            st.rerun()

    # Initialize session messages from DB
    if "messages" not in st.session_state:
        st.session_state.messages = get_chat_history(limit=40)

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role_label = "用户" if msg["role"] == "user" else "助手"
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("请输入您的材料科学问题（例如：TiO2 的光催化机理是什么？）")

    if user_input:
        # Optionally fetch MP data
        mp_context = ""
        if enable_mp:
            formula = detect_material_query(user_input)
            if formula:
                mp_context = query_materials_project(formula)

        # Build messages for API
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if mp_context:
            api_messages.append({
                "role": "system",
                "content": f"以下是从 Materials Project 数据库检索到的相关数据，请在回答中优先参考：\n{mp_context}"
            })
        for m in st.session_state.messages[-20:]:
            api_messages.append({"role": m["role"], "content": m["content"]})
        api_messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_chat_message("user", user_input)

        # Stream assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                stream = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    placeholder.markdown(full_response + "...")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"API 调用失败: {e}"
                placeholder.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_chat_message("assistant", full_response)

    # External prompt injection from M5
    if "m5_prompt" in st.session_state and st.session_state.m5_prompt:
        with st.expander("来自 M5 配方优化的分析请求", expanded=True):
            st.info(st.session_state.m5_prompt[:500] + "...")
            if st.button("发送至助手进行分析"):
                user_input = st.session_state.m5_prompt
                st.session_state.m5_prompt = None
                st.rerun()
