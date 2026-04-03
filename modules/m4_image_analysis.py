import streamlit as st
import base64
import io
from openai import OpenAI
from utils.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_VISION_MODEL
from utils.database import save_record
from utils.storage import save_file

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

IMAGE_ANALYSIS_PROMPTS = {
    "SEM - 扫描电子显微镜": """请对这张 SEM（扫描电子显微镜）图像进行专业的材料学分析，包括：
1. 微观形貌描述：颗粒/晶粒形状、尺寸范围估算、分布均匀性
2. 表面特征：孔隙率定性评估、表面粗糙度、缺陷或裂纹
3. 粒径分布趋势：是否存在团聚、单分散性评价
4. 微结构特征：晶界、相界、夹杂物
5. 材料学推断：可能的制备工艺、热处理历史
6. 对材料性能的影响分析
请给出学术水平的描述，引用相关领域的表征标准。""",

    "TEM - 透射电子显微镜": """请对这张 TEM（透射电子显微镜）图像进行专业分析，包括：
1. 晶体结构特征：晶格条纹、衍射斑点（如有）
2. 纳米结构：颗粒尺寸（估算）、核壳结构、异质结
3. 缺陷分析：位错、层错、孪晶界
4. 界面特征：相界、晶界厚度
5. 非晶/结晶比例的定性判断
6. 与材料性能的关联分析（光学、电学、力学等）
请给出学术水平的描述，参考相关表征文献。""",

    "XRD 衍射图谱": """请分析这张 XRD（X射线衍射）图谱，包括：
1. 主要衍射峰位置（2θ值）及对应晶面指数
2. 物相识别：可能的晶体结构和空间群
3. 结晶度评估：峰宽与非晶散射包
4. 使用 Scherrer 方程估算晶粒尺寸（若峰形清晰）
5. 晶格参数的定性变化（峰位偏移）
6. 相含量比例（定性）
7. 样品制备质量评估
请参考 JCPDS/ICDD 标准卡片格式给出分析。""",

    "光学显微镜图像": """请对这张光学显微镜图像进行材料学分析，包括：
1. 微观组织描述：相组成、晶粒形态
2. 晶粒尺寸估算（与标尺对应）
3. 相分布：基体相、析出相、夹杂物
4. 腐蚀特征（如为金相图）
5. 组织均匀性评价
6. 热处理状态推断
请给出专业的金相分析描述。""",

    "通用材料图像分析": """请对这张材料科学相关图像进行全面的专业分析，识别：
1. 图像类型及测试方法推断
2. 观察到的主要特征和规律
3. 材料微结构的关键信息
4. 对材料性质的定性推断
5. 建议的进一步表征方向
请基于材料科学原理给出专业意见。""",
}

def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def analyze_image_with_llm(image_bytes: bytes, mime_type: str, analysis_type: str,
                            custom_prompt: str = "", temperature: float = 0.3) -> str:
    """Send image to vision LLM and return analysis."""
    base64_image = encode_image_to_base64(image_bytes)
    prompt = IMAGE_ANALYSIS_PROMPTS.get(analysis_type, IMAGE_ANALYSIS_PROMPTS["通用材料图像分析"])
    if custom_prompt:
        prompt += f"\n\n补充要求: {custom_prompt}"

    try:
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位专业的材料科学家，擅长解读各类材料表征图像。请基于图像内容给出严谨、专业的分析，所有推断需基于可见的图像特征，不得无中生有。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=2048,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"图像分析失败: {str(e)}\n\n备注: 当前使用的视觉模型为 {GROQ_VISION_MODEL}，若出现 API 错误请检查模型可用性。"

def render():
    st.markdown('<div class="module-header">M4 - 图像/PDF 识别分析</div>', unsafe_allow_html=True)
    st.caption("上传 SEM/TEM/XRD/光学显微镜等图像，AI 自动识别微观形貌并生成专业分析报告")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**文件上传**")
        uploaded = st.file_uploader(
            "上传图像文件",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"],
            help="支持 PNG / JPG / TIFF 格式"
        )
        if uploaded:
            st.success(f"已加载: {uploaded.name}")

        st.markdown("**分析设置**")
        analysis_type = st.selectbox("分析类型", list(IMAGE_ANALYSIS_PROMPTS.keys()))
        temperature = st.slider("分析深度", 0.1, 0.8, 0.3, 0.1,
                                help="值越低输出越严谨保守，值越高越发散全面")

    if uploaded is None:
        st.info("请在侧边栏上传待分析的图像文件")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**支持的图像类型**")
            types = [
                ("SEM 扫描电子显微镜", "颗粒形貌、孔隙结构、断口分析"),
                ("TEM 透射电子显微镜", "晶格像、衍射斑点、纳米结构"),
                ("XRD 衍射图谱", "物相识别、结晶度、晶粒尺寸"),
                ("光学显微镜", "金相组织、相分布、晶粒尺寸"),
            ]
            for name, desc in types:
                st.markdown(f"**{name}**: {desc}")
        return

    # Display image
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.markdown("**上传图像**")
        image_bytes = uploaded.read()
        st.image(image_bytes, use_column_width=True, caption=uploaded.name)

        # Image info
        from PIL import Image
        img_pil = Image.open(io.BytesIO(image_bytes))
        st.caption(f"尺寸: {img_pil.width} x {img_pil.height} px | 模式: {img_pil.mode} | "
                   f"大小: {len(image_bytes)/1024:.1f} KB")

        # Custom prompt
        custom_prompt = st.text_area(
            "补充分析要求（可选）",
            placeholder="例如：重点分析图中左上角区域的纳米颗粒分布，并估算平均粒径...",
            height=80
        )

        analyze_btn = st.button("开始 AI 分析", type="primary", use_container_width=True)

    with col_result:
        st.markdown("**分析报告**")

        # Determine MIME type
        suffix = uploaded.name.lower().split(".")[-1]
        mime_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "bmp": "image/bmp",
            "tiff": "image/tiff", "tif": "image/tiff",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        if analyze_btn:
            with st.spinner(f"正在调用 AI 模型分析图像 ({analysis_type})..."):
                result = analyze_image_with_llm(
                    image_bytes, mime_type, analysis_type,
                    custom_prompt=custom_prompt,
                    temperature=temperature
                )

            st.session_state["m4_last_result"] = result
            st.session_state["m4_last_filename"] = uploaded.name

        if "m4_last_result" in st.session_state:
            result_text = st.session_state["m4_last_result"]
            st.markdown(result_text)

            st.markdown("---")
            save_col1, save_col2 = st.columns(2)

            with save_col1:
                st.download_button(
                    "下载分析报告 (.txt)",
                    result_text.encode("utf-8"),
                    f"analysis_{uploaded.name}.txt",
                    "text/plain"
                )

            with save_col2:
                if st.button("保存至数据库"):
                    img_path = save_file(image_bytes, "images", uploaded.name)
                    report_path = save_file(
                        result_text.encode("utf-8"), "images",
                        f"report_{uploaded.name}.txt"
                    )
                    save_record(
                        module="M4",
                        name=f"{analysis_type} - {uploaded.name}",
                        file_path=str(report_path),
                        metadata={
                            "analysis_type": analysis_type,
                            "image_file": str(img_path),
                            "image_name": uploaded.name,
                        }
                    )
                    st.success("已保存")
        else:
            st.info("点击「开始 AI 分析」后，分析报告将显示在此处")
