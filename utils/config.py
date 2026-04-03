from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Storage
DATA_DIR = ROOT_DIR / "data_storage"
RECIPES_DIR = DATA_DIR / "recipes"
PLOTS_DIR = DATA_DIR / "plots"
SIMULATIONS_DIR = DATA_DIR / "simulations"
IMAGES_DIR = DATA_DIR / "images"
PREDICTIONS_DIR = DATA_DIR / "predictions"
DB_PATH = DATA_DIR / "lab.db"

# API Configuration
GROQ_API_KEY = "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"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "qwen/qwen3-32b"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

MP_API_KEY = "zOvPxw66NtiD8ppSIYRIHn9LdXdQ6Y68"
MP_BASE_URL = "https://api.materialsproject.org"

# App settings
APP_TITLE = "Lab - 虚拟实验室平台"
APP_SUBTITLE = "智能材料研发一体化工作台"

MODULES = {
    "M1 - 研发助手": "m1_assistant",
    "M2 - 分子模拟": "m2_molecular",
    "M3 - 实验数据可视化": "m3_visualization",
    "M4 - 图像识别分析": "m4_image_analysis",
    "M5 - 配方优化": "m5_optimization",
    "M6 - 性能预测": "m6_prediction",
    "M7 - 实验室工作流": "m7_workflow",
    "M8 - 数据库管理": "m8_database",
}
