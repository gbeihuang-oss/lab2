# Lab - 虚拟实验室平台

材料科学智能研发一体化工作台，基于 Streamlit + Qwen3-32B (Groq) + Scikit-learn 构建。

---

## 快速启动（开发模式）

```bash
# 1. 克隆或解压项目
cd lab

# 2. 创建虚拟环境（推荐）
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动应用
streamlit run app.py
```

浏览器访问: http://localhost:8501

---

## 项目结构

```
lab/
├── app.py                   # 主入口，全局 CSS，侧边栏导航
├── launcher.py              # PyInstaller 打包入口
├── lab.spec                 # PyInstaller 打包配置
├── requirements.txt         # Python 依赖
├── build_windows.bat        # Windows 一键打包
├── build_mac.sh             # macOS 一键打包
│
├── utils/
│   ├── config.py            # 全局配置（API Key、路径等）
│   ├── database.py          # SQLite 数据库操作
│   └── storage.py           # 本地文件存储
│
├── modules/
│   ├── m1_assistant.py      # M1 研发助手（Qwen + Materials Project）
│   ├── m2_molecular.py      # M2 分子模拟（RDKit + py3Dmol）
│   ├── m3_visualization.py  # M3 实验数据可视化（Nature/Science 风格）
│   ├── m4_image_analysis.py # M4 图像识别（多模态 LLM）
│   ├── m5_optimization.py   # M5 配方优化（GBR + 高通量筛选）
│   ├── m6_prediction.py     # M6 性能预测（多模型对比）
│   ├── m7_workflow.py       # M7 实验室工作流（可视化卡片 + AI 生成手册）
│   └── m8_database.py       # M8 数据库管理（增删改查 + 预览）
│
└── data_storage/            # 自动创建，本地持久化
    ├── recipes/             # 配方、实验手册
    ├── plots/               # 图表文件
    ├── simulations/         # 分子结构文件
    ├── images/              # 图像及分析报告
    └── predictions/         # 性能预测结果
```

---

## 功能模块说明

| 模块 | 功能 | 核心技术 |
|------|------|----------|
| M1 研发助手 | 专业问答 + Materials Project 实时检索 | Qwen3-32B, Groq API |
| M2 分子模拟 | SMILES 转 3D 模型，可拖拽旋转缩放 | RDKit, py3Dmol (3Dmol.js) |
| M3 实验数据可视化 | DSC/TGA/XRD 等科研级绘图 | Plotly, Nature 风格主题 |
| M4 图像识别分析 | SEM/TEM 图像 AI 分析报告 | Llama-4 Vision, Groq |
| M5 配方优化 | GBR + 5000 组高通量筛选 | Scikit-learn GBR |
| M6 性能预测 | 多模型对比 + 新样本预测 | GBR / RF / ET / MLP |
| M7 实验室工作流 | 可视化步骤卡片 + AI 实验手册 | Qwen3-32B |
| M8 数据库管理 | 历史记录增删查预览 | SQLite, Pathlib |

---

## 模块联动说明

```
M5 配方优化
    ├─→ M1 研发助手（发送 Top20 配方，请求机理解释）
    └─→ M7 实验室工作流（自动生成实验操作手册）
```

---

## 打包为独立可执行文件

### Windows (.exe)

```bat
build_windows.bat
```

输出: `dist\Lab\Lab.exe`（双击运行，自动打开浏览器）

### macOS (.app)

```bash
chmod +x build_mac.sh
./build_mac.sh
```

输出: `dist/Lab.app` 和 `dist/Lab.dmg`

### 打包注意事项

- RDKit 打包需要额外的 DLL/dylib，建议在目标平台上原生构建
- 首次启动需约 3-5 秒初始化 Streamlit 服务
- 打包后应用体积约 500 MB - 1 GB（含所有 ML 库）
- 推荐 Python 3.10 或 3.11 进行打包

---

## 依赖安装故障排查

### RDKit 安装

```bash
# 推荐方式（pip）
pip install rdkit

# 备选方式（conda）
conda install -c conda-forge rdkit
```

### Plotly 静态导出 (kaleido)

```bash
pip install kaleido
# macOS 若报错：
pip install kaleido==0.2.1
```

### Windows 中文路径问题

确保项目路径不含中文字符，使用纯英文路径运行。

---

## API 配置

所有 API Key 集中在 `utils/config.py`：

```python
GROQ_API_KEY   = "..."   # Groq (Qwen3-32B)
MP_API_KEY     = "..."   # Materials Project
```

如需更换模型，修改 `GROQ_MODEL` 常量即可。

---

## 系统要求

- Python 3.10+
- 内存: 建议 8 GB 以上
- 网络: M1 / M4 模块需访问 Groq API 和 Materials Project API
- 操作系统: Windows 10+ / macOS 12+ / Linux (Ubuntu 20.04+)
