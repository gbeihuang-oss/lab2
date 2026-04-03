import streamlit as st
import streamlit.components.v1 as components
import json
from utils.database import save_record
from utils.storage import save_text

# Try to import rdkit and py3Dmol
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

try:
    import py3Dmol
    PY3DMOL_OK = True
except ImportError:
    PY3DMOL_OK = False

COMMON_SMILES = {
    "水 (H2O)": "O",
    "乙醇 (C2H5OH)": "CCO",
    "苯 (C6H6)": "c1ccccc1",
    "葡萄糖 (C6H12O6)": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "咖啡因": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "阿司匹林": "CC(=O)Oc1ccccc1C(=O)O",
    "二氧化钛 (TiO2, rutile proxy)": "O=[Ti]=O",
    "氧化铝 (Al2O3 proxy)": "[Al](=O)O[Al]=O",
    "聚苯乙烯单体 (苯乙烯)": "C=Cc1ccccc1",
    "PVDF 单体 (偏氟乙烯)": "FC(F)=C",
}

def smiles_to_3d_molblock(smiles: str):
    """Convert SMILES to 3D mol block string."""
    if not RDKIT_OK:
        return None, "RDKit 未安装，请运行: pip install rdkit"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"无法解析 SMILES: {smiles}"
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result == -1:
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result == -1:
            return None, "3D 坐标生成失败，请检查分子结构"
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        molblock = Chem.MolToMolBlock(mol)
        return molblock, None
    except Exception as e:
        return None, str(e)

def get_molecule_properties(smiles: str) -> dict:
    """Calculate basic molecular properties."""
    if not RDKIT_OK:
        return {}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "分子量 (g/mol)": round(Descriptors.MolWt(mol), 4),
            "精确质量 (g/mol)": round(Descriptors.ExactMolWt(mol), 4),
            "LogP (亲脂性)": round(Descriptors.MolLogP(mol), 4),
            "氢键供体数": rdMolDescriptors.CalcNumHBD(mol),
            "氢键受体数": rdMolDescriptors.CalcNumHBA(mol),
            "可旋转键数": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "极性表面积 (A2)": round(Descriptors.TPSA(mol), 4),
            "重原子数": mol.GetNumHeavyAtoms(),
            "环数": rdMolDescriptors.CalcNumRings(mol),
            "芳香环数": rdMolDescriptors.CalcNumAromaticRings(mol),
        }
    except Exception:
        return {}

def render_3d_viewer(molblock: str, style: str = "stick", bg_color: str = "0xffffff"):
    """Render 3D molecule using py3Dmol via HTML component."""
    style_map = {
        "stick": '{"stick": {"radius": 0.15}}',
        "sphere": '{"sphere": {"radius": 0.4}}',
        "line": '{"line": {}}',
        "cartoon": '{"cartoon": {"color": "spectrum"}}',
        "surface": '{"sphere": {"radius": 0.4}}',
    }
    js_style = style_map.get(style, '{"stick": {"radius": 0.15}}')

    molblock_escaped = molblock.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #{bg_color[2:]}; }}
        #viewer {{ width: 100%; height: 480px; position: relative; }}
    </style>
</head>
<body>
    <div id="viewer"></div>
    <script>
        var config = {{ backgroundColor: {bg_color} }};
        var viewer = $3Dmol.createViewer(document.getElementById('viewer'), config);
        var molData = `{molblock_escaped}`;
        viewer.addModel(molData, 'mol');
        viewer.setStyle({{}}, {js_style});
        viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.1, color: 'white'}});
        viewer.zoomTo();
        viewer.spin(true);
        viewer.render();
    </script>
</body>
</html>
"""
    components.html(html, height=490)

def render():
    st.markdown('<div class="module-header">M2 - 分子模拟</div>', unsafe_allow_html=True)
    st.caption("输入 SMILES 字符串或化学式，生成可交互的 3D 分子模型")

    if not RDKIT_OK:
        st.error("RDKit 未安装。请运行: pip install rdkit")
        st.code("pip install rdkit", language="bash")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**输入方式**")
        input_mode = st.radio("", ["预设分子", "手动输入 SMILES"], horizontal=True)

        if input_mode == "预设分子":
            preset = st.selectbox("选择常见分子", list(COMMON_SMILES.keys()))
            smiles_input = COMMON_SMILES[preset]
            st.code(smiles_input, language="text")
        else:
            smiles_input = st.text_input(
                "SMILES 字符串",
                value="c1ccccc1",
                help="例: CC(=O)Oc1ccccc1C(=O)O (阿司匹林)"
            )

        st.markdown("**渲染设置**")
        vis_style = st.selectbox("显示风格", ["stick", "sphere", "line"])
        bg_color = st.color_picker("背景颜色", "#ffffff")
        bg_hex = "0x" + bg_color[1:]

        run_btn = st.button("生成 3D 模型", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("**分子性质**")
        if smiles_input and RDKIT_OK:
            props = get_molecule_properties(smiles_input)
            if props:
                for k, v in props.items():
                    st.metric(k, v)

    with col_right:
        if run_btn or smiles_input:
            if not RDKIT_OK:
                st.warning("RDKit 未安装，无法生成 3D 模型")
            else:
                with st.spinner("正在计算 3D 坐标并渲染..."):
                    molblock, err = smiles_to_3d_molblock(smiles_input)
                if err:
                    st.error(f"生成失败: {err}")
                else:
                    st.markdown("**3D 分子模型** (可拖拽旋转 / 滚轮缩放)")
                    render_3d_viewer(molblock, style=vis_style, bg_color=bg_hex)

                    # Save option
                    st.markdown("---")
                    save_col1, save_col2 = st.columns(2)
                    with save_col1:
                        save_name = st.text_input("保存名称", value=f"molecule_{smiles_input[:10]}")
                    with save_col2:
                        if st.button("保存至数据库"):
                            path = save_text(molblock, "simulations", f"{save_name}.mol")
                            save_record(
                                module="M2",
                                name=save_name,
                                file_path=str(path),
                                metadata={"smiles": smiles_input, "style": vis_style}
                            )
                            st.success("已保存")

                    # Download molblock
                    st.download_button(
                        "下载 .mol 文件",
                        data=molblock.encode(),
                        file_name=f"{smiles_input[:12]}.mol",
                        mime="chemical/x-mdl-molfile"
                    )
