import streamlit as st
import pandas as pd
from pathlib import Path
import json
from utils.database import get_records, delete_record
from utils.storage import list_files, get_file_info, DATA_DIR
from utils.config import RECIPES_DIR, PLOTS_DIR, SIMULATIONS_DIR, IMAGES_DIR, PREDICTIONS_DIR

MODULE_NAMES = {
    "M1": "研发助手",
    "M2": "分子模拟",
    "M3": "实验数据可视化",
    "M4": "图像识别分析",
    "M5": "配方优化",
    "M6": "性能预测",
    "M7": "虚拟实验室工作流",
}

CAT_DIRS = {
    "recipes": RECIPES_DIR,
    "plots": PLOTS_DIR,
    "simulations": SIMULATIONS_DIR,
    "images": IMAGES_DIR,
    "predictions": PREDICTIONS_DIR,
}

def get_dir_size(directory: Path) -> str:
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    if total < 1024:
        return f"{total} B"
    elif total < 1024 ** 2:
        return f"{total/1024:.1f} KB"
    else:
        return f"{total/1024**2:.2f} MB"

def render():
    st.markdown('<div class="module-header">M8 - 本地数据库管理</div>', unsafe_allow_html=True)
    st.caption("查看、检索、预览和管理所有模块保存的历史数据")

    # Storage overview
    st.markdown("**存储概览**")
    ov_cols = st.columns(len(CAT_DIRS))
    for i, (cat, d) in enumerate(CAT_DIRS.items()):
        files = list(d.glob("*")) if d.exists() else []
        size = get_dir_size(d) if d.exists() else "0 B"
        ov_cols[i].metric(
            cat.upper(),
            f"{len(files)} 个文件",
            size,
        )

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["数据库记录", "文件浏览器", "存储统计"])

    with tab1:
        st.markdown("**历史记录**")

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            module_filter = st.selectbox(
                "按模块筛选",
                ["全部"] + list(MODULE_NAMES.keys()),
            )
        with fc2:
            search_kw = st.text_input("关键词搜索", placeholder="搜索记录名称...")
        with fc3:
            sort_by = st.selectbox("排序方式", ["时间（最新优先）", "名称 A-Z", "模块"])

        # Fetch records
        if module_filter == "全部":
            records = get_records()
        else:
            records = get_records(module_filter)

        # Apply keyword filter
        if search_kw:
            records = [r for r in records if search_kw.lower() in r["name"].lower()]

        # Sort
        if sort_by == "名称 A-Z":
            records.sort(key=lambda r: r["name"])
        elif sort_by == "模块":
            records.sort(key=lambda r: r["module"])

        if not records:
            st.info("暂无符合条件的历史记录")
        else:
            st.caption(f"共找到 {len(records)} 条记录")
            for rec in records:
                with st.expander(
                    f"[{rec['module']} - {MODULE_NAMES.get(rec['module'], '')}] "
                    f"{rec['name']}  |  {rec['created_at']}"
                ):
                    col_info, col_action = st.columns([3, 1])

                    with col_info:
                        if rec.get("metadata"):
                            try:
                                meta = json.loads(rec["metadata"]) if isinstance(rec["metadata"], str) else rec["metadata"]
                                st.json(meta)
                            except Exception:
                                st.text(str(rec.get("metadata", "")))

                        if rec.get("file_path"):
                            p = Path(rec["file_path"])
                            st.caption(f"文件路径: {p}")
                            if p.exists():
                                st.caption(f"文件大小: {p.stat().st_size / 1024:.2f} KB")

                                # Preview based on file type
                                suffix = p.suffix.lower()
                                if suffix in [".txt", ".json", ".mol", ".csv"]:
                                    try:
                                        content = p.read_text(encoding="utf-8")
                                        st.code(content[:1500] + ("..." if len(content) > 1500 else ""), language="text")
                                    except Exception:
                                        st.caption("(无法预览此文件)")
                                elif suffix in [".png", ".jpg", ".jpeg"]:
                                    st.image(str(p), use_column_width=True)

                                # Download
                                st.download_button(
                                    "下载此文件",
                                    data=p.read_bytes(),
                                    file_name=p.name,
                                    key=f"dl_{rec['id']}"
                                )
                            else:
                                st.warning("关联文件已移动或删除")

                    with col_action:
                        if st.button("删除记录", key=f"del_{rec['id']}", type="secondary"):
                            delete_record(rec["id"])
                            st.success("已删除")
                            st.rerun()

    with tab2:
        st.markdown("**文件浏览器**")

        cat_sel = st.selectbox("浏览分类", ["全部"] + list(CAT_DIRS.keys()))

        if cat_sel == "全部":
            files = list_files()
        else:
            files = list_files(cat_sel)

        if not files:
            st.info("该分类下暂无文件")
        else:
            file_infos = [get_file_info(f) for f in files if f.is_file()]
            file_df = pd.DataFrame(file_infos)
            file_df.index += 1
            st.dataframe(file_df[["name", "size_kb", "modified", "suffix"]], use_container_width=True)

            # Batch delete
            st.markdown("---")
            del_names = st.multiselect(
                "选择要删除的文件",
                [f["name"] for f in file_infos]
            )
            if del_names and st.button("批量删除所选文件", type="secondary"):
                for info in file_infos:
                    if info["name"] in del_names:
                        p = Path(info["path"])
                        if p.exists():
                            p.unlink()
                st.success(f"已删除 {len(del_names)} 个文件")
                st.rerun()

    with tab3:
        st.markdown("**存储用量统计**")

        # Per-category stats
        stat_data = []
        for cat, d in CAT_DIRS.items():
            if d.exists():
                files = list(d.glob("*"))
                total_bytes = sum(f.stat().st_size for f in files if f.is_file())
                stat_data.append({
                    "分类": cat,
                    "文件数量": len([f for f in files if f.is_file()]),
                    "总大小 (KB)": round(total_bytes / 1024, 2),
                    "路径": str(d),
                })

        stat_df = pd.DataFrame(stat_data)
        st.dataframe(stat_df, use_container_width=True)

        total_kb = stat_df["总大小 (KB)"].sum()
        st.metric("总存储用量", f"{total_kb:.2f} KB ({total_kb/1024:.2f} MB)")

        # DB path info
        from utils.config import DB_PATH
        if DB_PATH.exists():
            st.caption(f"数据库文件: {DB_PATH} ({DB_PATH.stat().st_size/1024:.2f} KB)")

        if st.button("清空全部文件 (谨慎操作)", type="secondary"):
            confirm = st.checkbox("确认清空所有存储文件")
            if confirm:
                for cat, d in CAT_DIRS.items():
                    if d.exists():
                        for f in d.glob("*"):
                            if f.is_file():
                                f.unlink()
                st.success("已清空全部文件")
                st.rerun()
