import shutil
import json
from pathlib import Path
from datetime import datetime
from utils.config import (
    DATA_DIR, RECIPES_DIR, PLOTS_DIR,
    SIMULATIONS_DIR, IMAGES_DIR, PREDICTIONS_DIR
)

_DIRS = [DATA_DIR, RECIPES_DIR, PLOTS_DIR, SIMULATIONS_DIR, IMAGES_DIR, PREDICTIONS_DIR]

def init_storage():
    for d in _DIRS:
        d.mkdir(parents=True, exist_ok=True)

def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_file(source_bytes: bytes, category: str, filename: str) -> Path:
    """Save bytes to a category folder and return the saved path."""
    cat_map = {
        "recipes": RECIPES_DIR,
        "plots": PLOTS_DIR,
        "simulations": SIMULATIONS_DIR,
        "images": IMAGES_DIR,
        "predictions": PREDICTIONS_DIR,
    }
    target_dir = cat_map.get(category, DATA_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    # Unique filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    save_path = target_dir / f"{stem}_{_timestamp()}{suffix}"
    save_path.write_bytes(source_bytes)
    return save_path

def save_text(content: str, category: str, filename: str) -> Path:
    return save_file(content.encode("utf-8"), category, filename)

def save_json(data: dict, category: str, filename: str) -> Path:
    return save_file(
        json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        category, filename
    )

def list_files(category: str = None):
    if category:
        cat_map = {
            "recipes": RECIPES_DIR,
            "plots": PLOTS_DIR,
            "simulations": SIMULATIONS_DIR,
            "images": IMAGES_DIR,
            "predictions": PREDICTIONS_DIR,
        }
        d = cat_map.get(category, DATA_DIR)
        return sorted(d.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        files = []
        for d in [RECIPES_DIR, PLOTS_DIR, SIMULATIONS_DIR, IMAGES_DIR, PREDICTIONS_DIR]:
            files.extend(d.glob("*"))
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

def get_file_info(path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size_kb": round(stat.st_size / 1024, 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "suffix": path.suffix,
    }
