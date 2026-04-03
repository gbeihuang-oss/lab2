# lab.spec
# Usage:
#   Windows: pyinstaller lab.spec
#   macOS:   pyinstaller lab.spec

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH)

# Collect all data files
datas = [
    (str(ROOT / "app.py"), "."),
    (str(ROOT / "utils"), "utils"),
    (str(ROOT / "modules"), "modules"),
]

# Streamlit static files
try:
    import streamlit
    st_path = Path(streamlit.__file__).parent
    datas += [
        (str(st_path / "static"), "streamlit/static"),
        (str(st_path / "runtime"), "streamlit/runtime"),
    ]
except ImportError:
    pass

# Collect submodules
hiddenimports = (
    collect_submodules("streamlit")
    + collect_submodules("sklearn")
    + collect_submodules("plotly")
    + collect_submodules("pandas")
    + collect_submodules("numpy")
    + collect_submodules("rdkit")
    + [
        "sqlite3",
        "openai",
        "requests",
        "PIL",
        "openpyxl",
        "scipy",
        "utils.config",
        "utils.database",
        "utils.storage",
        "modules.m1_assistant",
        "modules.m2_molecular",
        "modules.m3_visualization",
        "modules.m4_image_analysis",
        "modules.m5_optimization",
        "modules.m6_prediction",
        "modules.m7_workflow",
        "modules.m8_database",
    ]
)

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Lab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,        # No terminal window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Lab",
)

# macOS .app bundle
app = BUNDLE(
    coll,
    name="Lab.app",
    icon=None,
    bundle_identifier="com.lab.virtuallab",
    info_plist={
        "NSPrincipalClass": "NSApplication",
        "NSAppleScriptEnabled": False,
        "CFBundleDocumentTypes": [],
        "LSMinimumSystemVersion": "10.14",
        "NSHighResolutionCapable": True,
    },
)
