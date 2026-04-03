"""
launcher.py
-----------
Entry point for PyInstaller packaging.
Launches the Streamlit app in a subprocess and opens the browser.
"""
import sys
import os
import time
import threading
import subprocess
import webbrowser
from pathlib import Path

PORT = 8501
APP_FILE = Path(__file__).parent / "app.py"


def open_browser():
    time.sleep(3)
    webbrowser.open(f"http://localhost:{PORT}")


def run_streamlit():
    # When frozen by PyInstaller, sys._MEIPASS contains unpacked files
    base = getattr(sys, "_MEIPASS", str(Path(__file__).parent))
    app_path = os.path.join(base, "app.py")

    cmd = [
        sys.executable if not getattr(sys, "frozen", False) else os.path.join(base, "streamlit"),
        "run",
        app_path,
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]

    # When packaged, use the bundled streamlit
    if getattr(sys, "frozen", False):
        import streamlit.web.cli as stcli
        sys.argv = [
            "streamlit", "run", app_path,
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ]
        sys.exit(stcli.main())
    else:
        subprocess.run(cmd)


if __name__ == "__main__":
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    run_streamlit()
