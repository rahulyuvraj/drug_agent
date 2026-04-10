#!/usr/bin/env python3
"""
Launch the Drug Agent Streamlit app on port 8502.

Usage:
    python -m agentic_ai_wf.drug_agent_streamlit.run
    # or
    python agentic_ai_wf/drug_agent_streamlit/run.py

Teammates access via http://<server-ip>:8502
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_PATH = Path(__file__).resolve().parent / "app.py"


def main():
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(APP_PATH),
        "--server.address", "0.0.0.0",
        "--server.port", "8502",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print(f"Starting Drug Agent Streamlit on http://0.0.0.0:8502")
    print(f"App: {APP_PATH}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
