#!/usr/bin/env python3
"""Launcher utilities for the wakeword Gradio interface."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP_PATH = REPO_ROOT / "gradio_app.py"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"


def install_requirements() -> bool:
    """Install the application's Python dependencies."""
    print("Installing required packages...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)])
        print("Dependencies are up to date.")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Failed to install packages: {exc}")
        return False


def check_environment() -> bool:
    """Smoke-test core dependencies so the user sees actionable feedback."""
    print("Checking Python environment...")

    try:
        import torch  # noqa: F401
        import gradio  # noqa: F401
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
        import matplotlib  # noqa: F401
        import plotly  # noqa: F401
        print("Core packages are importable.")
        return True
    except ImportError as exc:
        print(f"Missing package: {exc}")
        return False


def launch_app() -> None:
    """Run the main Gradio application."""
    if not APP_PATH.exists():
        print(f"Could not find {APP_PATH.name}.")
        sys.exit(1)

    print("Launching wakeword training UI...")
    try:
        subprocess.run([sys.executable, str(APP_PATH)], check=False)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")


def main() -> None:
    print("Wakeword Training Gradio Launcher")
    print("=" * 38)

    if not check_environment():
        print("\nRun `pip install -r requirements.txt` and try again.")
        choice = input("Install or refresh dependencies now? (y/N): ").strip().lower()
        if choice == "y" and not install_requirements():
            sys.exit(1)
    else:
        choice = input("Install or refresh dependencies anyway? (y/N): ").strip().lower()
        if choice == "y" and not install_requirements():
            sys.exit(1)

    print("\nOpen the browser tab Gradio prints to begin training.")
    launch_app()


if __name__ == "__main__":
    main()

