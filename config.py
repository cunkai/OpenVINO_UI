# config.py
from pathlib import Path
import sys
import os

# Determine if we are running in a bundled environment
if getattr(sys, 'frozen', False):
    # Bundled App: The executable is in the top level of the resource dir
    BASE_DIR = Path(sys._MEIPASS)
    # For bundled app, store DB in a standard app data location
    APP_DATA_DIR = Path(os.getenv('APPDATA') or os.path.expanduser('~/.config')) / 'com.tauri.openvino.chat'
else:
    # Development: The script is in 'src-tauri/backend'
    BASE_DIR = Path(__file__).resolve().parent.parent
    APP_DATA_DIR = BASE_DIR

# Ensure the App Data Directory exists
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_BASE_DIR = BASE_DIR / "model"
