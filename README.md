# TSC2_Behavorial_Analysis

## Unified Extractor

Use `behavior_data_extractor.py` for one consolidated extraction flow.

### Colab Quick Paste

For Colab, copy/paste the full contents of:

- `colab_snippet.py`

This gives you one-cell setup (force-sync repo to latest `main` + open extractor widget).

`import os
import shutil
import subprocess
import sys
from google.colab import drive

REPO_URL = "https://github.com/JeremyManwaring/TSC2_Behavorial_Analysis.git"
REPO_DIR = "/content/TSC2_Behavorial_Analysis"

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", "main", REPO_URL, REPO_DIR],
    check=True,
)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

if "behavior_data_extractor" in sys.modules:
    del sys.modules["behavior_data_extractor"]

import behavior_data_extractor as bde

drive.mount('/content/drive', force_remount=True)

DATA_FOLDER = "/content/drive/MyDrive/Jeremy/"  # Corrected path to your Drive Jeremy folder
context = bde.load_auto_context(DATA_FOLDER, selected_day=None, default_scope="auto")
bde.display_all_scope_results(context)
bde.show_extraction_widget(DATA_FOLDER)`
