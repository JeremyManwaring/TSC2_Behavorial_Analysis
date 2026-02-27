"""
Copy this entire file into a single Google Colab code cell and run it.

What it does:
1) Clones (or updates) this repo under /content
2) Imports behavior_data_extractor.py
3) Opens the floating extraction widget

Optional:
- Set DATA_FOLDER to your own Jeremy data folder in Drive.
"""

import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/JeremyManwaring/TSC2_Behavorial_Analysis.git"
REPO_DIR = Path("/content/TSC2_Behavorial_Analysis")

# Set this to your Drive path if you want your own data, for example:
# DATA_FOLDER = "/content/drive/MyDrive/TSC2/Jeremy"
DATA_FOLDER = None


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


if REPO_DIR.exists():
    # Force-sync to origin/main to avoid stale Colab copies.
    run(["git", "-C", str(REPO_DIR), "fetch", "origin", "main"])
    run(["git", "-C", str(REPO_DIR), "checkout", "-B", "main", "origin/main"])
else:
    run(["git", "clone", "--branch", "main", "--single-branch", REPO_URL, str(REPO_DIR)])

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

if DATA_FOLDER is None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass

# Avoid stale module objects if this cell is re-run in the same runtime.
if "behavior_data_extractor" in sys.modules:
    del sys.modules["behavior_data_extractor"]

from behavior_data_extractor import show_extraction_widget

if DATA_FOLDER is None:
    DATA_FOLDER = str(REPO_DIR / "Jeremy")
    print(f"Using bundled sample data: {DATA_FOLDER}")
else:
    print(f"Using DATA_FOLDER: {DATA_FOLDER}")

show_extraction_widget(DATA_FOLDER)
print("Data extractor ready. Select folders and click 'Load + Plot' in the panel.")
