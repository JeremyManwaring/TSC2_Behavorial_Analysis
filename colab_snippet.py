"""
Copy this entire file into a single Google Colab code cell and run it.

What it does:
1) Clones (or updates) this repo under /content
2) Imports behavior_data_extractor.py
3) Opens the floating extraction widget

Optional:
- Set DATA_FOLDER to your own Jeremy data folder in Drive.
- Set AUTO_PRELOAD_AND_PLOT to True to render graphs immediately.
"""

import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/JeremyManwaring/TSC2_Behavorial_Analysis.git"
REPO_DIR = Path("/content/TSC2_Behavorial_Analysis")

# Set this to your Drive path if you want your own data, for example:
# DATA_FOLDER = "/content/drive/MyDrive/TSC2/Jeremy"
DATA_FOLDER = None
AUTO_PRELOAD_AND_PLOT = True


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


if REPO_DIR.exists():
    run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
else:
    run(["git", "clone", REPO_URL, str(REPO_DIR)])

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

if DATA_FOLDER is None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass

from behavior_data_extractor import load_auto_context, plot_auto_scope, show_extraction_widget

if DATA_FOLDER is None:
    DATA_FOLDER = str(REPO_DIR / "Jeremy")
    print(f"Using bundled sample data: {DATA_FOLDER}")
else:
    print(f"Using DATA_FOLDER: {DATA_FOLDER}")

show_extraction_widget(DATA_FOLDER)
if AUTO_PRELOAD_AND_PLOT:
    context = load_auto_context(DATA_FOLDER, selected_day=None, default_scope="auto")
    rendered = plot_auto_scope(context, scope="auto")
    print(f"Preloaded and rendered scope: {rendered}")
else:
    print("Optional preload:")
    print("context = load_auto_context(DATA_FOLDER, selected_day=None, default_scope='auto')")
    print("plot_auto_scope(context, scope='auto')")
