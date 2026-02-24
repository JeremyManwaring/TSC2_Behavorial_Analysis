TSC2 Behavioral Analysis
Unified Behavioral Data Extraction Pipeline

This repository provides a consolidated extraction workflow for multi-session mouse behavioral datasets.

The core extraction logic lives in:

behavior_data_extractor.py

For Google Colab users, a one-cell setup script is provided to automatically:

Clone the latest version of the repository

Mount Google Drive

Load behavioral data

Display scope results

Launch the extraction widget

🚀 Google Colab Quick Start

Copy and paste the following into a single Colab cell:

import os
import shutil
import subprocess
import sys
from google.colab import drive

REPO_URL = "https://github.com/JeremyManwaring/TSC2_Behavorial_Analysis.git"
REPO_DIR = "/content/TSC2_Behavorial_Analysis"

# Ensure fresh clone of latest main branch
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", "main", REPO_URL, REPO_DIR],
    check=True,
)

# Add repo to Python path
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Clear cached module (prevents stale imports)
if "behavior_data_extractor" in sys.modules:
    del sys.modules["behavior_data_extractor"]

import behavior_data_extractor as bde

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Set path to your dataset folder in Drive
DATA_FOLDER = "/content/drive/MyDrive/Jeremy/"

# Auto-load context
context = bde.load_auto_context(
    DATA_FOLDER,
    selected_day=None,
    default_scope="auto"
)

# Display scope results
bde.display_all_scope_results(context)

# Launch interactive extraction widget
bde.show_extraction_widget(DATA_FOLDER)
📁 Expected Google Drive Structure

Your behavioral data should be organized as:

My Drive/
└── Jeremy/
    ├── 260120/
    ├── 260121/
    ├── 260122/
    └── ...

DATA_FOLDER must point to the directory containing all session folders:

/content/drive/MyDrive/Jeremy/

If your data is nested deeper, update the path accordingly.
