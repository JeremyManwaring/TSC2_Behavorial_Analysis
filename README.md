# TSC2_Behavorial_Analysis

## Unified Extractor

Use `behavior_data_extractor.py` for one consolidated extraction flow.

### Colab Quick Paste

For Colab, copy/paste the full contents of:

- `colab_snippet.py`

This gives you one-cell setup (force-sync repo to latest `main` + open extractor widget).

If you ever see an import mismatch (stale runtime files), run this hard-refresh cell:

```python
import os
import shutil
import subprocess
import sys

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

DATA_FOLDER = f"{REPO_DIR}/Jeremy"  # replace with your Drive Jeremy path if needed
context = bde.load_auto_context(DATA_FOLDER, selected_day=None, default_scope="auto")
bde.display_all_scope_results(context)
bde.show_extraction_widget(DATA_FOLDER)
```

### Notebook UI (bottom popup button)

```python
from behavior_data_extractor import show_extraction_widget

# Use your Jeremy folder path if different
show_extraction_widget("Jeremy")
```

This adds an `Open Data Extractor` button near the bottom-right of the notebook.
Inside the popup you can:

- input the folder path
- optionally choose an anchor day (or keep `Most Recent`)
- click `Load + Plot` once and automatically prepare all scopes:
- `day` (anchor day)
- `week` (anchor day and previous 6 days)
- `all` (all folders)
- auto-render labeled and separated sections for `day`, `week`, and `all`
- overview now uses a lick raster panel (replacing `Trials per Day`)
- week outcome panel is a 7-day time-series line graph (Hit/Miss/False Alarm/Correct Reject only)
- rolling-rate axes auto-scale to avoid clipping

After loading, all scope-specific data is available:

- `trials_day`, `lick_df_day`, `stimulus_data_day`, `header_data_day`
- `trials_week`, `lick_df_week`, `stimulus_data_week`, `header_data_week`
- `trials_all`, `lick_df_all`, `stimulus_data_all`, `header_data_all`

Default aliases for existing analysis cells are auto-set:

- `trials`, `lick_df`, `stimulus_data`, `header_data`

Use `Default: Auto/Day/Week/All` in the popup to control which scope feeds those aliases.

### Notebook API

Auto-load everything and set aliases in one call:

```python
from behavior_data_extractor import display_all_scope_results, load_auto_context

context = load_auto_context("Jeremy", selected_day=None, default_scope="auto")
# selected_day=None uses most recent folder
display_all_scope_results(context)
```

Manually switch alias scope later:

```python
from behavior_data_extractor import apply_scope_aliases

apply_scope_aliases(context, "day")   # or "week", "all", "auto"
```

Legacy single-scope loading is still available:

```python
from behavior_data_extractor import load_into_namespace

analysis = load_into_namespace("Jeremy", mode="week")
```

One-day full analysis suite (compatible with the original workflow) is available:

```python
from behavior_data_extractor import run_one_day_analysis_suite

run_one_day_analysis_suite(context.day, window_size=10)
```

### CLI usage

```bash
python3 behavior_data_extractor.py --folder Jeremy --mode day --day 260220
python3 behavior_data_extractor.py --folder Jeremy --mode week --anchor-day 260219
python3 behavior_data_extractor.py --folder Jeremy --mode all
```
