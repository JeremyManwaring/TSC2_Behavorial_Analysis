# TSC2_Behavorial_Analysis

## Unified Extractor

Use `behavior_data_extractor.py` for one consolidated extraction flow.

### Colab Quick Paste

For Colab, copy/paste the full contents of:

- `colab_snippet.py`

This gives you one-cell setup (clone/pull repo + open extractor widget).

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
- auto-render overview graphs for the default scope

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
from behavior_data_extractor import load_auto_context, plot_auto_scope

context = load_auto_context("Jeremy", selected_day=None, default_scope="auto")
# selected_day=None uses most recent folder
plot_auto_scope(context, scope="auto")
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

### CLI usage

```bash
python3 behavior_data_extractor.py --folder Jeremy --mode day --day 260220
python3 behavior_data_extractor.py --folder Jeremy --mode week --anchor-day 260219
python3 behavior_data_extractor.py --folder Jeremy --mode all
```
