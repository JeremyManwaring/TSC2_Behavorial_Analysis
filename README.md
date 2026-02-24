# TSC2_Behavorial_Analysis

## Unified Extractor

Use `behavior_data_extractor.py` for one consolidated extraction flow.

### Notebook UI (bottom popup button)

```python
from behavior_data_extractor import show_extraction_widget

# Use your Jeremy folder path if different
show_extraction_widget("Jeremy")
```

This adds an `Open Data Extractor` button near the bottom-right of the notebook.
Inside the popup you can:

- input the folder path
- choose `Single Day`, `One Week (Most Recent 7 Days)`, or `All Time`
- extract data directly into DataFrames

Loaded data is stored in:

- `LAST_EXTRACTION_RESULT.dataframes["trials"]`
- `LAST_EXTRACTION_RESULT.dataframes["stimuli"]`
- `LAST_EXTRACTION_RESULT.dataframes["header"]`
- `LAST_EXTRACTION_RESULT.dataframes["lick"]`

To directly refresh notebook variables used by existing analysis cells:

```python
from behavior_data_extractor import load_into_namespace

analysis = load_into_namespace("Jeremy", mode="week")
# Populates: trials, lick_df, stimulus_data, header_data, extraction_result, analysis_data
```

### CLI usage

```bash
python3 behavior_data_extractor.py --folder Jeremy --mode day --day 260220
python3 behavior_data_extractor.py --folder Jeremy --mode week
python3 behavior_data_extractor.py --folder Jeremy --mode all
```
