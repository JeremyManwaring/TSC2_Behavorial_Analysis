from __future__ import annotations

import argparse
import __main__
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Tuple, Union

import pandas as pd


DATE_FOLDER_PATTERN = re.compile(r"^\d{6}$")
SESSION_PATTERN = re.compile(r"_S(\d+)_")


@dataclass
class ExtractionResult:
    root_folder: Path
    mode: str
    selected_folders: List[Path]
    files_by_type: Dict[str, List[Path]]
    dataframes: Dict[str, pd.DataFrame]

    def summary_lines(self) -> List[str]:
        lines = [
            f"Root folder: {self.root_folder}",
            f"Mode: {self.mode}",
            f"Selected day folders ({len(self.selected_folders)}): "
            + ", ".join(folder.name for folder in self.selected_folders),
        ]
        for file_type in ("trials", "stimuli", "header", "lick"):
            file_count = len(self.files_by_type.get(file_type, []))
            row_count = len(self.dataframes.get(file_type, pd.DataFrame()))
            lines.append(f"{file_type}: {file_count} files, {row_count} rows")
        return lines

    def summary_text(self) -> str:
        return "\n".join(self.summary_lines())


LAST_EXTRACTION_RESULT: Optional[ExtractionResult] = None


@dataclass
class AnalysisData:
    extraction: ExtractionResult
    trials: pd.DataFrame
    lick_df: pd.DataFrame
    stimulus_data: pd.DataFrame
    header_data: pd.DataFrame

    def summary_text(self) -> str:
        return (
            f"{self.extraction.summary_text()}\n"
            f"analysis trials rows: {len(self.trials)}\n"
            f"analysis lick_df rows: {len(self.lick_df)}"
        )


LAST_ANALYSIS_DATA: Optional[AnalysisData] = None


def parse_day_folder(folder_name: str) -> Optional[date]:
    if not DATE_FOLDER_PATTERN.match(folder_name):
        return None
    try:
        yy = int(folder_name[0:2])
        mm = int(folder_name[2:4])
        dd = int(folder_name[4:6])
        return datetime(2000 + yy, mm, dd).date()
    except ValueError:
        return None


def list_day_folders(root_folder: Union[str, Path]) -> List[Tuple[Path, date]]:
    root = Path(root_folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {root}")

    folders: List[Tuple[Path, date]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        parsed = parse_day_folder(child.name)
        if parsed is not None:
            folders.append((child, parsed))
    folders.sort(key=lambda item: item[1])
    return folders


def _normalize_root_and_day(
    root_folder: Union[str, Path], mode: str, day_folder: Optional[str]
) -> Tuple[Path, Optional[str]]:
    root = Path(root_folder).expanduser().resolve()
    parsed_root_day = parse_day_folder(root.name)

    if parsed_root_day is not None and root.parent.exists():
        if mode == "day" and day_folder is None:
            day_folder = root.name
        root = root.parent

    return root, day_folder


def _select_folders(
    dated_folders: List[Tuple[Path, date]], mode: str, day_folder: Optional[str]
) -> List[Tuple[Path, date]]:
    if not dated_folders:
        raise ValueError("No day folders found. Expected folder names like YYMMDD.")

    if mode == "day":
        if day_folder:
            selected = [item for item in dated_folders if item[0].name == day_folder]
            if not selected:
                raise ValueError(f"Day folder not found: {day_folder}")
            return selected
        return [dated_folders[-1]]

    if mode == "week":
        newest_day = dated_folders[-1][1]
        start_day = newest_day - timedelta(days=6)
        return [item for item in dated_folders if start_day <= item[1] <= newest_day]

    if mode == "all":
        return dated_folders

    raise ValueError(f"Invalid mode: {mode}. Use 'day', 'week', or 'all'.")


def _session_id_map(selected_folders: List[Tuple[Path, date]]) -> Dict[str, int]:
    sorted_names = [folder.name for folder, _ in sorted(selected_folders, key=lambda x: x[1])]
    return {name: idx for idx, name in enumerate(sorted_names)}


def _read_table(path: Path, file_type: str, session_id: int, folder_date: date) -> pd.DataFrame:
    read_kwargs = {"sep": "\t"}
    if file_type == "lick":
        read_kwargs["header"] = None

    try:
        frame = pd.read_csv(path, **read_kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        fallback_kwargs = {"header": None} if file_type == "lick" else {}
        frame = pd.read_csv(path, **fallback_kwargs)

    if frame.empty:
        return frame

    if file_type == "lick" and 0 in frame.columns:
        frame = frame.rename(columns={0: "TrNum_Lick"})

    session_match = SESSION_PATTERN.search(path.name)
    session_number = int(session_match.group(1)) if session_match else -1

    frame["SessionID"] = session_id
    frame["_day_folder"] = path.parent.name
    frame["_date"] = folder_date.isoformat()
    frame["_session_number"] = session_number
    frame["_source_file"] = path.name
    frame["_source_path"] = str(path)
    return frame


def _collect_files(selected_folders: List[Tuple[Path, date]]) -> Dict[str, List[Path]]:
    files_by_type = {"trials": [], "stimuli": [], "header": [], "lick": []}
    for folder, _ in selected_folders:
        files_by_type["trials"].extend(sorted(folder.glob("*_trials.csv")))
        files_by_type["stimuli"].extend(sorted(folder.glob("*_stimuli.csv")))
        files_by_type["header"].extend(sorted(folder.glob("*_header.csv")))
        files_by_type["lick"].extend(sorted(folder.glob("*_lick.csv")))
    return files_by_type


def extract_behavior_data(
    root_folder: Union[str, Path], mode: str = "day", day_folder: Optional[str] = None
) -> ExtractionResult:
    normalized_mode = mode.strip().lower()
    normalized_root, normalized_day = _normalize_root_and_day(root_folder, normalized_mode, day_folder)
    dated_folders = list_day_folders(normalized_root)
    selected = _select_folders(dated_folders, normalized_mode, normalized_day)
    files_by_type = _collect_files(selected)
    sessions = _session_id_map(selected)

    loaded: Dict[str, List[pd.DataFrame]] = {"trials": [], "stimuli": [], "header": [], "lick": []}
    folder_dates = {folder.name: parsed_date for folder, parsed_date in selected}

    for file_type, paths in files_by_type.items():
        for path in paths:
            folder_name = path.parent.name
            frame = _read_table(path, file_type, sessions[folder_name], folder_dates[folder_name])
            if not frame.empty:
                loaded[file_type].append(frame)

    combined = {
        key: pd.concat(value, ignore_index=True) if value else pd.DataFrame()
        for key, value in loaded.items()
    }

    result = ExtractionResult(
        root_folder=normalized_root,
        mode=normalized_mode,
        selected_folders=[folder for folder, _ in selected],
        files_by_type=files_by_type,
        dataframes=combined,
    )

    global LAST_EXTRACTION_RESULT
    LAST_EXTRACTION_RESULT = result
    return result


def _with_analysis_columns(frame: pd.DataFrame, mouse_label: str) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out

    if "_date" in out.columns:
        date_series = pd.to_datetime(out["_date"], errors="coerce")
    else:
        date_series = pd.Series(pd.NaT, index=out.index)
    out["date"] = date_series.dt.date

    iso_week = date_series.dt.isocalendar().week
    out["week"] = iso_week.astype("Int64")

    if "_session_number" in out.columns:
        out["session"] = pd.to_numeric(out["_session_number"], errors="coerce").fillna(-1).astype(int)
    else:
        out["session"] = -1

    out["mouse"] = mouse_label

    day_folders = out["_day_folder"].astype(str) if "_day_folder" in out.columns else pd.Series("unknown", index=out.index)
    fallback_session = (
        pd.to_numeric(out["SessionID"], errors="coerce").fillna(-1).astype(int)
        if "SessionID" in out.columns
        else pd.Series(-1, index=out.index)
    )

    session_ids: List[str] = []
    for idx in out.index:
        date_val = out.at[idx, "date"]
        session_val = int(out.at[idx, "session"])
        if isinstance(date_val, date) and session_val >= 0:
            session_ids.append(f"{date_val.isoformat()}_S{session_val}")
        else:
            session_ids.append(f"{day_folders.loc[idx]}_S{int(fallback_session.loc[idx])}")
    out["session_id"] = session_ids
    return out


def _lick_timestamp_columns(lick_table: pd.DataFrame) -> List[Union[int, str]]:
    metadata_columns = {
        "TrNum_Lick",
        "SessionID",
        "_day_folder",
        "_date",
        "_session_number",
        "_source_file",
        "_source_path",
        "mouse",
        "date",
        "week",
        "session",
        "session_id",
    }
    lick_columns: List[Union[int, str]] = []
    for col in lick_table.columns:
        if col in metadata_columns:
            continue
        if isinstance(col, int) and col > 0:
            lick_columns.append(col)
            continue
        if isinstance(col, str) and col.isdigit() and int(col) > 0:
            lick_columns.append(col)
    return sorted(lick_columns, key=lambda c: int(c))


def _build_relative_lick_df(trials: pd.DataFrame, lick_table: pd.DataFrame) -> pd.DataFrame:
    expected_cols = ["mouse", "date", "session", "TrNum", "week", "session_id", "lick_times_rel_stim_ms"]
    if trials.empty or lick_table.empty:
        return pd.DataFrame(columns=expected_cols)

    trial_lookup: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    for _, row in trials.iterrows():
        try:
            tr_num = int(row["TrNum"])
        except Exception:
            continue
        session_num = int(row.get("session", -1))
        day_folder = str(row.get("_day_folder", ""))
        trial_lookup[(day_folder, session_num, tr_num)] = {
            "mouse": row.get("mouse"),
            "date": row.get("date"),
            "session": session_num,
            "TrNum": tr_num,
            "week": int(row.get("week")) if pd.notna(row.get("week")) else -1,
            "session_id": row.get("session_id"),
            "TrStartTime": float(row.get("TrStartTime", 0) or 0),
            "StimOnsetTime": float(row.get("StimOnsetTime", 500) or 500),
        }

    lick_cols = _lick_timestamp_columns(lick_table)
    aggregated: Dict[Tuple[str, int], Dict[str, object]] = {}

    for _, row in lick_table.iterrows():
        tr_num_val = row.get("TrNum_Lick")
        try:
            tr_num = int(tr_num_val)
        except Exception:
            continue

        try:
            session_num = int(row.get("session", row.get("_session_number", -1)))
        except Exception:
            session_num = -1
        day_folder = str(row.get("_day_folder", ""))

        trial_info = trial_lookup.get((day_folder, session_num, tr_num))
        if trial_info is None:
            continue

        lick_times: List[float] = []
        for col in lick_cols:
            value = row.get(col)
            try:
                t = float(value)
            except Exception:
                continue
            if pd.isna(t) or t <= 0 or t > 1e7:
                continue
            rel_stim = (t - float(trial_info["TrStartTime"])) - float(trial_info["StimOnsetTime"])
            lick_times.append(rel_stim)

        aggregate_key = (str(trial_info["session_id"]), tr_num)
        if aggregate_key not in aggregated:
            aggregated[aggregate_key] = {
                "mouse": trial_info["mouse"],
                "date": trial_info["date"],
                "session": trial_info["session"],
                "TrNum": trial_info["TrNum"],
                "week": trial_info["week"],
                "session_id": trial_info["session_id"],
                "lick_times_rel_stim_ms": [],
            }
        aggregated[aggregate_key]["lick_times_rel_stim_ms"].extend(lick_times)

    lick_rows = list(aggregated.values())
    lick_rows.sort(
        key=lambda row: (
            row["date"] if isinstance(row["date"], date) else date.min,
            int(row["session"]),
            int(row["TrNum"]),
        )
    )
    return pd.DataFrame(lick_rows, columns=expected_cols)


def build_analysis_data(extraction: ExtractionResult) -> AnalysisData:
    mouse_label = extraction.root_folder.name

    trials = _with_analysis_columns(extraction.dataframes.get("trials", pd.DataFrame()), mouse_label)
    stimulus_data = _with_analysis_columns(extraction.dataframes.get("stimuli", pd.DataFrame()), mouse_label)
    header_data = _with_analysis_columns(extraction.dataframes.get("header", pd.DataFrame()), mouse_label)
    raw_lick = _with_analysis_columns(extraction.dataframes.get("lick", pd.DataFrame()), mouse_label)

    if not trials.empty and "TrNum" in trials.columns:
        trials["TrNum"] = pd.to_numeric(trials["TrNum"], errors="coerce").fillna(-1).astype(int)

    outcome_map = {1: "Hit", 2: "Correct Reject", 3: "False Alarm", 4: "Miss"}
    if not trials.empty and "TrOutcome" in trials.columns:
        trials["outcome_label"] = pd.to_numeric(trials["TrOutcome"], errors="coerce").map(outcome_map)

    time_cols = {"RWStartTime", "RWEndTime", "TrStartTime", "StimOnsetTime"}
    if not trials.empty and time_cols.issubset(set(trials.columns)):
        trials["RWStart_rel_stim"] = trials["RWStartTime"] - trials["TrStartTime"] - trials["StimOnsetTime"]
        trials["RWEnd_rel_stim"] = trials["RWEndTime"] - trials["TrStartTime"] - trials["StimOnsetTime"]

    lick_df = _build_relative_lick_df(trials, raw_lick)

    analysis = AnalysisData(
        extraction=extraction,
        trials=trials,
        lick_df=lick_df,
        stimulus_data=stimulus_data,
        header_data=header_data,
    )

    global LAST_ANALYSIS_DATA
    LAST_ANALYSIS_DATA = analysis
    return analysis


def load_analysis_data(
    root_folder: Union[str, Path], mode: str = "all", day_folder: Optional[str] = None
) -> AnalysisData:
    extraction = extract_behavior_data(root_folder=root_folder, mode=mode, day_folder=day_folder)
    return build_analysis_data(extraction)


def load_into_namespace(
    root_folder: Union[str, Path],
    mode: str = "all",
    day_folder: Optional[str] = None,
    namespace: Optional[MutableMapping[str, object]] = None,
) -> AnalysisData:
    target_namespace = namespace if namespace is not None else __main__.__dict__
    analysis = load_analysis_data(root_folder=root_folder, mode=mode, day_folder=day_folder)
    target_namespace["trials"] = analysis.trials
    target_namespace["lick_df"] = analysis.lick_df
    target_namespace["stimulus_data"] = analysis.stimulus_data
    target_namespace["header_data"] = analysis.header_data
    target_namespace["extraction_result"] = analysis.extraction
    target_namespace["analysis_data"] = analysis
    return analysis


def show_extraction_widget(default_folder: Union[str, Path] = "Jeremy"):
    import ipywidgets as widgets
    from IPython.display import display

    folder_input = widgets.Text(
        value=str(Path(default_folder).expanduser()),
        description="Folder:",
        placeholder="/path/to/Jeremy",
        layout=widgets.Layout(width="100%"),
    )
    mode_input = widgets.ToggleButtons(
        options=[
            ("Single Day", "day"),
            ("One Week (Most Recent 7 Days)", "week"),
            ("All Time", "all"),
        ],
        description="Mode:",
    )
    day_input = widgets.Dropdown(
        options=[],
        description="Day:",
        disabled=False,
        layout=widgets.Layout(width="100%"),
    )
    scan_button = widgets.Button(description="Scan Folders", button_style="info")
    extract_button = widgets.Button(description="Extract Data", button_style="success")
    close_button = widgets.Button(description="Close")
    status = widgets.HTML()
    output = widgets.Output(layout=widgets.Layout(max_height="260px", overflow_y="auto"))

    launch_button = widgets.Button(
        description="Open Data Extractor",
        button_style="primary",
        tooltip="Open extraction panel",
        layout=widgets.Layout(
            position="fixed",
            bottom="20px",
            right="20px",
            width="180px",
            z_index="9999",
        ),
    )

    panel = widgets.VBox(
        [
            widgets.HTML("<b>Behavior Data Extractor</b>"),
            folder_input,
            mode_input,
            day_input,
            widgets.HBox([scan_button, extract_button, close_button]),
            status,
            output,
        ],
        layout=widgets.Layout(
            display="none",
            position="fixed",
            bottom="70px",
            right="20px",
            width="520px",
            border="1px solid #d0d0d0",
            padding="10px",
            background_color="white",
            z_index="9999",
        ),
    )

    def refresh_day_options() -> None:
        try:
            folders = list_day_folders(folder_input.value)
            options = [folder.name for folder, _ in folders]
            day_input.options = options
            if options:
                day_input.value = options[-1]
                status.value = (
                    f"<span style='color:#2f6f37'>Found {len(options)} folders. "
                    f"Most recent: <b>{options[-1]}</b>.</span>"
                )
            else:
                status.value = "<span style='color:#a33'>No YYMMDD folders found.</span>"
        except Exception as exc:
            day_input.options = []
            status.value = f"<span style='color:#a33'>{exc}</span>"

    def update_day_enabled(*_args) -> None:
        day_input.disabled = mode_input.value != "day"

    def handle_launch(_):
        panel.layout.display = "none" if panel.layout.display == "flex" else "flex"
        if panel.layout.display == "flex":
            refresh_day_options()
            update_day_enabled()

    def handle_close(_):
        panel.layout.display = "none"

    def handle_scan(_):
        refresh_day_options()
        update_day_enabled()

    def handle_extract(_):
        with output:
            output.clear_output()
            try:
                selected_day = day_input.value if mode_input.value == "day" else None
                analysis = load_into_namespace(
                    root_folder=folder_input.value,
                    mode=mode_input.value,
                    day_folder=selected_day,
                )
                print(analysis.summary_text())
                print()
                print("Notebook variables updated:")
                print("  trials")
                print("  lick_df")
                print("  stimulus_data")
                print("  header_data")
                print("  extraction_result")
                print("  analysis_data")
            except Exception as exc:
                print(f"Extraction failed: {exc}")

    launch_button.on_click(handle_launch)
    close_button.on_click(handle_close)
    scan_button.on_click(handle_scan)
    extract_button.on_click(handle_extract)
    mode_input.observe(update_day_enabled, names="value")

    display(launch_button)
    display(panel)
    return launch_button, panel


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract behavior data from Jeremy day folders.")
    parser.add_argument(
        "--folder",
        default="Jeremy",
        help="Path to the Jeremy folder that contains YYMMDD day folders.",
    )
    parser.add_argument(
        "--mode",
        choices=["day", "week", "all"],
        default="day",
        help="Extraction scope: one day, most recent 7-day window, or all folders.",
    )
    parser.add_argument(
        "--day",
        default=None,
        help="Specific day folder (YYMMDD). Used only with --mode day.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    result = extract_behavior_data(args.folder, mode=args.mode, day_folder=args.day)
    print(result.summary_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
