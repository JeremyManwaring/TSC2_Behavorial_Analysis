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


@dataclass
class AutoExtractionContext:
    root_folder: Path
    selected_day: str
    day: AnalysisData
    week: AnalysisData
    all_time: AnalysisData
    resolved_scope: str

    def summary_text(self) -> str:
        return "\n".join(
            [
                f"Root folder: {self.root_folder}",
                f"Selected day: {self.selected_day}",
                f"Resolved default scope: {self.resolved_scope}",
                f"Day rows: trials={len(self.day.trials)}, lick_df={len(self.day.lick_df)}",
                f"Week rows: trials={len(self.week.trials)}, lick_df={len(self.week.lick_df)}",
                f"All rows: trials={len(self.all_time.trials)}, lick_df={len(self.all_time.lick_df)}",
                "Week folders: " + ", ".join(folder.name for folder in self.week.extraction.selected_folders),
            ]
        )


LAST_AUTO_CONTEXT: Optional[AutoExtractionContext] = None


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
        if mode in ("day", "week") and day_folder is None:
            day_folder = root.name
        root = root.parent

    return root, day_folder


def _select_folders(
    dated_folders: List[Tuple[Path, date]],
    mode: str,
    day_folder: Optional[str],
    anchor_day: Optional[str] = None,
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
        anchor_date = dated_folders[-1][1]
        if anchor_day:
            selected_anchor = [item for item in dated_folders if item[0].name == anchor_day]
            if not selected_anchor:
                raise ValueError(f"Anchor day folder not found: {anchor_day}")
            anchor_date = selected_anchor[0][1]
        start_day = anchor_date - timedelta(days=6)
        return [item for item in dated_folders if start_day <= item[1] <= anchor_date]

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
    root_folder: Union[str, Path],
    mode: str = "day",
    day_folder: Optional[str] = None,
    anchor_day: Optional[str] = None,
) -> ExtractionResult:
    normalized_mode = mode.strip().lower()
    requested_anchor = anchor_day if anchor_day is not None else day_folder
    normalized_root, normalized_anchor = _normalize_root_and_day(root_folder, normalized_mode, requested_anchor)
    dated_folders = list_day_folders(normalized_root)
    if normalized_mode == "week":
        selected = _select_folders(
            dated_folders,
            normalized_mode,
            day_folder=None,
            anchor_day=normalized_anchor,
        )
    else:
        selected = _select_folders(
            dated_folders,
            normalized_mode,
            day_folder=normalized_anchor,
        )
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


def _resolve_stimulus_time_mode(trials: pd.DataFrame, tolerance_ms: float = 1.0) -> str:
    required = {"TrStartTime", "StimOnsetTime", "TrEndTime"}
    if trials.empty or not required.issubset(set(trials.columns)):
        return "relative"

    tr_start = pd.to_numeric(trials["TrStartTime"], errors="coerce")
    stim_on = pd.to_numeric(trials["StimOnsetTime"], errors="coerce")
    tr_end = pd.to_numeric(trials["TrEndTime"], errors="coerce")
    valid_mask = tr_start.notna() & stim_on.notna() & tr_end.notna() & (tr_end >= tr_start - tolerance_ms)
    if not bool(valid_mask.any()):
        return "relative"

    start_valid = tr_start[valid_mask]
    stim_valid = stim_on[valid_mask]
    end_valid = tr_end[valid_mask]
    abs_hits = int(((stim_valid >= start_valid - tolerance_ms) & (stim_valid <= end_valid + tolerance_ms)).sum())
    n_valid = int(valid_mask.sum())
    if abs_hits >= (n_valid / 2.0):
        return "absolute"
    return "relative"


def _stimulus_absolute_timestamp(tr_start_ms: float, stim_on_ms: float, stim_mode: str) -> float:
    if stim_mode == "absolute":
        return stim_on_ms
    return tr_start_ms + stim_on_ms


def _normalize_lick_timestamp(
    lick_value_ms: float,
    tr_start_ms: float,
    tr_end_ms: float,
    stim_abs_ms: float,
    session_min_ms: Optional[float],
    session_max_ms: Optional[float],
    stim_mode: str,
) -> Optional[float]:
    if not pd.notna(lick_value_ms) or lick_value_ms < 0 or lick_value_ms > 2e9:
        return None
    if not pd.notna(stim_abs_ms) or not pd.notna(tr_start_ms):
        return None

    rel_if_abs = lick_value_ms - stim_abs_ms
    rel_if_rel = (tr_start_ms + lick_value_ms) - stim_abs_ms
    plausible_min = -10000.0
    plausible_max = 120000.0

    abs_plausible = plausible_min <= rel_if_abs <= plausible_max
    rel_plausible = plausible_min <= rel_if_rel <= plausible_max

    if abs_plausible and not rel_plausible:
        return lick_value_ms
    if rel_plausible and not abs_plausible:
        return tr_start_ms + lick_value_ms

    trial_dur_ms = max(tr_end_ms - tr_start_ms, 0.0) if pd.notna(tr_end_ms) else 0.0
    session_like = (
        session_min_ms is not None
        and session_max_ms is not None
        and session_min_ms - 2000.0 <= lick_value_ms <= session_max_ms + 2000.0
    )
    relative_like = -1000.0 <= lick_value_ms <= trial_dur_ms + 30000.0

    if stim_mode == "absolute":
        if session_like:
            return lick_value_ms
        if relative_like:
            return tr_start_ms + lick_value_ms
    else:
        if relative_like:
            return tr_start_ms + lick_value_ms
        if session_like:
            return lick_value_ms

    if abs(rel_if_abs) <= abs(rel_if_rel):
        return lick_value_ms
    return tr_start_ms + lick_value_ms


def _build_relative_lick_df(trials: pd.DataFrame, lick_table: pd.DataFrame) -> pd.DataFrame:
    expected_cols = ["mouse", "date", "session", "TrNum", "week", "session_id", "lick_times_rel_stim_ms"]
    if trials.empty or lick_table.empty:
        return pd.DataFrame(columns=expected_cols)

    stim_mode = _resolve_stimulus_time_mode(trials)
    trial_lookup: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    session_bounds: Dict[Tuple[str, int], Dict[str, float]] = {}
    for _, row in trials.iterrows():
        try:
            tr_num = int(row["TrNum"])
        except Exception:
            continue
        session_num = int(row.get("session", -1))
        day_folder = str(row.get("_day_folder", ""))
        tr_start = float(row.get("TrStartTime", 0) or 0)
        tr_end = float(row.get("TrEndTime", tr_start) or tr_start)
        stim_on = float(row.get("StimOnsetTime", 500) or 500)
        stim_abs = _stimulus_absolute_timestamp(tr_start, stim_on, stim_mode)
        trial_lookup[(day_folder, session_num, tr_num)] = {
            "mouse": row.get("mouse"),
            "date": row.get("date"),
            "session": session_num,
            "TrNum": tr_num,
            "week": int(row.get("week")) if pd.notna(row.get("week")) else -1,
            "session_id": row.get("session_id"),
            "TrStartTime": tr_start,
            "TrEndTime": tr_end,
            "StimAbsTime": stim_abs,
        }
        session_key = (day_folder, session_num)
        bounds = session_bounds.setdefault(session_key, {"min": float("inf"), "max": float("-inf")})
        if pd.notna(tr_start):
            bounds["min"] = min(bounds["min"], tr_start)
            bounds["max"] = max(bounds["max"], tr_start)
        if pd.notna(tr_end):
            bounds["min"] = min(bounds["min"], tr_end)
            bounds["max"] = max(bounds["max"], tr_end)

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
        bounds = session_bounds.get((day_folder, session_num), {})
        session_min = bounds.get("min")
        session_max = bounds.get("max")
        if session_min is not None and not pd.notna(session_min):
            session_min = None
        if session_max is not None and not pd.notna(session_max):
            session_max = None

        lick_times: List[float] = []
        for col in lick_cols:
            value = row.get(col)
            try:
                t = float(value)
            except Exception:
                continue
            if pd.isna(t):
                continue
            normalized_abs = _normalize_lick_timestamp(
                lick_value_ms=t,
                tr_start_ms=float(trial_info["TrStartTime"]),
                tr_end_ms=float(trial_info["TrEndTime"]),
                stim_abs_ms=float(trial_info["StimAbsTime"]),
                session_min_ms=float(session_min) if session_min is not None else None,
                session_max_ms=float(session_max) if session_max is not None else None,
                stim_mode=stim_mode,
            )
            if normalized_abs is None:
                continue
            rel_stim = normalized_abs - float(trial_info["StimAbsTime"])
            if not pd.notna(rel_stim):
                continue
            lick_times.append(float(rel_stim))

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
    for row in lick_rows:
        row["lick_times_rel_stim_ms"] = sorted(row["lick_times_rel_stim_ms"])
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
    if not trials.empty and {"TrStartTime", "StimOnsetTime"}.issubset(set(trials.columns)):
        stim_mode = _resolve_stimulus_time_mode(trials)
        tr_start = pd.to_numeric(trials["TrStartTime"], errors="coerce")
        stim_on = pd.to_numeric(trials["StimOnsetTime"], errors="coerce")
        stim_abs = stim_on if stim_mode == "absolute" else tr_start + stim_on
        trials["StimAbsTime"] = stim_abs
        trials["StimTimeMode"] = stim_mode
    if not trials.empty and time_cols.issubset(set(trials.columns)) and "StimAbsTime" in trials.columns:
        rw_start = pd.to_numeric(trials["RWStartTime"], errors="coerce")
        rw_end = pd.to_numeric(trials["RWEndTime"], errors="coerce")
        trials["RWStart_rel_stim"] = rw_start - trials["StimAbsTime"]
        trials["RWEnd_rel_stim"] = rw_end - trials["StimAbsTime"]

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
    root_folder: Union[str, Path],
    mode: str = "all",
    day_folder: Optional[str] = None,
    anchor_day: Optional[str] = None,
) -> AnalysisData:
    extraction = extract_behavior_data(
        root_folder=root_folder,
        mode=mode,
        day_folder=day_folder,
        anchor_day=anchor_day,
    )
    return build_analysis_data(extraction)


def load_into_namespace(
    root_folder: Union[str, Path],
    mode: str = "all",
    day_folder: Optional[str] = None,
    anchor_day: Optional[str] = None,
    namespace: Optional[MutableMapping[str, object]] = None,
) -> AnalysisData:
    target_namespace = namespace if namespace is not None else __main__.__dict__
    analysis = load_analysis_data(
        root_folder=root_folder,
        mode=mode,
        day_folder=day_folder,
        anchor_day=anchor_day,
    )
    target_namespace["trials"] = analysis.trials
    target_namespace["lick_df"] = analysis.lick_df
    target_namespace["stimulus_data"] = analysis.stimulus_data
    target_namespace["header_data"] = analysis.header_data
    target_namespace["extraction_result"] = analysis.extraction
    target_namespace["analysis_data"] = analysis
    return analysis


def _resolve_selected_day(root_folder: Union[str, Path], selected_day: Optional[str]) -> str:
    dated_folders = list_day_folders(root_folder)
    if not dated_folders:
        raise ValueError("No day folders found. Expected folder names like YYMMDD.")
    folder_names = {folder.name for folder, _ in dated_folders}
    if selected_day:
        if selected_day not in folder_names:
            raise ValueError(f"Selected day folder not found: {selected_day}")
        return selected_day
    return dated_folders[-1][0].name


def apply_scope_aliases(
    context: AutoExtractionContext,
    scope: str = "auto",
    namespace: Optional[MutableMapping[str, object]] = None,
) -> str:
    target_namespace = namespace if namespace is not None else __main__.__dict__
    normalized_scope = scope.strip().lower()
    if normalized_scope == "auto":
        normalized_scope = "week"
    if normalized_scope not in {"day", "week", "all"}:
        raise ValueError("Scope must be one of: auto, day, week, all.")

    by_scope = {
        "day": context.day,
        "week": context.week,
        "all": context.all_time,
    }
    selected = by_scope[normalized_scope]

    target_namespace["analysis_scope"] = normalized_scope
    target_namespace["trials"] = selected.trials
    target_namespace["lick_df"] = selected.lick_df
    target_namespace["stimulus_data"] = selected.stimulus_data
    target_namespace["header_data"] = selected.header_data
    target_namespace["extraction_result"] = selected.extraction
    target_namespace["analysis_data"] = selected
    return normalized_scope


def load_auto_context(
    root_folder: Union[str, Path],
    selected_day: Optional[str] = None,
    default_scope: str = "auto",
    namespace: Optional[MutableMapping[str, object]] = None,
) -> AutoExtractionContext:
    target_namespace = namespace if namespace is not None else __main__.__dict__
    resolved_day = _resolve_selected_day(root_folder, selected_day)

    day_data = load_analysis_data(root_folder=root_folder, mode="day", day_folder=resolved_day)
    week_data = load_analysis_data(root_folder=root_folder, mode="week", anchor_day=resolved_day)
    all_data = load_analysis_data(root_folder=root_folder, mode="all")

    context = AutoExtractionContext(
        root_folder=Path(root_folder).expanduser().resolve(),
        selected_day=resolved_day,
        day=day_data,
        week=week_data,
        all_time=all_data,
        resolved_scope="week",
    )

    target_namespace["analysis_context"] = context

    target_namespace["analysis_day"] = day_data
    target_namespace["trials_day"] = day_data.trials
    target_namespace["lick_df_day"] = day_data.lick_df
    target_namespace["stimulus_data_day"] = day_data.stimulus_data
    target_namespace["header_data_day"] = day_data.header_data

    target_namespace["analysis_week"] = week_data
    target_namespace["trials_week"] = week_data.trials
    target_namespace["lick_df_week"] = week_data.lick_df
    target_namespace["stimulus_data_week"] = week_data.stimulus_data
    target_namespace["header_data_week"] = week_data.header_data

    target_namespace["analysis_all"] = all_data
    target_namespace["trials_all"] = all_data.trials
    target_namespace["lick_df_all"] = all_data.lick_df
    target_namespace["stimulus_data_all"] = all_data.stimulus_data
    target_namespace["header_data_all"] = all_data.header_data

    resolved_scope = apply_scope_aliases(context, scope=default_scope, namespace=target_namespace)
    context.resolved_scope = resolved_scope

    global LAST_AUTO_CONTEXT
    LAST_AUTO_CONTEXT = context
    return context


def _ordered_trials(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return trials
    order_cols = [col for col in ["date", "session", "TrNum"] if col in trials.columns]
    if not order_cols:
        return trials.reset_index(drop=True)
    return trials.sort_values(order_cols).reset_index(drop=True)


def _first_nonnegative(values: object) -> float:
    if not isinstance(values, list):
        return float("nan")
    for value in values:
        try:
            numeric = float(value)
        except Exception:
            continue
        if numeric >= 0:
            return numeric
    return float("nan")


def _safe_float(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def calculate_dprime(hits: int, misses: int, false_alarms: int, correct_rejections: int) -> Tuple[float, float, float]:
    from statistics import NormalDist

    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    fa_rate = false_alarms / (false_alarms + correct_rejections) if (false_alarms + correct_rejections) > 0 else 0.0
    hit_rate = min(max(hit_rate, 1e-5), 1 - 1e-5)
    fa_rate = min(max(fa_rate, 1e-5), 1 - 1e-5)
    dist = NormalDist()
    return float(dist.inv_cdf(hit_rate) - dist.inv_cdf(fa_rate)), float(hit_rate), float(fa_rate)


def session_summary(trials: pd.DataFrame) -> Dict[str, float]:
    if trials.empty:
        return {
            "Abortflag": 0.0,
            "TotalTrials": 0.0,
            "GoTrials": 0.0,
            "NoGoTrials": 0.0,
            "Hit": 0.0,
            "Miss": 0.0,
            "CorrectReject": 0.0,
            "FalseAlarm": 0.0,
            "HitPct": 0.0,
            "MissPct": 0.0,
            "CorrectRejectPct": 0.0,
            "FalseAlarmPct": 0.0,
        }

    tr_type = pd.to_numeric(trials.get("TrType"), errors="coerce").fillna(-1).astype(int)
    outcome = pd.to_numeric(trials.get("TrOutcome"), errors="coerce").fillna(-1).astype(int)

    go_trials = int((tr_type == 1).sum())
    nogo_trials = int((tr_type == 0).sum())
    hit = int((outcome == 1).sum())
    correct_reject = int((outcome == 2).sum())
    false_alarm = int((outcome == 3).sum())
    miss = int((outcome == 4).sum())
    abort_flag = 1 if int((outcome == 6).sum()) > 0 else 0

    hit_pct = round((hit / go_trials) * 100, 1) if go_trials > 0 else 0.0
    miss_pct = round((miss / go_trials) * 100, 1) if go_trials > 0 else 0.0
    cr_pct = round((correct_reject / nogo_trials) * 100, 1) if nogo_trials > 0 else 0.0
    fa_pct = round((false_alarm / nogo_trials) * 100, 1) if nogo_trials > 0 else 0.0

    return {
        "Abortflag": float(abort_flag),
        "TotalTrials": float(len(trials)),
        "GoTrials": float(go_trials),
        "NoGoTrials": float(nogo_trials),
        "Hit": float(hit),
        "Miss": float(miss),
        "CorrectReject": float(correct_reject),
        "FalseAlarm": float(false_alarm),
        "HitPct": float(hit_pct),
        "MissPct": float(miss_pct),
        "CorrectRejectPct": float(cr_pct),
        "FalseAlarmPct": float(fa_pct),
    }


def print_session_summary(trials: pd.DataFrame) -> Dict[str, float]:
    summary = session_summary(trials)
    print("=" * 40)
    print("SESSION SUMMARY")
    print("=" * 40)
    print(f"Abortflag is {int(summary['Abortflag'])}")
    print(f"The number of Go trials is {int(summary['GoTrials'])}")
    print(f"The number of No Go trials is {int(summary['NoGoTrials'])}")
    print(f"The number of Hit trials is {int(summary['Hit'])}")
    print(f"The number of Miss trials is {int(summary['Miss'])}")
    print(f"The number of Correct Rejections trials is {int(summary['CorrectReject'])}")
    print(f"The number of False Alarm trials is {int(summary['FalseAlarm'])}\n")
    print(f"The % of Hit trials is {summary['HitPct']}")
    print(f"The % of Miss trials is {summary['MissPct']}")
    print(f"The % of Correct Rejections trials is {summary['CorrectRejectPct']}")
    print(f"The % of False Alarm trials is {summary['FalseAlarmPct']}\n")
    return summary


def _trial_lookup(analysis: AnalysisData) -> Dict[Tuple[str, int], pd.Series]:
    lookup: Dict[Tuple[str, int], pd.Series] = {}
    if analysis.trials.empty:
        return lookup
    for _, row in analysis.trials.iterrows():
        session_id = str(row.get("session_id", ""))
        tr_num = pd.to_numeric(row.get("TrNum"), errors="coerce")
        if pd.isna(tr_num):
            continue
        lookup[(session_id, int(tr_num))] = row
    return lookup


def _first_lick_metrics(analysis: AnalysisData) -> pd.DataFrame:
    cols = [
        "FirstLick_RelativeTrial",
        "FirstLick_RelativeEndTrial",
        "StimRelativeTrial",
        "RWRelativeTrial",
        "Aborted",
    ]
    if analysis.lick_df.empty:
        return pd.DataFrame(columns=cols)

    trial_map = _trial_lookup(analysis)
    rows: List[Dict[str, object]] = []

    for _, lick_row in analysis.lick_df.iterrows():
        lick_times = lick_row.get("lick_times_rel_stim_ms")
        first_rel_stim = _first_nonnegative(lick_times)
        if pd.isna(first_rel_stim):
            continue

        session_id = str(lick_row.get("session_id", ""))
        tr_num = pd.to_numeric(lick_row.get("TrNum"), errors="coerce")
        if pd.isna(tr_num):
            continue
        trial = trial_map.get((session_id, int(tr_num)))
        if trial is None:
            continue

        tr_start = _safe_float(trial.get("TrStartTime"), default=0.0)
        stim_on = _safe_float(trial.get("StimOnsetTime"), default=0.0)
        rw_start = _safe_float(trial.get("RWStartTime"), default=0.0)
        tr_end = _safe_float(trial.get("TrEndTime"), default=0.0)
        outcome_num = pd.to_numeric(trial.get("TrOutcome"), errors="coerce")
        outcome = int(outcome_num) if not pd.isna(outcome_num) else -1

        first_rel_trial_sec = (first_rel_stim + (stim_on - tr_start)) / 1000.0
        trial_dur_sec = (tr_end - tr_start) / 1000.0

        rows.append(
            {
                "FirstLick_RelativeTrial": float(first_rel_trial_sec),
                "FirstLick_RelativeEndTrial": float(first_rel_trial_sec - trial_dur_sec),
                "StimRelativeTrial": float((stim_on - tr_start) / 1000.0),
                "RWRelativeTrial": float((rw_start - tr_start) / 1000.0),
                "Aborted": outcome == 6,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def plot_first_lick_histogram(analysis: AnalysisData, ax=None) -> None:
    import matplotlib.pyplot as plt

    df = _first_lick_metrics(analysis)
    if df.empty:
        if ax is not None:
            ax.text(0.5, 0.5, "No first-lick data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            print("No first-lick data.")
        return

    df = df[(df["FirstLick_RelativeTrial"] >= 0) & (df["FirstLick_RelativeEndTrial"] <= 1)].copy()
    if df.empty:
        if ax is not None:
            ax.text(0.5, 0.5, "No filtered first-lick data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            print("No filtered first-lick data.")
        return

    own_fig = False
    if ax is None:
        own_fig = True
        fig, ax = plt.subplots(figsize=(9, 6))

    aborted_false = df[df["Aborted"] == False]["FirstLick_RelativeTrial"]  # noqa: E712
    aborted_true = df[df["Aborted"] == True]["FirstLick_RelativeTrial"]  # noqa: E712
    ax.hist(aborted_false, bins=40, alpha=0.7, label="Aborted=False", color="#1f77b4")
    if not aborted_true.empty:
        ax.hist(aborted_true, bins=40, alpha=0.55, label="Aborted=True", color="#9aa0a6")

    most_common_rw = float(df["RWRelativeTrial"].mode().iloc[0]) if not df["RWRelativeTrial"].mode().empty else 0.0
    most_common_stim = float(df["StimRelativeTrial"].mode().iloc[0]) if not df["StimRelativeTrial"].mode().empty else 0.0
    ax.axvline(x=most_common_stim, color="red", linestyle="dashed", label="Stim Onset", linewidth=2)
    ax.axvspan(most_common_rw, most_common_rw + 2, color="gray", alpha=0.20, label="Reward Window")

    ax.set_title("Time to First Lick in Trial")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    if own_fig:
        plt.tight_layout()
        plt.show()


def _trial_licks_by_key(analysis: AnalysisData) -> Dict[Tuple[str, int], List[float]]:
    by_key: Dict[Tuple[str, int], List[float]] = {}
    if analysis.lick_df.empty:
        return by_key
    for _, row in analysis.lick_df.iterrows():
        session_id = str(row.get("session_id", ""))
        tr_num = pd.to_numeric(row.get("TrNum"), errors="coerce")
        if pd.isna(tr_num):
            continue
        key = (session_id, int(tr_num))
        licks = row.get("lick_times_rel_stim_ms")
        if isinstance(licks, (list, tuple)):
            lick_values = list(licks)
        elif hasattr(licks, "__iter__") and not isinstance(licks, (str, bytes)):
            try:
                lick_values = list(licks)
            except Exception:
                continue
        else:
            continue
        cleaned: List[float] = []
        for lick in lick_values:
            try:
                value = float(lick)
            except Exception:
                continue
            if pd.isna(value):
                continue
            cleaned.append(value)
        if key not in by_key:
            by_key[key] = []
        by_key[key].extend(cleaned)
    return by_key


def plot_lick_raster(
    analysis: AnalysisData,
    ax=None,
    max_trials: int = 200,
    *,
    time_window_sec: Tuple[float, float] = (-1.0, 4.0),
    bin_width_sec: float = 0.05,
    include_other_outcomes: bool = True,
    standalone_style: str = "faceted",
) -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    trials = _ordered_trials(analysis.trials)
    needed_cols = {"session_id", "TrNum", "TrStartTime", "TrEndTime", "StimOnsetTime"}
    if trials.empty or not needed_cols.issubset(set(trials.columns)):
        if ax is not None:
            ax.text(0.5, 0.5, "No lick raster data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            print("No lick raster data.")
        return 0

    trials = trials.head(max_trials).copy()
    lick_map = _trial_licks_by_key(analysis)
    stim_mode = _resolve_stimulus_time_mode(trials)

    outcome_order = [1, 4, 3, 2]  # Hit, Miss, False Alarm, Correct Rejection
    outcome_labels = {
        1: "Hit",
        4: "Miss",
        3: "False Alarm",
        2: "Correct Rejection",
    }
    outcome_colors = {
        1: "#1f77b4",
        4: "#8a8f98",
        3: "#d62728",
        2: "#2ca02c",
    }
    other_color = "#3a3a3a"
    reward_color = "#f59e0b"
    stim_color = "black"

    records: List[Dict[str, object]] = []
    for _, tr_row in trials.iterrows():
        tr_num_raw = pd.to_numeric(tr_row.get("TrNum"), errors="coerce")
        if pd.isna(tr_num_raw):
            continue
        tr_num = int(float(tr_num_raw))
        session_id = str(tr_row.get("session_id", ""))

        tr_start = _safe_float(tr_row.get("TrStartTime"), default=0.0)
        tr_end = _safe_float(tr_row.get("TrEndTime"), default=tr_start)
        stim_on = _safe_float(tr_row.get("StimOnsetTime"), default=0.0)
        stim_abs = _safe_float(tr_row.get("StimAbsTime"), default=float("nan"))
        if not pd.notna(stim_abs):
            stim_abs = _stimulus_absolute_timestamp(tr_start, stim_on, stim_mode)
        rw_start = _safe_float(tr_row.get("RWStartTime"), default=float("nan"))
        rw_sec = (rw_start - stim_abs) / 1000.0 if pd.notna(rw_start) and pd.notna(stim_abs) else float("nan")

        licks_ms = lick_map.get((session_id, tr_num), [])
        licks_sec: List[float] = []
        for lick_ms in licks_ms:
            try:
                value_sec = float(lick_ms) / 1000.0
            except Exception:
                continue
            if np.isfinite(value_sec):
                licks_sec.append(value_sec)

        outcome_raw = pd.to_numeric(tr_row.get("TrOutcome"), errors="coerce")
        outcome_code = int(outcome_raw) if not pd.isna(outcome_raw) else -1
        records.append(
            {
                "tr_num": tr_num,
                "outcome": outcome_code,
                "licks_sec": sorted(licks_sec),
                "reward_sec": float(rw_sec),
            }
        )

    if not records:
        if ax is not None:
            ax.text(0.5, 0.5, "No lick raster data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            print("No lick raster data.")
        return 0

    all_licks = [float(lick) for rec in records for lick in rec["licks_sec"]]
    total_points = len(all_licks)
    x_min, x_max = float(time_window_sec[0]), float(time_window_sec[1])
    if all_licks:
        lick_series = pd.Series(all_licks, dtype=float).replace([float("inf"), float("-inf")], pd.NA).dropna()
        if not lick_series.empty:
            q01 = float(lick_series.quantile(0.01))
            q99 = float(lick_series.quantile(0.99))
            if q01 < x_min - 0.5:
                x_min = q01 - 0.2
            if q99 > x_max + 0.5:
                x_max = q99 + 0.2
    if x_max <= x_min:
        x_max = x_min + 1.0

    def _reward_summary(rows: List[Dict[str, object]]) -> Optional[Tuple[float, float, float]]:
        rewards = [float(r["reward_sec"]) for r in rows if np.isfinite(float(r["reward_sec"]))]
        if not rewards:
            return None
        arr = np.array(rewards, dtype=float)
        q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
        return float(q1), float(med), float(q3)

    def _plot_reward_band(target_ax, rows: List[Dict[str, object]]) -> None:
        summary = _reward_summary(rows)
        if summary is None:
            return
        q1, med, q3 = summary
        target_ax.axvspan(q1, q3, color=reward_color, alpha=0.12, linewidth=0, zorder=0)
        target_ax.axvline(med, color=reward_color, linewidth=1.2, alpha=0.95, zorder=1)

    # Embedded mode: compact summary for dashboard panels.
    if ax is not None or standalone_style != "faceted":
        raster_ax = ax if ax is not None else plt.gca()
        for idx, rec in enumerate(records, start=1):
            outcome_code = int(rec["outcome"])
            color = outcome_colors.get(outcome_code, other_color)
            licks = [float(v) for v in rec["licks_sec"]]
            if licks:
                raster_ax.vlines(licks, idx - 0.35, idx + 0.35, color=color, linewidth=0.6, alpha=0.8)
        _plot_reward_band(raster_ax, records)
        raster_ax.axvline(0, color=stim_color, linestyle="--", linewidth=1.2, alpha=0.9)
        raster_ax.set_xlim([x_min, x_max])
        raster_ax.set_ylim([0.5, len(records) + 0.5])
        raster_ax.invert_yaxis()
        raster_ax.set_yticks([])
        raster_ax.set_ylabel("Trials")
        raster_ax.set_xlabel("Time from Stimulus Onset (s)")
        title_suffix = f" (first {len(records)} trials)" if len(records) < len(analysis.trials) else ""
        raster_ax.set_title(f"Stimulus-Aligned Licks{title_suffix}")
        raster_ax.grid(alpha=0.2, axis="x")
        if ax is None:
            plt.tight_layout()
            plt.show()
        return total_points

    # Standalone faceted mode.
    outcome_groups: List[Tuple[str, List[Dict[str, object]], str]] = []
    for outcome_code in outcome_order:
        rows = [rec for rec in records if int(rec["outcome"]) == outcome_code]
        label = outcome_labels[outcome_code]
        color = outcome_colors[outcome_code]
        outcome_groups.append((label, rows, color))

    other_codes = sorted({int(rec["outcome"]) for rec in records if int(rec["outcome"]) not in outcome_order})
    if include_other_outcomes and other_codes:
        other_rows = [rec for rec in records if int(rec["outcome"]) in other_codes]
        outcome_groups.append((f"Other ({', '.join(str(c) for c in other_codes)})", other_rows, other_color))

    n_panels = len(outcome_groups)
    fig_height = 2.2 + (1.35 * n_panels)
    fig = plt.figure(figsize=(12, fig_height))
    gs = fig.add_gridspec(1 + n_panels, 1, height_ratios=[1.2] + [1.0] * n_panels, hspace=0.10)

    psth_ax = fig.add_subplot(gs[0, 0])
    raster_axes = [fig.add_subplot(gs[idx + 1, 0], sharex=psth_ax) for idx in range(n_panels)]

    # PSTH by outcome group.
    edges = np.arange(x_min, x_max + bin_width_sec, bin_width_sec)
    if len(edges) < 2:
        edges = np.array([x_min, x_max + max(0.1, bin_width_sec)])
    centers = edges[:-1] + (bin_width_sec / 2.0)

    legend_handles: List[Line2D] = []
    for label, rows, color in outcome_groups:
        trial_count = len(rows)
        if trial_count <= 0:
            continue
        licks = [float(v) for rec in rows for v in rec["licks_sec"]]
        if not licks:
            continue
        counts, _ = np.histogram(np.array(licks, dtype=float), bins=edges)
        rate_hz = counts.astype(float) / (trial_count * bin_width_sec)
        if len(rate_hz) > 2:
            rate_hz = pd.Series(rate_hz, dtype=float).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
        psth_ax.plot(centers, rate_hz, color=color, linewidth=1.9, alpha=0.95)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2.0, label=f"{label}"))

    psth_ax.axvline(0, color=stim_color, linestyle="--", linewidth=1.2, alpha=0.9)
    psth_ax.set_ylabel("Lick Rate (Hz)")
    psth_ax.set_title("Stimulus-Aligned Lick PSTH + Outcome Faceted Raster")
    psth_ax.grid(alpha=0.2, axis="x")

    # Raster panels.
    for raster_ax, (label, rows, color) in zip(raster_axes, outcome_groups):
        raster_ax.axvline(0, color=stim_color, linestyle="--", linewidth=1.1, alpha=0.9, zorder=0)
        _plot_reward_band(raster_ax, rows)

        if rows:
            for idx, rec in enumerate(rows, start=1):
                licks = [float(v) for v in rec["licks_sec"]]
                if licks:
                    raster_ax.vlines(licks, idx - 0.35, idx + 0.35, color=color, linewidth=0.65, alpha=0.85, zorder=2)
            raster_ax.set_ylim([0.5, len(rows) + 0.5])
            raster_ax.invert_yaxis()
        else:
            raster_ax.text(0.5, 0.5, "No trials", ha="center", va="center", transform=raster_ax.transAxes, fontsize=9)
            raster_ax.set_ylim([0.0, 1.0])

        raster_ax.set_yticks([])
        raster_ax.set_ylabel(f"{label}\n(n={len(rows)})", fontsize=9)
        raster_ax.grid(alpha=0.2, axis="x")

    raster_axes[-1].set_xlabel("Time from Stimulus Onset (s)")

    # Figure-level legend.
    legend_handles.append(
        Line2D([0], [0], color=stim_color, linestyle="--", linewidth=1.2, label="Stimulus Onset (t=0)")
    )
    legend_handles.append(Line2D([0], [0], color=reward_color, linewidth=1.2, label="Reward Median"))
    legend_handles.append(Patch(facecolor=reward_color, alpha=0.12, edgecolor="none", label="Reward IQR"))
    fig.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)

    for axis in [psth_ax] + raster_axes:
        axis.set_xlim([x_min, x_max])

    fig.subplots_adjust(top=0.94, right=0.82, hspace=0.12)
    plt.show()
    return total_points


def plot_dprime_and_rates(analysis: AnalysisData, window_size: int = 10) -> None:
    import matplotlib.pyplot as plt

    trials = _ordered_trials(analysis.trials)
    if trials.empty or "TrType" not in trials.columns or "TrOutcome" not in trials.columns:
        print("No d' data available.")
        return

    tr_type = pd.to_numeric(trials["TrType"], errors="coerce").fillna(-1).astype(int)
    outcome = pd.to_numeric(trials["TrOutcome"], errors="coerce").fillna(-1).astype(int)

    total_trial_num = len(trials)
    if total_trial_num < window_size:
        window_size = max(1, total_trial_num)

    dprime_list: List[float] = []
    hit_rate_list: List[float] = []
    fa_rate_list: List[float] = []
    trial_axis: List[int] = []

    for i in range(total_trial_num - window_size + 1):
        block_type = tr_type.iloc[i : i + window_size]
        block_outcome = outcome.iloc[i : i + window_size]

        hits = int(((block_type == 1) & (block_outcome == 1)).sum())
        misses = int(((block_type == 1) & (block_outcome == 4)).sum())
        cr = int(((block_type == 0) & (block_outcome == 2)).sum())
        fa = int(((block_type == 0) & (block_outcome == 3)).sum())

        dprime, hr, far = calculate_dprime(hits, misses, fa, cr)
        dprime_list.append(dprime)
        hit_rate_list.append(hr)
        fa_rate_list.append(far)
        trial_axis.append(i + window_size)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    line1 = ax1.plot(trial_axis, dprime_list, color="#5e5656", label="Rolling d'", linewidth=2.2)
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("d'")

    ax2 = ax1.twinx()
    line2 = ax2.plot(trial_axis, hit_rate_list, color="#1f77b4", label="Rolling Hit Rate", linewidth=2.2)
    line3 = ax2.plot(trial_axis, fa_rate_list, color="#ea6161", label="Rolling FA Rate", linewidth=2.2)
    ax2.set_ylabel("Rates")
    rate_values = pd.Series(hit_rate_list + fa_rate_list, dtype=float)
    rate_values = pd.to_numeric(rate_values, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    if not rate_values.empty:
        y_min = min(0.0, float(rate_values.min()))
        y_max = max(1.0, float(rate_values.max()))
        pad = 0.05 * max(1.0, y_max - y_min)
        ax2.set_ylim([y_min - pad, y_max + pad])
    else:
        ax2.set_ylim([-0.05, 1.05])

    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")
    plt.title(f"Rolling d', Hit Rate, and False Alarm Rate (window={window_size})")
    plt.tight_layout()
    plt.show()


def plot_performance_bars(analysis: AnalysisData) -> None:
    import matplotlib.pyplot as plt

    summary = session_summary(analysis.trials)
    labels = ["Hit", "Miss", "False Alarm", "Correct Rejection"]
    rates = [summary["HitPct"], summary["MissPct"], summary["FalseAlarmPct"], summary["CorrectRejectPct"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, rates, color=["#1f77b4", "#3f4240", "#ea6161", "#62d18d"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Hit/Miss/False Alarm/Correct Rejection Rates")
    ax.set_ylim([0, max(100.0, max(rates) + 10)])

    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def run_one_day_analysis_suite(analysis: AnalysisData, window_size: int = 10) -> None:
    print_session_summary(analysis.trials)
    plot_first_lick_histogram(analysis)
    plot_lick_raster(analysis)
    plot_dprime_and_rates(analysis, window_size=window_size)
    plot_performance_bars(analysis)


def _plot_outcome_panel(trials: pd.DataFrame, mode: str, ax) -> None:
    valid_outcomes = ["Hit", "Miss", "False Alarm", "Correct Reject"]
    colors = {
        "Hit": "#2a9d8f",
        "Miss": "#9aa0a6",
        "False Alarm": "#e76f51",
        "Correct Reject": "#264653",
    }

    if "outcome_label" not in trials.columns:
        ax.text(0.5, 0.5, "No outcome labels", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    filtered_trials = trials[trials["outcome_label"].isin(valid_outcomes)].copy()
    if filtered_trials.empty:
        ax.text(0.5, 0.5, "No outcome data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    if mode == "week" and "date" in filtered_trials.columns:
        date_series = pd.to_datetime(filtered_trials["date"], errors="coerce").dt.normalize()
        filtered_trials = filtered_trials.assign(_plot_date=date_series).dropna(subset=["_plot_date"])
        if filtered_trials.empty:
            ax.text(0.5, 0.5, "No dated outcome data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        by_date = (
            filtered_trials.groupby(["_plot_date", "outcome_label"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=valid_outcomes, fill_value=0)
            .sort_index()
        )

        anchor_date = by_date.index.max()
        full_index = pd.date_range(anchor_date - pd.Timedelta(days=6), anchor_date, freq="D")
        by_date = by_date.reindex(full_index, fill_value=0)
        x_labels = [dt.strftime("%Y-%m-%d") for dt in by_date.index]
        for outcome in valid_outcomes:
            ax.plot(
                x_labels,
                by_date[outcome].values,
                marker="o",
                linewidth=2.0,
                color=colors[outcome],
                label=outcome,
            )
        ax.set_title("Outcome Change Over 7-Day Window")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        return

    counts = filtered_trials["outcome_label"].value_counts().reindex(valid_outcomes, fill_value=0)
    ax.bar(counts.index, counts.values, color=[colors[label] for label in counts.index])
    ax.set_title("Outcome Counts")
    ax.tick_params(axis="x", rotation=20)


def plot_scope_overview(analysis: AnalysisData, rolling_window: int = 20) -> None:
    import matplotlib.pyplot as plt

    trials = _ordered_trials(analysis.trials)
    if trials.empty:
        print("No trial data available for plotting.")
        return

    if "TrType" in trials.columns and "TrOutcome" in trials.columns:
        tr_type = pd.to_numeric(trials["TrType"], errors="coerce").fillna(-1).astype(int)
        outcome = pd.to_numeric(trials["TrOutcome"], errors="coerce").fillna(-1).astype(int)
        go_trials = tr_type == 1
        nogo_trials = tr_type == 0
        hit_events = (tr_type == 1) & (outcome == 1)
        fa_events = (tr_type == 0) & (outcome == 3)
        hit_denominator = go_trials.rolling(rolling_window, min_periods=1).sum().replace(0, float("nan"))
        fa_denominator = nogo_trials.rolling(rolling_window, min_periods=1).sum().replace(0, float("nan"))
        hit_rate = (hit_events.rolling(rolling_window, min_periods=1).sum() / hit_denominator).astype(float)
        fa_rate = (fa_events.rolling(rolling_window, min_periods=1).sum() / fa_denominator).astype(float)
    else:
        hit_rate = pd.Series(index=trials.index, dtype=float)
        fa_rate = pd.Series(index=trials.index, dtype=float)

    first_lick_sec = pd.Series(dtype=float)
    if not analysis.lick_df.empty and "lick_times_rel_stim_ms" in analysis.lick_df.columns:
        first_lick_ms = analysis.lick_df["lick_times_rel_stim_ms"].apply(_first_nonnegative)
        first_lick_sec = (first_lick_ms / 1000.0).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(range(1, len(trials) + 1), hit_rate, label="Hit Rate", color="#1f77b4")
    axes[0, 0].plot(range(1, len(trials) + 1), fa_rate, label="FA Rate", color="#ea6161")
    axes[0, 0].set_title(f"Rolling Rates (window={rolling_window})")
    axes[0, 0].set_xlabel("Trial")
    axes[0, 0].set_ylabel("Rate")
    rolling_values = pd.concat([hit_rate, fa_rate], ignore_index=True)
    rolling_values = pd.to_numeric(rolling_values, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    if not rolling_values.empty:
        y_min = min(0.0, float(rolling_values.min()))
        y_max = max(1.0, float(rolling_values.max()))
        pad = 0.05 * max(1.0, y_max - y_min)
        axes[0, 0].set_ylim(y_min - pad, y_max + pad)
    else:
        axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(alpha=0.3)

    _plot_outcome_panel(trials, analysis.extraction.mode, axes[0, 1])

    plot_lick_raster(analysis, ax=axes[1, 0], max_trials=120)

    if not first_lick_sec.empty:
        axes[1, 1].hist(first_lick_sec, bins=30, color="#457b9d", alpha=0.85)
        axes[1, 1].set_title("First Lick Latency (>= 0s)")
        axes[1, 1].set_xlabel("Seconds from stimulus")
        axes[1, 1].set_ylabel("Count")
    else:
        axes[1, 1].text(0.5, 0.5, "No first-lick data", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_axis_off()

    fig.suptitle(
        f"{analysis.extraction.mode.upper()} scope overview | "
        f"folders={len(analysis.extraction.selected_folders)} | trials={len(analysis.trials)}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def _scope_row(label: str, analysis: AnalysisData) -> Dict[str, object]:
    folders = [folder.name for folder in analysis.extraction.selected_folders]
    day_min = ""
    day_max = ""
    if "date" in analysis.trials.columns and not analysis.trials.empty:
        non_null_dates = analysis.trials["date"].dropna()
        if not non_null_dates.empty:
            day_min = str(non_null_dates.min())
            day_max = str(non_null_dates.max())
    return {
        "scope": label,
        "folder_count": len(folders),
        "folders": ", ".join(folders),
        "trial_rows": len(analysis.trials),
        "lick_rows": len(analysis.lick_df),
        "date_min": day_min,
        "date_max": day_max,
    }


def _display_heading(text: str, level: int = 3) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(f"{'#' * max(1, level)} {text}"))
    except Exception:
        print(text)


def display_all_scope_results(context: AutoExtractionContext, rolling_window: int = 20) -> pd.DataFrame:
    rows = [
        _scope_row("day", context.day),
        _scope_row("week", context.week),
        _scope_row("all", context.all_time),
    ]
    summary_df = pd.DataFrame(rows)

    _display_heading("Extraction Summary", level=3)
    try:
        from IPython.display import display

        display(summary_df)
    except Exception:
        print(summary_df.to_string(index=False))

    for label, analysis in [("DAY", context.day), ("WEEK", context.week), ("ALL", context.all_time)]:
        _display_heading(f"{label} Results", level=3)
        print("folders:", ", ".join(folder.name for folder in analysis.extraction.selected_folders))
        plot_scope_overview(analysis, rolling_window=rolling_window)

    return summary_df


def plot_auto_scope(context: AutoExtractionContext, scope: str = "auto", rolling_window: int = 20) -> str:
    normalized_scope = scope.strip().lower()
    if normalized_scope == "auto":
        normalized_scope = context.resolved_scope if context.resolved_scope in {"day", "week", "all"} else "week"
    if normalized_scope not in {"day", "week", "all"}:
        raise ValueError("Scope must be one of: auto, day, week, all.")
    by_scope = {
        "day": context.day,
        "week": context.week,
        "all": context.all_time,
    }
    plot_scope_overview(by_scope[normalized_scope], rolling_window=rolling_window)
    return normalized_scope


def show_extraction_widget(default_folder: Union[str, Path] = "Jeremy"):
    import ipywidgets as widgets
    from IPython.display import display

    most_recent_token = "__MOST_RECENT__"

    folder_input = widgets.Text(
        value=str(Path(default_folder).expanduser()),
        description="Folder:",
        placeholder="/path/to/Jeremy",
        layout=widgets.Layout(width="100%"),
    )
    default_scope_input = widgets.ToggleButtons(
        options=[
            ("Auto", "auto"),
            ("Day", "day"),
            ("Week", "week"),
            ("All", "all"),
        ],
        description="Default:",
    )
    day_input = widgets.Dropdown(
        options=[("Most Recent", most_recent_token)],
        description="Anchor Day:",
        disabled=False,
        layout=widgets.Layout(width="100%"),
    )
    scan_button = widgets.Button(description="Scan Folders", button_style="info")
    extract_button = widgets.Button(description="Load + Plot", button_style="success")
    apply_scope_button = widgets.Button(description="Apply Default Scope")
    close_button = widgets.Button(description="Close")
    auto_plot_input = widgets.Checkbox(value=True, description="Show day/week/all plots")
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
            default_scope_input,
            day_input,
            auto_plot_input,
            widgets.HBox([scan_button, extract_button, apply_scope_button, close_button]),
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
            if not folders:
                day_input.options = [("Most Recent", most_recent_token)]
                status.value = "<span style='color:#a33'>No YYMMDD folders found.</span>"
                return
            options = [(f"Most Recent ({folders[-1][0].name})", most_recent_token)]
            options.extend((folder.name, folder.name) for folder, _ in folders)
            day_input.options = options
            day_input.value = most_recent_token
            status.value = (
                f"<span style='color:#2f6f37'>Found {len(folders)} folders. "
                f"Most recent: <b>{folders[-1][0].name}</b>.</span>"
            )
        except Exception as exc:
            day_input.options = [("Most Recent", most_recent_token)]
            status.value = f"<span style='color:#a33'>{exc}</span>"

    def selected_day_value() -> Optional[str]:
        return None if day_input.value == most_recent_token else str(day_input.value)

    def handle_launch(_):
        panel.layout.display = "none" if panel.layout.display == "flex" else "flex"
        if panel.layout.display == "flex":
            refresh_day_options()

    def handle_close(_):
        panel.layout.display = "none"

    def handle_scan(_):
        refresh_day_options()

    def handle_extract(_):
        with output:
            output.clear_output()
            try:
                context = load_auto_context(
                    root_folder=folder_input.value,
                    selected_day=selected_day_value(),
                    default_scope=default_scope_input.value,
                )
                print(context.summary_text())
                print()
                print("Notebook variables updated:")
                print("  analysis_context")
                print("  trials_day, lick_df_day, stimulus_data_day, header_data_day")
                print("  trials_week, lick_df_week, stimulus_data_week, header_data_week")
                print("  trials_all, lick_df_all, stimulus_data_all, header_data_all")
                print("Default aliases:")
                print("  trials, lick_df, stimulus_data, header_data")
                print(f"Active alias scope: {context.resolved_scope}")
                if auto_plot_input.value:
                    print()
                    print("Rendering day/week/all plots...")
                    summary_df = display_all_scope_results(context)
                    print(f"Rendered all scopes ({len(summary_df)} rows in summary table).")
            except Exception as exc:
                print(f"Extraction failed: {exc}")

    def handle_apply_scope(_):
        with output:
            try:
                if LAST_AUTO_CONTEXT is None:
                    print("No loaded context yet. Click 'Load Folder' first.")
                    return
                resolved = apply_scope_aliases(LAST_AUTO_CONTEXT, scope=default_scope_input.value)
                LAST_AUTO_CONTEXT.resolved_scope = resolved
                print(f"Applied default aliases to scope: {resolved}")
            except Exception as exc:
                print(f"Scope update failed: {exc}")

    launch_button.on_click(handle_launch)
    close_button.on_click(handle_close)
    scan_button.on_click(handle_scan)
    extract_button.on_click(handle_extract)
    apply_scope_button.on_click(handle_apply_scope)

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
        help="Specific day folder (YYMMDD). Used with --mode day, or as fallback anchor for --mode week.",
    )
    parser.add_argument(
        "--anchor-day",
        default=None,
        help="Anchor day folder (YYMMDD) for --mode week. Defaults to most recent day.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    # Jupyter/Colab inject kernel args (for example, "-f <kernel.json>").
    # Ignore unknown args so this entrypoint is notebook-safe.
    args, _unknown = parser.parse_known_args()
    day_folder = args.day if args.mode == "day" else None
    anchor_day = args.anchor_day if args.anchor_day is not None else (args.day if args.mode == "week" else None)
    result = extract_behavior_data(
        args.folder,
        mode=args.mode,
        day_folder=day_folder,
        anchor_day=anchor_day,
    )
    print(result.summary_text())
    try:
        analysis = build_analysis_data(result)
        plot_scope_overview(analysis)
    except Exception:
        # Keep CLI resilient even if plotting backend is unavailable.
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
