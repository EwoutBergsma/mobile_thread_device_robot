import os
import re
from glob import glob
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, fields
from typing import List, Dict, Optional, Any
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Length (in bytes) of the ping reply ICMPv6 message in the log.
PING_REPLY_LEN = 56

# Base directory: the directory where this script resides
SCRIPT_DIR = Path(__file__).resolve().parent

# Data directory (relative to this file)
DATA_DIR = SCRIPT_DIR / "data"

# Regex for RTT lines like:
# [2025-12-04T17:52:44.956] 1 packets transmitted, 1 packets received.
# Packet loss = 0.0%. Round-trip min/avg/max = 239/239.000/239 ms.
rtt_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]               # [2025-12-04T17:54:36.720]
    .*?Round-trip\ min/avg/max\ =\   # Round-trip min/avg/max =
    (?P<min>\d+(?:\.\d+)?)/          # min
    (?P<avg>\d+(?:\.\d+)?)/          # avg
    (?P<max>\d+(?:\.\d+)?)\ ms\.?    # max ms.
    """,
    re.VERBOSE
)

# Regex for packet summary lines with packet loss (0.0%, 10.0%, 100.0%, etc.).
loss_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]\s+                  # timestamp in brackets
    (?P<tx>\d+)\s+packets\s+transmitted,\s+
    (?P<rx>\d+)\s+packets\s+received\.\s+
    Packet\ loss\s*=\s*
    (?P<loss>\d+(?:\.\d+)?)%              # e.g. 0.0, 10.0, 100.0
    """,
    re.VERBOSE
)

# Regex for parent lines, using RLOC16 as parent IDs.
parent_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]      # timestamp in brackets
    .*?Rloc:\s*
    (?P<rloc>[0-9A-Fa-f]+)  # hex-only RLOC16 value
    """,
    re.VERBOSE
)

# Regex for OT state lines.
state_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]              # timestamp
    .*?\b(?P<state>disabled|detached|child|router|leader)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Regex for *ping reply* RSS lines (MeshForwarder Received IPv6 ... msg).
rss_re = re.compile(
    r"""
    ^\[(?P<ts>[^\]]+)\]\s+                      # timestamp at start of line
    .*?MeshForwarder-:\s+Received\ IPv6\s+\S+\s+msg,\s+
    len:(?P<len>\d+),.*?                        # capture len:NN
    rss:(?P<rss>-?\d+(?:\.\d+)?)                # capture rss:-90.0
    """,
    re.VERBOSE
)


# Helper to parse timestamps like 2025-12-04T17:52:42.396
def parse_timestamp(ts_str: str) -> datetime:
    try:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")


def normalize_rloc16(rloc: str) -> str:
    """
    Normalize an RLOC16 string to a 4-digit lowercase hex representation.

    Examples:
        "c00"  -> "0c00"
        "0C00" -> "0c00"
        "0000" -> "0000"
    """
    r = rloc.strip().lower()
    if r.startswith("0x"):
        r = r[2:]
    try:
        value = int(r, 16)
    except ValueError:
        # If not valid hex, just return the stripped original.
        return rloc.strip()
    return f"{value:04x}"


@dataclass
class LogMetrics:
    rtt_timestamps: List[datetime]
    rtt_avgs_ms: List[float]
    loss_timestamps: List[datetime]
    parent_timestamps: List[datetime]
    parent_ids: List[str]
    rss_timestamps: List[datetime]
    rss_values: List[float]
    state_timestamps: List[datetime]
    states: List[str]
    eff_parent_timestamps: List[datetime]
    eff_parents: List[str]
    # Totals for PDR over the entire file.
    total_tx: int
    total_rx: int


def compute_effective_parent(
    state: Optional[str],
    parent: Optional[str],
) -> Optional[str]:
    """
    Determine effective parent for the *current* state/parent pair.
    """
    def norm_state(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        return s.strip().lower()

    st = norm_state(state)

    # Explicit "no parent" states.
    if st in ("detached", "disabled", "blank"):
        return "No Parent"

    # No explicit parent information.
    if parent is None:
        return None

    p = parent.strip()
    if not p or p.lower() in ("none", "nan", "no parent"):
        return "No Parent"

    # Default: a valid parent, normalize RLOC16.
    return normalize_rloc16(p)


def build_parent_timeline(metrics: LogMetrics) -> None:
    """
    Build a time series of "effective parent" with forward-filled state/parent.
    """
    parents_at_ts: Dict[datetime, List[str]] = defaultdict(list)
    for ts, pid in zip(metrics.parent_timestamps, metrics.parent_ids):
        parents_at_ts[ts].append(pid)

    states_at_ts: Dict[datetime, List[str]] = defaultdict(list)
    for ts, st in zip(metrics.state_timestamps, metrics.states):
        states_at_ts[ts].append(st)

    all_ts_set = set()
    all_ts_set.update(metrics.rtt_timestamps)
    all_ts_set.update(metrics.loss_timestamps)
    all_ts_set.update(metrics.parent_timestamps)
    all_ts_set.update(metrics.state_timestamps)
    all_ts_set.update(metrics.rss_timestamps)

    if not all_ts_set:
        metrics.eff_parent_timestamps = []
        metrics.eff_parents = []
        return

    all_ts = sorted(all_ts_set)

    current_state: Optional[str] = None
    current_parent: Optional[str] = None

    eff_ts: List[datetime] = []
    eff_parents: List[str] = []

    for ts in all_ts:
        if ts in states_at_ts:
            current_state = states_at_ts[ts][-1].strip().lower()
        if ts in parents_at_ts:
            current_parent = parents_at_ts[ts][-1].strip()

        eff_parent = compute_effective_parent(current_state, current_parent)
        if eff_parent is None:
            continue

        eff_ts.append(ts)
        eff_parents.append(eff_parent)

    metrics.eff_parent_timestamps = eff_ts
    metrics.eff_parents = eff_parents


def parse_log_file(log_path: str) -> LogMetrics:
    """
    Read a log file and extract metrics.
    """
    print(f"\n[PROCESS] Starting file: {log_path}")

    metrics = LogMetrics(
        rtt_timestamps=[],
        rtt_avgs_ms=[],
        loss_timestamps=[],
        parent_timestamps=[],
        parent_ids=[],
        rss_timestamps=[],
        rss_values=[],
        state_timestamps=[],
        states=[],
        eff_parent_timestamps=[],
        eff_parents=[],
        total_tx=0,
        total_rx=0,
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line_stripped = line.rstrip("\n")

            # --- Packet loss / summary detection (any percentage). ---
            m_loss = loss_re.search(line_stripped)
            if m_loss:
                ts_loss_str = m_loss.group("ts")
                ts_loss = parse_timestamp(ts_loss_str)

                tx = int(m_loss.group("tx"))
                rx = int(m_loss.group("rx"))
                loss_pct = float(m_loss.group("loss"))

                # Accumulate totals for overall PDR.
                metrics.total_tx += tx
                metrics.total_rx += rx

                # Keep timestamp for non-zero loss events.
                if loss_pct > 0.0:
                    print(f"  [LOSS] Packet loss {loss_pct}% at {ts_loss_str}")
                    metrics.loss_timestamps.append(ts_loss)

            # --- RTT detection (only lines with Round-trip stats). ---
            m_rtt = rtt_re.search(line_stripped)
            if m_rtt:
                print(f"  [USE RTT] Line {line_no}: {line_stripped}")
                ts_str = m_rtt.group("ts")
                avg_str = m_rtt.group("avg")
                ts = parse_timestamp(ts_str)
                avg_ms = float(avg_str)
                metrics.rtt_timestamps.append(ts)
                metrics.rtt_avgs_ms.append(avg_ms)

            # --- Parent detection (RLOC16 from "Rloc:" lines). ---
            m_parent = parent_re.search(line_stripped)
            if m_parent:
                ts_str = m_parent.group("ts")
                raw_rloc16 = m_parent.group("rloc")
                rloc16 = normalize_rloc16(raw_rloc16)
                ts = parse_timestamp(ts_str)
                print(f"  [USE PARENT] Line {line_no}: {line_stripped}")
                print(f"               -> Parent RLOC16: {rloc16}")
                metrics.parent_timestamps.append(ts)
                metrics.parent_ids.append(rloc16)

            # --- State detection. ---
            m_state = state_re.search(line_stripped)
            if m_state:
                ts_str = m_state.group("ts")
                state_str = m_state.group("state")
                ts = parse_timestamp(ts_str)
                state_norm = state_str.strip().lower()
                print(f"  [STATE] {state_norm} at {ts_str}")
                metrics.state_timestamps.append(ts)
                metrics.states.append(state_norm)

            # --- RSS detection for ping replies. ---
            m_rss = rss_re.search(line_stripped)
            if m_rss:
                length = int(m_rss.group("len"))
                if length == PING_REPLY_LEN:
                    ts_str = m_rss.group("ts")
                    rss_str = m_rss.group("rss")
                    ts = parse_timestamp(ts_str)
                    rss_val = float(rss_str)
                    print(f"  [PING RSS] len={length} -> {rss_val} dBm at {ts_str}")
                    metrics.rss_timestamps.append(ts)
                    metrics.rss_values.append(rss_val)

    build_parent_timeline(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Generic visualization
# ---------------------------------------------------------------------------

def _format_time_axis(ax):
    """Helper to format datetime x-axis consistently."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, which="both", linestyle="--", alpha=0.3)


def _discover_time_series(metrics: LogMetrics) -> List[Dict[str, Any]]:
    """
    Inspect LogMetrics and automatically discover time series.

    Logic:
      - Any field ending with `_timestamps` and containing a non-empty list
        is treated as a time axis.
      - For its 'root' (everything before `_timestamps`), find list fields:
          * whose name starts with root + '_'
          * or whose name equals root
          * or whose name equals root + 's' (simple plural)
        and that have the same length as the timestamps list.
      - If at least one matching value list exists, each becomes a value
        series on that time axis; otherwise the root becomes an event-only
        series.
    """
    series: List[Dict[str, Any]] = []

    # Map field name -> value.
    metric_values: Dict[str, Any] = {
        f.name: getattr(metrics, f.name) for f in fields(LogMetrics)
    }

    for fname, fval in metric_values.items():
        if not fname.endswith("_timestamps"):
            continue
        if not isinstance(fval, list) or not fval:
            continue

        timestamps = fval

        # Root is everything before the final "_timestamps" token.
        tokens = fname.split("_")[:-1]  # drop 'timestamps'
        root = "_".join(tokens) if tokens else fname[:-len("_timestamps")]
        root_label = root.replace("_", " ").title() if root else fname

        # Candidate value names to match against.
        candidate_prefix = root + "_"
        candidate_exact = root
        candidate_plural = root + "s"

        candidates: List[Dict[str, Any]] = []
        for other_name, other_val in metric_values.items():
            if other_name == fname:
                continue
            if not isinstance(other_val, list):
                continue
            if len(other_val) != len(timestamps):
                continue

            name_matches = (
                other_name.startswith(candidate_prefix)
                or other_name == candidate_exact
                or other_name == candidate_plural
            )
            if not name_matches:
                continue

            candidates.append({"name": other_name, "values": other_val})

        if candidates:
            for c in candidates:
                values = c["values"]
                # Determine if categorical (string) or numeric.
                non_none = next((v for v in values if v is not None), None)
                categorical = isinstance(non_none, str) if non_none is not None else False

                # Build a readable label: "Root – Suffix" where possible.
                val_name = c["name"]
                if val_name.startswith(candidate_prefix):
                    suffix = val_name[len(candidate_prefix):]
                elif val_name in (candidate_exact, candidate_plural):
                    suffix = val_name  # keep as-is
                else:
                    suffix = val_name
                pretty_suffix = suffix.replace("_", " ").title()

                if pretty_suffix and pretty_suffix.lower() != root_label.lower():
                    label = f"{root_label} – {pretty_suffix}"
                else:
                    label = root_label

                series.append(
                    {
                        "timestamps": timestamps,
                        "values": values,
                        "label": label,
                        "categorical": categorical,
                    }
                )
        else:
            # Event-only series (no associated value list).
            label = f"{root_label} Events"
            series.append(
                {
                    "timestamps": timestamps,
                    "values": None,
                    "label": label,
                    "categorical": False,
                }
            )

    return series


def visualize_metrics(metrics: LogMetrics, title: str = "OpenThread log metrics") -> None:
    """
    Generic visualization of all parsed metrics.

    Every discovered time series (or event series) gets its own subplot:
      - numeric values: scatter only (no line)
      - categorical values: stepped series with discrete y-ticks
      - events (timestamps only): vertical lines vs. time
    """
    series_list = _discover_time_series(metrics)

    if not series_list:
        print("[VIS] No metrics to visualize.")
        return

    nrows = len(series_list)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(12, 3 * nrows))
    if nrows == 1:
        axes = [axes]

    # Overall PDR summary for the figure title.
    if metrics.total_tx > 0:
        pdr = 100.0 * metrics.total_rx / metrics.total_tx
        sup_title = f"{title} – PDR: {metrics.total_rx}/{metrics.total_tx} ({pdr:.2f}%)"
    else:
        sup_title = title
    fig.suptitle(sup_title, fontsize=14)

    for ax, spec in zip(axes, series_list):
        ts = spec["timestamps"]
        values = spec["values"]
        label = spec["label"]
        categorical = spec["categorical"]

        if values is None:
            # Event-only series: vertical lines.
            for ts_i in ts:
                ax.axvline(ts_i, linestyle="--", alpha=0.5)
            ax.set_title(label)
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            if categorical:
                # Map categories to integer y values.
                categories = sorted({v for v in values if v is not None})
                y_map = {cat: idx for idx, cat in enumerate(categories)}
                y_vals = [y_map.get(v, None) for v in values]

                x_plot = [t for t, y in zip(ts, y_vals) if y is not None]
                y_plot = [y for y in y_vals if y is not None]

                if x_plot:
                    ax.step(x_plot, y_plot, where="post")
                ax.set_yticks(range(len(categories)))
                ax.set_yticklabels(categories)
                ax.set_ylabel(label)
                ax.set_title(label)
            else:
                # Numeric time series: scatter only, small markers.
                ax.scatter(ts, values, s=8)  # small points, no connecting line
                ax.set_ylabel(label)
                ax.set_title(label)

        _format_time_axis(ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point: glob log files, generic visualization, no boxplots
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir_path = DATA_DIR

    # Search for all .log files under data_dir_path (relative to this script)
    pattern = str(DATA_DIR / "**" / "*.log")
    all_candidates = glob(pattern, recursive=True)

    log_files: List[str] = []
    for path_str in all_candidates:
        p = Path(path_str)
        try:
            rel_parts = p.relative_to(data_dir_path).parts
        except ValueError:
            # Should not happen, but be defensive.
            continue

        dir_parts = rel_parts[:-1]
        if any(part.startswith(".") for part in dir_parts):
            # Skip dot-directories under data/
            continue

        log_files.append(path_str)

    if not log_files:
        print(f"[INFO] No .log files found in {data_dir_path}")
        return

    print(f"[INFO] Found {len(log_files)} .log file(s) in {data_dir_path} (excluding dot-directories):")
    for lf in log_files:
        print(f"  - {lf}")

    # Process each log file: parse, summarize, visualize
    for log_path in log_files:
        metrics = parse_log_file(log_path)
        rel_label = str(Path(log_path).relative_to(data_dir_path))

        print("\n[SUMMARY]")
        print(f"  File:            {rel_label}")
        print(f"  RTT samples:     {len(metrics.rtt_timestamps)}")
        print(f"  RSS samples:     {len(metrics.rss_timestamps)}")
        print(f"  State changes:   {len(metrics.state_timestamps)}")
        print(f"  Parent changes:  {len(metrics.parent_timestamps)}")
        print(f"  Eff parents:     {len(metrics.eff_parent_timestamps)}")
        print(f"  Loss events:     {len(metrics.loss_timestamps)}")
