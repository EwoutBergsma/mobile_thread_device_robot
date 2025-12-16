# openthread_log_visualizer.py
from __future__ import annotations

import os
from datetime import date, datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

from openthread_log_parser import DATA_DIR as PARSER_DATA_DIR
from openthread_log_parser import LogMetrics, parse_log_file


# -----------------------------------------------------------------------------
# Output / plotting configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(PARSER_DATA_DIR) if PARSER_DATA_DIR is not None else (SCRIPT_DIR / "data")

GRAPHS_DIR = SCRIPT_DIR / "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

RSS_YLIM: Optional[Tuple[float, float]] = (-120, 0)  # dBm

# Horizontal threshold line on RSS subplot
PPS_RSS_THRESHOLD_DBM: float = -65.0

# Packet-loss "rug" settings (fraction of y-range at the top)
LOSS_RUG_FRACTION: float = 1  # fraction of y-range at top

# Legend styling
LEGEND_FRAME_ALPHA: float = 1.0
LEGEND_ZORDER: int = 50

# X-axis (relative time) formatting
TIME_TICK_INTERVAL_MINUTES: int = 10

# Parent plot styling
NO_PARENT_COLOR: str = "0.35"  # dark grey (0=black, 1=white)

# Trim settings
TRIM_WINDOW_SECONDS: float = 2 * 60 * 60  # exactly 2 hours


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------

def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    n = len(values)
    mu = sum(values) / n
    if n > 1:
        var = sum((v - mu) ** 2 for v in values) / n  # population variance
        sigma = var ** 0.5
    else:
        sigma = 0.0
    return mu, sigma


def _overall_pdr(metrics: LogMetrics) -> Optional[float]:
    if getattr(metrics, "total_ping_tx_packets", 0) > 0:
        return 100.0 * metrics.total_ping_rx_packets / metrics.total_ping_tx_packets
    return None


def _add_legend_on_top(ax) -> None:
    """Make legend opaque (no transparency) and ensure it draws above plot artists."""
    leg = ax.legend(framealpha=LEGEND_FRAME_ALPHA)
    if leg is not None:
        leg.set_zorder(LEGEND_ZORDER)
        frame = leg.get_frame()
        if frame is not None:
            frame.set_alpha(LEGEND_FRAME_ALPHA)


def _remove_x_whitespace(axes) -> None:
    for ax in axes:
        ax.margins(x=0)
        if hasattr(ax, "set_xmargin"):
            ax.set_xmargin(0)


# -----------------------------------------------------------------------------
# Relative-time axis formatting (elapsed HH:MM)
# -----------------------------------------------------------------------------

def _format_elapsed_hhmm(x_seconds: float, _pos: int) -> str:
    if x_seconds is None:
        return ""
    if x_seconds < 0:
        x_seconds = 0.0
    total_seconds = int(round(x_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def _configure_elapsed_time_axis_hhmm(axes, *, interval_minutes: int = TIME_TICK_INTERVAL_MINUTES) -> None:
    tick_step_seconds = interval_minutes * 60
    locator = MultipleLocator(tick_step_seconds)
    formatter = FuncFormatter(_format_elapsed_hhmm)

    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.offsetText.set_visible(False)
        for lbl in ax.get_xticklabels(which="major"):
            lbl.set_rotation(0)
            lbl.set_horizontalalignment("center")


# -----------------------------------------------------------------------------
# Timestamp normalization + trimming
# -----------------------------------------------------------------------------

def _convert_metrics_timestamps_to_relative_seconds(metrics: LogMetrics) -> None:
    """
    Convert known timestamp series on metrics to relative seconds from the earliest timestamp
    found across all known timestamp series. Mutates metrics in-place.
    """
    ts_attr_names = [
        "ping_rtt_timestamps",
        "ping_rss_timestamps",
        "ping_packet_loss_timestamps",
        "parent_router_from_rloc16_transition_timestamps",
        "parent_rloc16_from_query_timestamps",
    ]

    all_ts: List[object] = []
    for name in ts_attr_names:
        vals = getattr(metrics, name, [])
        if vals:
            all_ts.extend(vals)

    if not all_ts:
        return

    has_dt = any(isinstance(t, (datetime, date)) for t in all_ts)

    if has_dt:
        dt_values: List[datetime] = []
        for t in all_ts:
            if isinstance(t, datetime):
                dt_values.append(t)
            elif isinstance(t, date):
                dt_values.append(datetime.combine(t, datetime.min.time()))
        if not dt_values:
            return
        t0 = min(dt_values)

        def _to_seconds(t: object) -> float:
            if isinstance(t, datetime):
                return max(0.0, (t - t0).total_seconds())
            if isinstance(t, date):
                td = datetime.combine(t, datetime.min.time()) - t0
                return max(0.0, td.total_seconds())
            return 0.0

    else:
        t0_num = min(float(t) for t in all_ts)

        def _to_seconds(t: object) -> float:
            return max(0.0, float(t) - t0_num)

    for name in ts_attr_names:
        vals = getattr(metrics, name, [])
        if vals:
            setattr(metrics, name, [_to_seconds(t) for t in vals])


def _filter_xy(
    ts: List[float],
    ys: List[float],
    start: float,
    end: float,
) -> Tuple[List[float], List[float]]:
    if not ts or not ys:
        return [], []
    out_ts: List[float] = []
    out_ys: List[float] = []
    for t, y in zip(ts, ys):
        t_f = float(t)
        if start <= t_f <= end:
            out_ts.append(t_f)
            out_ys.append(y)
    return out_ts, out_ys


def _filter_t(
    ts: List[float],
    start: float,
    end: float,
) -> List[float]:
    if not ts:
        return []
    return [float(t) for t in ts if start <= float(t) <= end]


def _trim_parent_series(
    ts: List[float],
    vals: List[str],
    start: float,
    end: float,
) -> Tuple[List[float], List[str]]:
    """
    Trim a change-point parent series to [start, end] while preserving the parent state at 'start'.

    Strategy:
      - Determine the last sample at or before 'start' to establish the value at window start.
      - Insert a synthetic sample exactly at 'start' with that value.
      - Keep all subsequent change points with start < t <= end.
    """
    if not ts or not vals:
        return [], []

    pairs = sorted(zip(ts, vals), key=lambda x: float(x[0]))
    ts_sorted = [float(t) for t, _ in pairs]
    vals_sorted = [v for _, v in pairs]

    # Find the index of the last timestamp <= start
    idx = None
    for i, t in enumerate(ts_sorted):
        if t <= start:
            idx = i
        else:
            break

    if idx is not None:
        start_val = vals_sorted[idx]
    else:
        # No samples before window start; best available is the first sample's value.
        start_val = vals_sorted[0]

    new_ts: List[float] = [start]
    new_vals: List[str] = [start_val]

    for t, v in zip(ts_sorted, vals_sorted):
        if start < t <= end:
            new_ts.append(t)
            new_vals.append(v)

    return new_ts, new_vals


def _rebase_timestamps(metrics: LogMetrics, offset: float) -> None:
    """Subtract 'offset' from all known timestamp series (in-place)."""
    ts_attr_names = [
        "ping_rtt_timestamps",
        "ping_rss_timestamps",
        "ping_packet_loss_timestamps",
        "parent_router_from_rloc16_transition_timestamps",
        "parent_rloc16_from_query_timestamps",
    ]
    for name in ts_attr_names:
        vals = getattr(metrics, name, [])
        if vals:
            setattr(metrics, name, [float(t) - offset for t in vals])


def _trim_metrics_centered_to_window(metrics: LogMetrics, window_seconds: float) -> Optional[float]:
    """
    Trim all metric series to a centered window of length 'window_seconds',
    removing equal time from the beginning and end based on the *per-file* overall duration.

    Returns the resulting plotted duration (normally == window_seconds), or None if no timestamps exist.
    """
    ts_attr_names = [
        "ping_rtt_timestamps",
        "ping_rss_timestamps",
        "ping_packet_loss_timestamps",
        "parent_router_from_rloc16_transition_timestamps",
        "parent_rloc16_from_query_timestamps",
    ]

    all_ts: List[float] = []
    for name in ts_attr_names:
        vals = getattr(metrics, name, [])
        if vals:
            all_ts.extend([float(t) for t in vals])

    if not all_ts:
        return None

    t_min = min(all_ts)
    t_max = max(all_ts)
    duration = t_max - t_min

    # If the available duration is shorter than the window, keep as-is (still rebase to start).
    if duration <= window_seconds:
        window_start = t_min
        window_end = t_max
        plotted_duration = window_end - window_start

        # Rebase so the plot starts at 0.
        _rebase_timestamps(metrics, window_start)
        return plotted_duration

    excess = duration - window_seconds
    cut = excess / 2.0
    window_start = t_min + cut
    window_end = t_max - cut
    plotted_duration = window_end - window_start  # should be == window_seconds

    # RTT
    metrics.ping_rtt_timestamps, metrics.ping_rtt_avg_ms = _filter_xy(
        [float(t) for t in getattr(metrics, "ping_rtt_timestamps", [])],
        list(getattr(metrics, "ping_rtt_avg_ms", [])),
        window_start,
        window_end,
    )

    # RSS
    metrics.ping_rss_timestamps, metrics.ping_rss_dbm_values = _filter_xy(
        [float(t) for t in getattr(metrics, "ping_rss_timestamps", [])],
        list(getattr(metrics, "ping_rss_dbm_values", [])),
        window_start,
        window_end,
    )

    # Packet loss (timestamps only)
    metrics.ping_packet_loss_timestamps = _filter_t(
        [float(t) for t in getattr(metrics, "ping_packet_loss_timestamps", [])],
        window_start,
        window_end,
    )

    # Parent series (preserve state at window_start)
    ts1 = [float(t) for t in getattr(metrics, "parent_router_from_rloc16_transition_timestamps", [])]
    v1 = list(getattr(metrics, "parent_router_from_rloc16_transition_values", []))
    if ts1 and v1:
        new_ts1, new_v1 = _trim_parent_series(ts1, v1, window_start, window_end)
        setattr(metrics, "parent_router_from_rloc16_transition_timestamps", new_ts1)
        setattr(metrics, "parent_router_from_rloc16_transition_values", new_v1)

    ts2 = [float(t) for t in getattr(metrics, "parent_rloc16_from_query_timestamps", [])]
    v2 = list(getattr(metrics, "parent_rloc16_from_query_values", []))
    if ts2 and v2:
        new_ts2, new_v2 = _trim_parent_series(ts2, v2, window_start, window_end)
        setattr(metrics, "parent_rloc16_from_query_timestamps", new_ts2)
        setattr(metrics, "parent_rloc16_from_query_values", new_v2)

    # Rebase timestamps so the plotted window starts at 0.
    _rebase_timestamps(metrics, window_start)

    return plotted_duration


# -----------------------------------------------------------------------------
# Plotting primitives (3-panel figure)
# -----------------------------------------------------------------------------

def plot_rtt(ax, metrics: LogMetrics) -> None:
    ts = metrics.ping_rtt_timestamps
    rtt = metrics.ping_rtt_avg_ms
    n = len(rtt)

    if ts and rtt:
        ax.plot(ts, rtt, marker=".", linestyle="", label="RTT (ms)", zorder=4)
        _add_legend_on_top(ax)
    else:
        ax.text(0.5, 0.5, "No RTT data", transform=ax.transAxes, ha="center", va="center")

    ax.set_ylabel("RTT (ms)")

    mu, sigma = _mean_std(rtt)
    title = "Ping to Parent Round-trip Time"
    suffix_parts: List[str] = []
    if n > 0:
        suffix_parts.append(f"nRTT={n}")
    if mu is not None and sigma is not None:
        suffix_parts.append(f"avg={mu:.1f} ms, std={sigma:.1f} ms")
    if suffix_parts:
        title += " (" + ", ".join(suffix_parts) + ")"
    ax.set_title(title)

    ax.grid(True)

    upper = max(1000.0, float(max(rtt))) if rtt else 1000.0
    ax.set_ylim(0.0, upper)


def plot_rss_and_loss(ax, metrics: LogMetrics) -> None:
    ts_rss = metrics.ping_rss_timestamps
    rss = metrics.ping_rss_dbm_values
    loss_ts = metrics.ping_packet_loss_timestamps

    if RSS_YLIM is not None:
        y_min, y_max = RSS_YLIM
    else:
        if rss:
            y_min = min(rss) - 5.0
            y_max = max(rss) + 5.0
        else:
            y_min, y_max = (-120.0, 0.0)

        y_min = min(y_min, PPS_RSS_THRESHOLD_DBM - 5.0)
        y_max = max(y_max, PPS_RSS_THRESHOLD_DBM + 5.0)

    ax.set_ylim(y_min, y_max)

    if loss_ts:
        y_range = y_max - y_min
        rug_bottom = y_max - (LOSS_RUG_FRACTION * y_range)
        rug_top = y_max
        ax.vlines(
            loss_ts,
            rug_bottom,
            rug_top,
            colors="red",
            linestyles="--",
            linewidth=0.5,
            label="Packet loss",
            zorder=6,
        )

    if ts_rss and rss:
        ax.plot(ts_rss, rss, marker=".", linestyle="", label="RTT RSS", zorder=7)
    else:
        ax.text(0.5, 0.5, "No RSS data", transform=ax.transAxes, ha="center", va="center")

    ax.axhline(
        PPS_RSS_THRESHOLD_DBM,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="PPS RSS Threshold",
        zorder=10,
    )

    _add_legend_on_top(ax)

    ax.set_ylabel("RSS (dBm)")

    pdr = _overall_pdr(metrics)
    n_rss = len(rss)
    mu, sigma = _mean_std(rss)

    title = "Ping to Parent RSS & Packet Loss"
    suffix_parts: List[str] = []
    if pdr is not None:
        suffix_parts.append(f"PDR={pdr:.1f}%")
    if n_rss > 0:
        suffix_parts.append(f"nRSS={n_rss}")
    if mu is not None and sigma is not None:
        suffix_parts.append(f"avgRSS={mu:.1f} dBm, stdRSS={sigma:.1f} dB")
    if suffix_parts:
        title += " (" + ", ".join(suffix_parts) + ")"
    ax.set_title(title)

    ax.grid(True)


def _select_parent_series(metrics: LogMetrics):
    ts = getattr(metrics, "parent_router_from_rloc16_transition_timestamps", [])
    vals = getattr(metrics, "parent_router_from_rloc16_transition_values", [])
    if ts and vals:
        return ts, vals

    ts2 = getattr(metrics, "parent_rloc16_from_query_timestamps", [])
    vals2 = getattr(metrics, "parent_rloc16_from_query_values", [])
    return ts2, vals2


def plot_parents(ax, metrics: LogMetrics, *, end_time: Optional[float] = None) -> None:
    """
    Gantt-style parent connectivity timeline.

    If end_time is provided, the last segment is extended to that value (in seconds).
    """
    parent_ts, parent_vals = _select_parent_series(metrics)

    if not parent_ts:
        ax.set_title("Connected to Parent")
        ax.text(0.5, 0.5, "No parent data", transform=ax.transAxes, ha="center", va="center")
        ax.set_yticks([])
        ax.grid(True)
        return

    pairs = sorted(zip(parent_ts, parent_vals), key=lambda x: float(x[0]))
    parent_ts_sorted = [float(t) for t, _ in pairs]
    parent_vals_sorted = [p for _, p in pairs]

    # Establish an end time for the final segment.
    if end_time is not None:
        overall_end = float(end_time)
    else:
        all_ts: List[float] = []
        for name in (
            "ping_rtt_timestamps",
            "ping_rss_timestamps",
            "ping_packet_loss_timestamps",
            "parent_router_from_rloc16_transition_timestamps",
            "parent_rloc16_from_query_timestamps",
        ):
            vals = getattr(metrics, name, [])
            if vals:
                all_ts.extend([float(v) for v in vals])
        overall_end = max(all_ts) if all_ts else parent_ts_sorted[-1]

    # Build change-point segments: (start, end, parent)
    segments: List[Tuple[float, float, str]] = []
    cur_parent = parent_vals_sorted[0]
    cur_start = parent_ts_sorted[0]

    for t, p in zip(parent_ts_sorted[1:], parent_vals_sorted[1:]):
        if p != cur_parent:
            segments.append((cur_start, t, cur_parent))
            cur_parent = p
            cur_start = t

    segments.append((cur_start, overall_end, cur_parent))

    # Compute dwell time per parent for ordering.
    dwell: Dict[str, float] = {}
    for s, e, p in segments:
        dwell[p] = dwell.get(p, 0.0) + max(0.0, e - s)

    unique_set = set(parent_vals_sorted)

    def _hex_key(p: str) -> int:
        try:
            return int(p, 16)
        except Exception:
            return 1_000_000_000

    others = [p for p in unique_set if p != "No Parent"]
    others_sorted = sorted(others, key=lambda p: (-dwell.get(p, 0.0), _hex_key(p), p))

    if "No Parent" in unique_set:
        unique_parents = ["No Parent"] + others_sorted
    else:
        unique_parents = others_sorted

    parent_to_index = {p: i for i, p in enumerate(unique_parents)}

    # Color mapping (stable within this subplot).
    cycle = plt.rcParams.get("axes.prop_cycle", None)
    cycle_colors = cycle.by_key().get("color", []) if cycle is not None else []
    color_map: Dict[str, str] = {}

    next_idx = 0
    for p in unique_parents:
        if p == "No Parent":
            color_map[p] = NO_PARENT_COLOR
        else:
            if cycle_colors:
                color_map[p] = cycle_colors[next_idx % len(cycle_colors)]
                next_idx += 1
            else:
                color_map[p] = "0.2"

    # Draw bars.
    bar_h = 0.8
    eps = 1e-6

    for s, e, p in segments:
        y = parent_to_index.get(p, None)
        if y is None:
            continue

        dur = e - s
        if dur <= 0:
            dur = eps

        ax.broken_barh(
            [(s, dur)],
            (y - bar_h / 2.0, bar_h),
            facecolors=color_map.get(p, "0.2"),
            edgecolors="none",
            alpha=0.9,
            zorder=4,
        )

    ax.set_ylabel("Parent RLOC16")
    ax.set_yticks(range(len(unique_parents)))
    ax.set_yticklabels(unique_parents, fontfamily="monospace")
    ax.set_ylim(-0.5, len(unique_parents) - 0.5)

    n_parents = max(0, len(segments))
    ax.set_title(f"Connected to Parent (nParents={n_parents})")

    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


# -----------------------------------------------------------------------------
# Per-file processing (save the 3-panel figure)
# -----------------------------------------------------------------------------

def process_log_file(
    log_path: str,
    rtt_by_file: Dict[str, List[float]],
    rss_by_file: Dict[str, List[float]],
    show: bool = False,
) -> None:
    log_path_obj = Path(log_path)
    data_dir_path = Path(DATA_DIR)

    rel_log_path = log_path_obj.relative_to(data_dir_path)
    graph_dir = GRAPHS_DIR / rel_log_path.parent
    graph_dir.mkdir(parents=True, exist_ok=True)

    label_for_file = str(rel_log_path)

    metrics = parse_log_file(log_path)

    # 1) Convert to relative seconds (per file).
    _convert_metrics_timestamps_to_relative_seconds(metrics)

    # 2) Center-trim to exactly 2 hours (per file), rebasing so window starts at 0.
    plotted_duration = _trim_metrics_centered_to_window(metrics, TRIM_WINDOW_SECONDS)

    if metrics.ping_rtt_avg_ms:
        rtt_by_file[label_for_file] = metrics.ping_rtt_avg_ms

    if getattr(metrics, "ping_rss_dbm_values", None):
        rss_by_file[label_for_file] = metrics.ping_rss_dbm_values

    has_any = (
        bool(metrics.ping_rtt_timestamps)
        or bool(metrics.ping_rss_timestamps)
        or bool(metrics.ping_packet_loss_timestamps)
        or bool(getattr(metrics, "parent_router_from_rloc16_transition_timestamps", []))
        or bool(getattr(metrics, "parent_rloc16_from_query_timestamps", []))
    )
    if not has_any:
        print(f"[INFO] {label_for_file}: no time-series data to plot; skipping figure.")
        return

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8))
    ax_rtt, ax_rss, ax_parent = axes

    plot_rtt(ax_rtt, metrics)
    plot_rss_and_loss(ax_rss, metrics)

    # If we successfully trimmed to a 2-hour window, force parent bars to extend to that window end.
    parent_end = None
    if plotted_duration is not None and abs(plotted_duration - TRIM_WINDOW_SECONDS) < 1e-6:
        parent_end = TRIM_WINDOW_SECONDS
    plot_parents(ax_parent, metrics, end_time=parent_end)

    _remove_x_whitespace(axes)
    _configure_elapsed_time_axis_hhmm(axes, interval_minutes=TIME_TICK_INTERVAL_MINUTES)

    ax_parent.set_xlabel("Elapsed time (HH:MM)")

    # Force x-range to exactly 2 hours when trimming succeeded.
    if parent_end is not None:
        ax_parent.set_xlim(0.0, TRIM_WINDOW_SECONDS)

    fig.suptitle(label_for_file, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_name = log_path_obj.stem + "_timeseries.png"
    out_path = graph_dir / out_name
    fig.savefig(out_path)
    print(f"[OK] Saved combined time-series graph for {label_for_file} -> {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def create_rtt_boxplot(rtt_by_file: Dict[str, List[float]]) -> None:
    if not rtt_by_file:
        print("[INFO] No RTT data collected; skipping RTT box plot.")
        return

    labels = list(rtt_by_file.keys())
    data = [rtt_by_file[label] for label in labels]

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("RTT (ms)")
    plt.title("Ping to Parent Round-trip Time per File")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    boxplot_path = GRAPHS_DIR / "all_files_rtt_boxplot.png"
    plt.savefig(boxplot_path)
    plt.close()
    print(f"[OK] Saved RTT box plot -> {boxplot_path}")


def create_rss_boxplot(rss_by_file: Dict[str, List[float]]) -> None:
    if not rss_by_file:
        print("[INFO] No RSS data collected; skipping RSS box plot.")
        return

    labels = list(rss_by_file.keys())
    data = [rss_by_file[label] for label in labels]

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("RSS (dBm)")
    plt.title("Ping to Parent RSS per File")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    boxplot_path = GRAPHS_DIR / "all_files_rss_boxplot.png"
    plt.savefig(boxplot_path)
    plt.close()
    print(f"[OK] Saved RSS box plot -> {boxplot_path}")


def main(show: bool = False) -> None:
    data_dir_path = Path(DATA_DIR)

    pattern = str(data_dir_path / "**" / "*.log")
    all_candidates = glob(pattern, recursive=True)

    log_files: List[str] = []
    for path_str in all_candidates:
        p = Path(path_str)
        try:
            rel_parts = p.relative_to(data_dir_path).parts
        except ValueError:
            continue

        dir_parts = rel_parts[:-1]
        if any(part.startswith(".") for part in dir_parts):
            continue

        log_files.append(path_str)

    if not log_files:
        print(f"[INFO] No .log files found in {data_dir_path}")
        return

    print(f"[INFO] Found {len(log_files)} .log file(s) in {data_dir_path} (excluding dot-directories):")
    for lf in log_files:
        print(f"  - {lf}")

    rtt_by_file: Dict[str, List[float]] = {}
    rss_by_file: Dict[str, List[float]] = {}

    for log_path in log_files:
        process_log_file(log_path, rtt_by_file, rss_by_file, show=show)

    create_rtt_boxplot(rtt_by_file)
    create_rss_boxplot(rss_by_file)


if __name__ == "__main__":
    main(show=False)
