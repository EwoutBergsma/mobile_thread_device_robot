# openthread_log_visualizer.py
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.dates as mdates
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

# TX-failed "rug" settings (fraction of y-range at the top)
TXFAIL_RUG_FRACTION: float = 1  # fraction of y-range at top

# Legend styling
LEGEND_FRAME_ALPHA: float = 1.0
LEGEND_ZORDER: int = 50

# X-axis tick interval (used for BOTH relative and absolute time axes)
TIME_TICK_INTERVAL_MINUTES: int = 5

# Parent plot styling
NO_PARENT_COLOR: str = "0.35"  # dark grey (0=black, 1=white)

# Trim settings
TRIM_WINDOW_SECONDS: float = 2 * 60 * 60  # exactly 2 hours

# Toggle: enable/disable the top RTT subplot
SHOW_RTT_SUBPLOT: bool = False

# Toggle: relative elapsed time vs absolute timestamps on x-axis
# - True  => elapsed time (HH:MM)
# - False => absolute timestamps (requires datetime timestamps from parser)
USE_RELATIVE_TIME_AXIS: bool = True

# Subplot height ratios (make "Connected to Parent" slightly less tall)
PARENT_SUBPLOT_HEIGHT_RATIO: float = 0.5


# -----------------------------------------------------------------------------
# RLOC16 -> Router number mapping
# -----------------------------------------------------------------------------
# NOTE: Routers 2 and 4 have two possible RLOC16 values (RLOC16 changed over time).
RLOC16_TO_ROUTER_NUM: Dict[str, int] = {
    "7000": 2,
    "C400": 3,
    "7800": 3,
    "E000": 1,
    "C800": 5,
    "2400": 5,
    "0C00": 4,
}

# Always show these labels on the parent subplot, even if absent in the data.
BASE_PARENT_LABELS: List[str] = [
    "No Parent",
    "Router 1",
    "Router 2",
    "Router 3",
    "Router 4",
    "Router 5",
]


def _normalize_rloc16(rloc16: str) -> str:
    """
    Normalize an RLOC16 string to a canonical 4-hex-digit uppercase representation.
    Examples: "0xc400" -> "C400", "0c00" -> "0C00"
    """
    s = str(rloc16).strip()
    if s.lower().startswith("0x"):
        s = s[2:]
    s = s.upper()
    # Keep only hex characters if the string is noisy (defensive).
    s = "".join(ch for ch in s if ch in "0123456789ABCDEF")
    if len(s) == 0:
        return ""
    # Left-pad to 4 chars if shorter than expected.
    if len(s) < 4:
        s = s.zfill(4)
    return s


def _rloc16_value_to_router_label(value: object) -> str:
    """
    Convert a raw parent value (RLOC16 / "No Parent" / etc.) to a router label.
    - Known RLOC16s map to "Router N"
    - "No Parent" remains "No Parent"
    - Unknown values become "Unknown (<RLOC16>)"
    """
    if value is None:
        return "No Parent"

    s = str(value).strip()
    if not s:
        return "No Parent"
    if s.lower() == "no parent":
        return "No Parent"

    norm = _normalize_rloc16(s)
    if not norm:
        return "No Parent"

    router_num = RLOC16_TO_ROUTER_NUM.get(norm)
    if router_num is not None:
        return f"Router {router_num}"

    return f"Unknown ({norm})"


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
    tick_step_seconds = max(1, interval_minutes) * 60
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
# Absolute-time axis formatting (FIXED interval; honors TIME_TICK_INTERVAL_MINUTES)
# -----------------------------------------------------------------------------

def _configure_absolute_time_axis(axes, *, interval_minutes: int = TIME_TICK_INTERVAL_MINUTES) -> None:
    """
    Configure absolute datetime x-axis ticks using a fixed minute/hour interval.
    This makes TIME_TICK_INTERVAL_MINUTES effective in absolute mode.
    """
    if interval_minutes <= 0:
        interval_minutes = 1

    # Use hour-based ticks if interval is large and divisible by 60.
    if interval_minutes >= 60 and interval_minutes % 60 == 0:
        hour_interval = max(1, interval_minutes // 60)
        locator = mdates.HourLocator(interval=hour_interval)
    else:
        locator = mdates.MinuteLocator(interval=interval_minutes)

    formatter = mdates.ConciseDateFormatter(locator)

    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.offsetText.set_visible(False)
        for lbl in ax.get_xticklabels(which="major"):
            lbl.set_rotation(0)
            lbl.set_horizontalalignment("center")


def _configure_time_axis(
    axes,
    *,
    use_relative_time: bool,
    interval_minutes: int = TIME_TICK_INTERVAL_MINUTES,
) -> None:
    if use_relative_time:
        _configure_elapsed_time_axis_hhmm(axes, interval_minutes=interval_minutes)
    else:
        _configure_absolute_time_axis(axes, interval_minutes=interval_minutes)


# -----------------------------------------------------------------------------
# Timestamp normalization + trimming helpers
# -----------------------------------------------------------------------------

_TS_FLOAT = float
_TS_DT = datetime
TimeStamp = Union[_TS_FLOAT, _TS_DT]


def _to_datetime(t: object) -> Optional[datetime]:
    if isinstance(t, datetime):
        return t
    if isinstance(t, date):
        return datetime.combine(t, datetime.min.time())
    return None


def _metrics_has_datetime_timestamps(metrics: LogMetrics) -> bool:
    ts_attr_names = [
        "ping_rtt_timestamps",
        "ping_rss_timestamps",
        "mac_frame_tx_attempt_16_16_failed_timestamps",
        "parent_router_from_rloc16_transition_timestamps",
        "parent_rloc16_from_query_timestamps",
    ]
    for name in ts_attr_names:
        vals = getattr(metrics, name, [])
        if not vals:
            continue
        for t in vals:
            if isinstance(t, (datetime, date)):
                return True
    return False


def _convert_metrics_timestamps_to_relative_seconds(metrics: LogMetrics) -> None:
    """
    Convert known timestamp series on metrics to relative seconds from the earliest timestamp
    found across all known timestamp series. Mutates metrics in-place.
    """
    ts_attr_names = [
        "ping_rtt_timestamps",
        "ping_rss_timestamps",
        "mac_frame_tx_attempt_16_16_failed_timestamps",
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
            dt = _to_datetime(t)
            if dt is not None:
                dt_values.append(dt)
        if not dt_values:
            return
        t0 = min(dt_values)

        def _to_seconds(t: object) -> float:
            dt = _to_datetime(t)
            if dt is not None:
                return max(0.0, (dt - t0).total_seconds())
            return 0.0

    else:
        t0_num = min(float(t) for t in all_ts)

        def _to_seconds(t: object) -> float:
            return max(0.0, float(t) - t0_num)

    for name in ts_attr_names:
        vals = getattr(metrics, name, [])
        if vals:
            setattr(metrics, name, [_to_seconds(t) for t in vals])


def _filter_xy(ts: List[float], ys: List[float], start: float, end: float) -> Tuple[List[float], List[float]]:
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


def _filter_t(ts: List[float], start: float, end: float) -> List[float]:
    if not ts:
        return []
    return [float(t) for t in ts if start <= float(t) <= end]


def _filter_xy_dt(
    ts: List[datetime],
    ys: List[float],
    start: datetime,
    end: datetime,
) -> Tuple[List[datetime], List[float]]:
    if not ts or not ys:
        return [], []
    out_ts: List[datetime] = []
    out_ys: List[float] = []
    for t, y in zip(ts, ys):
        if start <= t <= end:
            out_ts.append(t)
            out_ys.append(y)
    return out_ts, out_ys


def _filter_t_dt(ts: List[datetime], start: datetime, end: datetime) -> List[datetime]:
    if not ts:
        return []
    return [t for t in ts if start <= t <= end]


def _trim_parent_series(ts: List[float], vals: List[str], start: float, end: float) -> Tuple[List[float], List[str]]:
    """
    Trim a change-point parent series to [start, end] while preserving the parent state at 'start'.
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

    start_val = vals_sorted[idx] if idx is not None else vals_sorted[0]

    new_ts: List[float] = [start]
    new_vals: List[str] = [start_val]

    for t, v in zip(ts_sorted, vals_sorted):
        if start < t <= end:
            new_ts.append(t)
            new_vals.append(v)

    return new_ts, new_vals


def _trim_parent_series_dt(
    ts: List[datetime],
    vals: List[str],
    start: datetime,
    end: datetime,
) -> Tuple[List[datetime], List[str]]:
    """
    Datetime version of _trim_parent_series().
    """
    if not ts or not vals:
        return [], []

    pairs = sorted(zip(ts, vals), key=lambda x: x[0])
    ts_sorted = [t for t, _ in pairs]
    vals_sorted = [v for _, v in pairs]

    idx = None
    for i, t in enumerate(ts_sorted):
        if t <= start:
            idx = i
        else:
            break

    start_val = vals_sorted[idx] if idx is not None else vals_sorted[0]

    new_ts: List[datetime] = [start]
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
        "mac_frame_tx_attempt_16_16_failed_timestamps",
        "parent_router_from_rloc16_transition_timestamps",
        "parent_rloc16_from_query_timestamps",
    ]
    for name in ts_attr_names:
        vals = getattr(metrics, name, [])
        if vals:
            setattr(metrics, name, [float(t) - offset for t in vals])


def _trim_metrics_centered_to_window(metrics: LogMetrics, window_seconds: float) -> Optional[float]:
    """
    Float-seconds version:
    Trim all metric series to a centered window of length 'window_seconds',
    rebasing timestamps so the plotted window starts at 0.

    Returns plotted duration, or None if no timestamps exist.
    """
    ts_attr_names = [
        "ping_rtt_timestamps",
        "ping_rss_timestamps",
        "mac_frame_tx_attempt_16_16_failed_timestamps",
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

    if duration <= window_seconds:
        window_start = t_min
        window_end = t_max
        plotted_duration = window_end - window_start
        _rebase_timestamps(metrics, window_start)
        return plotted_duration

    excess = duration - window_seconds
    cut = excess / 2.0
    window_start = t_min + cut
    window_end = t_max - cut
    plotted_duration = window_end - window_start

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

    # TX failed 16/16 (timestamps only)
    metrics.mac_frame_tx_attempt_16_16_failed_timestamps = _filter_t(
        [float(t) for t in getattr(metrics, "mac_frame_tx_attempt_16_16_failed_timestamps", [])],
        window_start,
        window_end,
    )

    # Parent series (preserve state at window_start)
    ts1 = [float(t) for t in getattr(metrics, "parent_router_from_rloc16_transition_timestamps", [])]
    v1 = list(getattr(metrics, "parent_router_from_rloc16_transition_values", []))
    if ts1 and v1:
        new_ts1, new_v1 = _trim_parent_series(ts1, v1, window_start, window_end)
        metrics.parent_router_from_rloc16_transition_timestamps = new_ts1
        metrics.parent_router_from_rloc16_transition_values = new_v1

    ts2 = [float(t) for t in getattr(metrics, "parent_rloc16_from_query_timestamps", [])]
    v2 = list(getattr(metrics, "parent_rloc16_from_query_values", []))
    if ts2 and v2:
        new_ts2, new_v2 = _trim_parent_series(ts2, v2, window_start, window_end)
        metrics.parent_rloc16_from_query_timestamps = new_ts2
        metrics.parent_rloc16_from_query_values = new_v2

    _rebase_timestamps(metrics, window_start)
    return plotted_duration


def _trim_metrics_centered_to_window_absolute(
    metrics: LogMetrics,
    window_seconds: float,
) -> Optional[Tuple[datetime, datetime]]:
    """
    Absolute (datetime) version:
    Trim all metric series to a centered window of length 'window_seconds',
    WITHOUT rebasing. Returns (window_start, window_end), or None if no timestamps exist.

    Requires datetime/date timestamps from the parser.
    """
    ts_attr_names = [
        "ping_rtt_timestamps",
        "ping_rss_timestamps",
        "mac_frame_tx_attempt_16_16_failed_timestamps",
        "parent_router_from_rloc16_transition_timestamps",
        "parent_rloc16_from_query_timestamps",
    ]

    all_dt: List[datetime] = []
    for name in ts_attr_names:
        vals = getattr(metrics, name, [])
        if not vals:
            continue
        for t in vals:
            dt = _to_datetime(t)
            if dt is not None:
                all_dt.append(dt)

    if not all_dt:
        return None

    t_min = min(all_dt)
    t_max = max(all_dt)
    duration = (t_max - t_min).total_seconds()

    if duration <= window_seconds:
        window_start = t_min
        window_end = t_max
    else:
        excess_seconds = duration - window_seconds
        cut = timedelta(seconds=excess_seconds / 2.0)
        window_start = t_min + cut
        window_end = t_max - cut

    def _get_dt_list(attr: str) -> List[datetime]:
        raw = getattr(metrics, attr, [])
        out: List[datetime] = []
        for t in raw:
            dt = _to_datetime(t)
            if dt is not None:
                out.append(dt)
        return out

    # RTT
    rtt_ts = _get_dt_list("ping_rtt_timestamps")
    rtt_vals = list(getattr(metrics, "ping_rtt_avg_ms", []))
    if rtt_ts and rtt_vals:
        new_ts, new_vals = _filter_xy_dt(rtt_ts, rtt_vals, window_start, window_end)
        metrics.ping_rtt_timestamps = new_ts
        metrics.ping_rtt_avg_ms = new_vals

    # RSS
    rss_ts = _get_dt_list("ping_rss_timestamps")
    rss_vals = list(getattr(metrics, "ping_rss_dbm_values", []))
    if rss_ts and rss_vals:
        new_ts, new_vals = _filter_xy_dt(rss_ts, rss_vals, window_start, window_end)
        metrics.ping_rss_timestamps = new_ts
        metrics.ping_rss_dbm_values = new_vals

    # TX fail
    tx_ts = _get_dt_list("mac_frame_tx_attempt_16_16_failed_timestamps")
    if tx_ts:
        metrics.mac_frame_tx_attempt_16_16_failed_timestamps = _filter_t_dt(tx_ts, window_start, window_end)

    # Parent series (preserve state at window_start)
    ts1 = _get_dt_list("parent_router_from_rloc16_transition_timestamps")
    v1 = list(getattr(metrics, "parent_router_from_rloc16_transition_values", []))
    if ts1 and v1:
        new_ts1, new_v1 = _trim_parent_series_dt(ts1, v1, window_start, window_end)
        metrics.parent_router_from_rloc16_transition_timestamps = new_ts1
        metrics.parent_router_from_rloc16_transition_values = new_v1

    ts2 = _get_dt_list("parent_rloc16_from_query_timestamps")
    v2 = list(getattr(metrics, "parent_rloc16_from_query_values", []))
    if ts2 and v2:
        new_ts2, new_v2 = _trim_parent_series_dt(ts2, v2, window_start, window_end)
        metrics.parent_rloc16_from_query_timestamps = new_ts2
        metrics.parent_rloc16_from_query_values = new_v2

    return window_start, window_end


# -----------------------------------------------------------------------------
# Plotting primitives
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


def plot_rss_and_txfail(ax, metrics: LogMetrics) -> None:
    """
    RSS scatter + vertical 'rug' for Frame tx attempt 16/16 failed events.
    """
    ts_rss = metrics.ping_rss_timestamps
    rss = metrics.ping_rss_dbm_values
    txfail_ts = getattr(metrics, "mac_frame_tx_attempt_16_16_failed_timestamps", [])

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

    # TX failed 16/16 rug (vertical markers)
    if txfail_ts:
        y_range = y_max - y_min
        rug_bottom = y_max - (TXFAIL_RUG_FRACTION * y_range)
        rug_top = y_max
        ax.vlines(
            txfail_ts,
            rug_bottom,
            rug_top,
            colors="red",
            linestyles="--",
            linewidth=0.5,
            label="Frame tx attempt 16/16 failed",
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
    n_txfail = len(txfail_ts)
    mu, sigma = _mean_std(rss)

    title = "Ping to Parent RSS & TX Fail (16/16)"
    suffix_parts: List[str] = []
    if pdr is not None:
        suffix_parts.append(f"PDR={pdr:.1f}%")
    if n_rss > 0:
        suffix_parts.append(f"nRSS={n_rss}")
    if n_txfail > 0:
        suffix_parts.append(f"nTxFail16/16={n_txfail}")
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


def plot_parents(ax, metrics: LogMetrics, *, end_time: Optional[object] = None) -> None:
    """
    Gantt-style parent connectivity timeline.

    Supports both:
      - relative float seconds
      - absolute datetime timestamps
    """
    parent_ts, parent_vals = _select_parent_series(metrics)

    if not parent_ts:
        ax.set_title("Connected to Parent (nParents=0)")
        ax.text(0.5, 0.5, "No parent data", transform=ax.transAxes, ha="center", va="center")

        ax.set_ylabel("Parent Router")
        ax.set_yticks(range(len(BASE_PARENT_LABELS)))
        ax.set_yticklabels(BASE_PARENT_LABELS)
        ax.set_ylim(-0.5, len(BASE_PARENT_LABELS) - 0.5)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        return

    is_dt = isinstance(parent_ts[0], (datetime, date))

    if is_dt:
        pairs = sorted(
            [(_to_datetime(t), v) for t, v in zip(parent_ts, parent_vals) if _to_datetime(t) is not None],
            key=lambda x: x[0],
        )
        parent_ts_sorted: List[datetime] = [t for t, _ in pairs]
        parent_router_labels = [_rloc16_value_to_router_label(p) for _, p in pairs]
    else:
        pairs = sorted(zip(parent_ts, parent_vals), key=lambda x: float(x[0]))
        parent_ts_sorted = [float(t) for t, _ in pairs]
        parent_router_labels = [_rloc16_value_to_router_label(p) for _, p in pairs]

    # Establish an end time for the final segment.
    if end_time is not None:
        overall_end = end_time
    else:
        if is_dt:
            all_dt: List[datetime] = []
            for name in (
                "ping_rtt_timestamps",
                "ping_rss_timestamps",
                "mac_frame_tx_attempt_16_16_failed_timestamps",
                "parent_router_from_rloc16_transition_timestamps",
                "parent_rloc16_from_query_timestamps",
            ):
                vals = getattr(metrics, name, [])
                if not vals:
                    continue
                for t in vals:
                    dt = _to_datetime(t)
                    if dt is not None:
                        all_dt.append(dt)
            overall_end = max(all_dt) if all_dt else parent_ts_sorted[-1]
        else:
            all_ts: List[float] = []
            for name in (
                "ping_rtt_timestamps",
                "ping_rss_timestamps",
                "mac_frame_tx_attempt_16_16_failed_timestamps",
                "parent_router_from_rloc16_transition_timestamps",
                "parent_rloc16_from_query_timestamps",
            ):
                vals = getattr(metrics, name, [])
                if vals:
                    all_ts.extend([float(v) for v in vals])
            overall_end = max(all_ts) if all_ts else parent_ts_sorted[-1]

    # Build change-point segments: (start, end, parent_router_label)
    segments: List[Tuple[object, object, str]] = []
    cur_parent = parent_router_labels[0]
    cur_start = parent_ts_sorted[0]

    for t, p in zip(parent_ts_sorted[1:], parent_router_labels[1:]):
        if p != cur_parent:
            segments.append((cur_start, t, cur_parent))
            cur_parent = p
            cur_start = t

    segments.append((cur_start, overall_end, cur_parent))

    unique_set = set(parent_router_labels)
    unique_parents: List[str] = list(BASE_PARENT_LABELS)
    extras = sorted([p for p in unique_set if p not in set(BASE_PARENT_LABELS)])
    unique_parents.extend(extras)

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

        if is_dt:
            s_dt = _to_datetime(s)
            e_dt = _to_datetime(e)
            if s_dt is None or e_dt is None:
                continue
            s_num = mdates.date2num(s_dt)
            e_num = mdates.date2num(e_dt)
            dur = e_num - s_num
            if dur <= 0:
                dur = eps
            ax.broken_barh(
                [(s_num, dur)],
                (y - bar_h / 2.0, bar_h),
                facecolors=color_map.get(p, "0.2"),
                edgecolors="none",
                alpha=0.9,
                zorder=4,
            )
        else:
            s_f = float(s)
            e_f = float(e)
            dur = e_f - s_f
            if dur <= 0:
                dur = eps
            ax.broken_barh(
                [(s_f, dur)],
                (y - bar_h / 2.0, bar_h),
                facecolors=color_map.get(p, "0.2"),
                edgecolors="none",
                alpha=0.9,
                zorder=4,
            )

    ax.set_ylabel("Parent Router")
    ax.set_yticks(range(len(unique_parents)))
    ax.set_yticklabels(unique_parents)
    ax.set_ylim(-0.5, len(unique_parents) - 0.5)

    n_parents = max(0, len(segments))
    ax.set_title(f"Connected to Parent (nParents={n_parents})")

    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


# -----------------------------------------------------------------------------
# Per-file processing (save the figure)
# -----------------------------------------------------------------------------

def process_log_file(
    log_path: str,
    rtt_by_file: Dict[str, List[float]],
    rss_by_file: Dict[str, List[float]],
    *,
    show: bool = False,
    show_rtt: bool = SHOW_RTT_SUBPLOT,
    use_relative_time: bool = USE_RELATIVE_TIME_AXIS,
) -> None:
    log_path_obj = Path(log_path)
    data_dir_path = Path(DATA_DIR)

    rel_log_path = log_path_obj.relative_to(data_dir_path)
    graph_dir = GRAPHS_DIR / rel_log_path.parent
    graph_dir.mkdir(parents=True, exist_ok=True)

    label_for_file = str(rel_log_path)

    metrics = parse_log_file(log_path)

    # If absolute time was requested but timestamps aren't datetimes, fall back to relative.
    if not use_relative_time and not _metrics_has_datetime_timestamps(metrics):
        print(
            f"[INFO] {label_for_file}: absolute time requested but no datetime timestamps detected; "
            f"falling back to relative time."
        )
        use_relative_time = True

    plotted_duration: Optional[float] = None
    abs_window: Optional[Tuple[datetime, datetime]] = None

    if use_relative_time:
        _convert_metrics_timestamps_to_relative_seconds(metrics)
        plotted_duration = _trim_metrics_centered_to_window(metrics, TRIM_WINDOW_SECONDS)
    else:
        abs_window = _trim_metrics_centered_to_window_absolute(metrics, TRIM_WINDOW_SECONDS)

    if metrics.ping_rtt_avg_ms:
        rtt_by_file[label_for_file] = metrics.ping_rtt_avg_ms

    if getattr(metrics, "ping_rss_dbm_values", None):
        rss_by_file[label_for_file] = metrics.ping_rss_dbm_values

    has_any = (
        (show_rtt and bool(metrics.ping_rtt_timestamps))
        or bool(metrics.ping_rss_timestamps)
        or bool(getattr(metrics, "mac_frame_tx_attempt_16_16_failed_timestamps", []))
        or bool(getattr(metrics, "parent_router_from_rloc16_transition_timestamps", []))
        or bool(getattr(metrics, "parent_rloc16_from_query_timestamps", []))
    )
    if not has_any:
        print(f"[INFO] {label_for_file}: no time-series data to plot; skipping figure.")
        return

    # Create either a 3-panel (with RTT) or 2-panel (without RTT) figure.
    # Make the "Connected to Parent" subplot slightly shorter via height_ratios.
    if show_rtt:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            sharex=True,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [1.0, 1.0, PARENT_SUBPLOT_HEIGHT_RATIO]},
        )
        ax_rtt, ax_rss, ax_parent = axes
        plot_rtt(ax_rtt, metrics)
    else:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(12, 6),
            gridspec_kw={"height_ratios": [1.0, PARENT_SUBPLOT_HEIGHT_RATIO]},
        )
        ax_rss, ax_parent = axes

    plot_rss_and_txfail(ax_rss, metrics)

    # Parent end handling
    parent_end: Optional[object] = None
    if use_relative_time:
        if plotted_duration is not None and abs(plotted_duration - TRIM_WINDOW_SECONDS) < 1e-6:
            parent_end = TRIM_WINDOW_SECONDS
    else:
        if abs_window is not None:
            parent_end = abs_window[1]

    plot_parents(ax_parent, metrics, end_time=parent_end)

    _remove_x_whitespace(axes)

    # IMPORTANT: interval now applies to BOTH relative and absolute axes.
    _configure_time_axis(
        axes,
        use_relative_time=use_relative_time,
        interval_minutes=TIME_TICK_INTERVAL_MINUTES,
    )

    if use_relative_time:
        ax_parent.set_xlabel("Elapsed time (HH:MM)")
    else:
        ax_parent.set_xlabel("Time")

    # Force x-range to exactly 2 hours when trimming succeeded.
    if use_relative_time and parent_end is not None:
        ax_parent.set_xlim(0.0, TRIM_WINDOW_SECONDS)
    elif (not use_relative_time) and abs_window is not None:
        # Parent axis uses broken_barh with date2num floats, so enforce numeric xlim.
        ax_parent.set_xlim(mdates.date2num(abs_window[0]), mdates.date2num(abs_window[1]))

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
    plt.title("Ping to Parent Round-trip Time")
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


def main(
    show: bool = False,
    show_rtt: bool = SHOW_RTT_SUBPLOT,
    use_relative_time: bool = USE_RELATIVE_TIME_AXIS,
) -> None:
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
        process_log_file(
            log_path,
            rtt_by_file,
            rss_by_file,
            show=show,
            show_rtt=show_rtt,
            use_relative_time=use_relative_time,
        )

    create_rtt_boxplot(rtt_by_file)
    create_rss_boxplot(rss_by_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenThread log visualizer")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")

    # Mutually exclusive RTT toggles (optional overrides)
    group_rtt = parser.add_mutually_exclusive_group()
    group_rtt.add_argument("--rtt", action="store_true", help="Enable the top RTT subplot")
    group_rtt.add_argument("--no-rtt", action="store_true", help="Disable the top RTT subplot")

    # Mutually exclusive time-axis toggles
    group_time = parser.add_mutually_exclusive_group()
    group_time.add_argument(
        "--relative-time",
        action="store_true",
        help="Use elapsed time (HH:MM) on x-axis (default unless config overrides)",
    )
    group_time.add_argument(
        "--absolute-time",
        action="store_true",
        help="Use absolute timestamps on x-axis (requires datetime timestamps from parser)",
    )

    args = parser.parse_args()

    # Default comes from config unless user overrides via CLI.
    show_rtt = SHOW_RTT_SUBPLOT
    if args.rtt:
        show_rtt = True
    elif args.no_rtt:
        show_rtt = False

    use_relative_time = USE_RELATIVE_TIME_AXIS
    if args.relative_time:
        use_relative_time = True
    elif args.absolute_time:
        use_relative_time = False

    main(
        show=args.show,
        show_rtt=show_rtt,
        use_relative_time=use_relative_time,
    )
