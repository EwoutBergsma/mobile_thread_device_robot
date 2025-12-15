# openthread_log_visualizer.py
from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from openthread_log_parser import parse_log_file, LogMetrics, DATA_DIR as PARSER_DATA_DIR


# -----------------------------------------------------------------------------
# Output / plotting configuration
# -----------------------------------------------------------------------------

# Base directory: the directory where this visualizer script resides
SCRIPT_DIR = Path(__file__).resolve().parent

# Data directory: prefer parser's DATA_DIR to keep behavior consistent
DATA_DIR = Path(PARSER_DATA_DIR) if PARSER_DATA_DIR is not None else (SCRIPT_DIR / "data")

# Graph output directory (relative to this visualizer file)
GRAPHS_DIR = SCRIPT_DIR / "graphs"

# Fixed y-axis ranges (set to None to auto-scale)
RTT_YLIM: Optional[Tuple[float, float]] = (0, 3000)    # ms
RSS_YLIM: Optional[Tuple[float, float]] = (-120, 0)    # dBm

os.makedirs(GRAPHS_DIR, exist_ok=True)


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


# -----------------------------------------------------------------------------
# Plotting primitives (match previous 3-panel figure style)
# -----------------------------------------------------------------------------

def plot_rtt(ax, metrics: LogMetrics) -> None:
    ts = metrics.ping_rtt_timestamps
    rtt = metrics.ping_rtt_avg_ms
    n = len(rtt)

    if ts and rtt:
        ax.plot(ts, rtt, marker=".", linestyle="", label="RTT (ms)")
        ax.legend()
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
    if RTT_YLIM is not None:
        ax.set_ylim(*RTT_YLIM)


def plot_rss_and_loss(ax, metrics: LogMetrics) -> None:
    ts_rss = metrics.ping_rss_timestamps
    rss = metrics.ping_rss_dbm_values
    loss_ts = metrics.ping_packet_loss_timestamps

    plotted_any = False

    # Packet loss vertical markers
    if loss_ts:
        for i, t in enumerate(loss_ts):
            ax.axvline(
                t,
                linestyle="--",
                color="red",
                alpha=0.7,
                label="Packet loss" if i == 0 else None,
            )
        plotted_any = True

    # RSS scatter
    if ts_rss and rss:
        ax.plot(ts_rss, rss, marker=".", linestyle="", label="RTT RSS")
        plotted_any = True
    else:
        ax.text(0.5, 0.5, "No RSS data", transform=ax.transAxes, ha="center", va="center")

    if plotted_any:
        ax.legend()

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
    if RSS_YLIM is not None:
        ax.set_ylim(*RSS_YLIM)


def _select_parent_series(metrics: LogMetrics):
    """
    Prefer parent-router derived from node RLOC16 transitions (matches earlier plots where
    parent IDs looked like 0c00/4000/7000/etc). Fall back to CLI parent query series.
    """
    ts = getattr(metrics, "parent_router_from_rloc16_transition_timestamps", [])
    vals = getattr(metrics, "parent_router_from_rloc16_transition_values", [])
    if ts and vals:
        return ts, vals

    ts2 = getattr(metrics, "parent_rloc16_from_query_timestamps", [])
    vals2 = getattr(metrics, "parent_rloc16_from_query_values", [])
    return ts2, vals2


def plot_parents(ax, metrics: LogMetrics) -> None:
    parent_ts, parent_vals = _select_parent_series(metrics)

    if not parent_ts:
        ax.set_title("Connected to Parent")
        ax.text(0.5, 0.5, "No parent data", transform=ax.transAxes, ha="center", va="center")
        ax.set_yticks([])
        ax.grid(True)
        return

    unique_set = set(parent_vals)
    other_parents = [p for p in unique_set if p != "No Parent"]

    def sort_key(p: str):
        try:
            return (0, int(p, 16))
        except Exception:
            return (1, p)

    other_sorted = sorted(other_parents, key=sort_key)

    if "No Parent" in unique_set:
        unique_parents = ["No Parent"] + other_sorted
    else:
        unique_parents = other_sorted

    parent_to_index = {p: i for i, p in enumerate(unique_parents)}
    y = [parent_to_index[p] for p in parent_vals]

    ax.scatter(parent_ts, y)
    ax.set_ylabel("Parent (RLOC16)")
    ax.set_yticks(range(len(unique_parents)))
    ax.set_yticklabels(unique_parents)
    ax.set_ylim(-0.5, len(unique_parents) - 0.5)
    ax.set_title("Connected to Parent")
    ax.grid(True)


# -----------------------------------------------------------------------------
# Per-file processing (save the same 3-panel figure as before)
# -----------------------------------------------------------------------------

def process_log_file(log_path: str, rtt_by_file: Dict[str, List[float]], show: bool = False) -> None:
    log_path_obj = Path(log_path)
    data_dir_path = Path(DATA_DIR)

    # Mirror input subdirectories under GRAPHS_DIR
    rel_log_path = log_path_obj.relative_to(data_dir_path)
    graph_dir = GRAPHS_DIR / rel_log_path.parent
    graph_dir.mkdir(parents=True, exist_ok=True)

    label_for_file = str(rel_log_path)

    metrics = parse_log_file(log_path)

    # Collect RTTs for the across-files boxplot
    if metrics.ping_rtt_avg_ms:
        rtt_by_file[label_for_file] = metrics.ping_rtt_avg_ms

    # Skip figure only if absolutely nothing relevant is present
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
    plot_parents(ax_parent, metrics)

    ax_parent.set_xlabel("Time")
    fig.autofmt_xdate(rotation=45)

    # Top title: only the file label; stats are in subplot titles
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

        # Exclude dot-directories (e.g., .git, .cache)
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
    for log_path in log_files:
        process_log_file(log_path, rtt_by_file, show=show)

    create_rtt_boxplot(rtt_by_file)


if __name__ == "__main__":
    # Set show=True if you want interactive windows in addition to saved PNGs.
    main(show=False)
