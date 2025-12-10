import re
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import os
from glob import glob
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

# Directories
DATA_DIR = "data"
GRAPHS_DIR = "graphs"

# Length (in bytes) of the ping reply ICMPv6 message in the log.
PING_REPLY_LEN = 56

# Fixed y-axis ranges (set to None to auto-scale)
RTT_YLIM = (0, 3000)       # ms
RSS_YLIM = (-120, 0)       # dBm

# Ensure base graphs directory exists
os.makedirs(GRAPHS_DIR, exist_ok=True)

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

# Regex for packet summary lines with packet loss (0.0%, 10.0%, 100.0%, etc.)
# Examples:
# [2025-12-04T17:52:34.591] 1 packets transmitted, 1 packets received. Packet loss = 0.0%. Round-trip ...
# [2025-12-04T17:53:28.088] 1 packets transmitted, 0 packets received. Packet loss = 100.0%.
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

# Regex for parent lines, using RLOC16 as parent IDs
parent_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]      # timestamp in brackets
    .*?Rloc:\s*
    (?P<rloc>[0-9A-Fa-f]+)  # hex-only RLOC16 value
    """,
    re.VERBOSE
)

# Regex for OT state lines
state_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]              # timestamp
    .*?\b(?P<state>disabled|detached|child|router|leader)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Regex for *ping reply* RSS lines (MeshForwarder Received IPv6 ... msg)
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
    # Totals for PDR over the entire file
    total_tx: int
    total_rx: int


def compute_effective_parent(state: Optional[str], parent: Optional[str]) -> str:
    """
    - If state is blank or "blank" or None -> "No Parent"
    - If state == "detached" -> "No Parent"
    - Else, if parent missing/none/nan/"" -> "No Parent"
    - Else -> parent RLOC16 string
    """
    def is_blank_state(v: Optional[str]) -> bool:
        if v is None:
            return True
        s = v.strip().lower()
        return s == "" or s == "blank"

    st = (state or "").strip().lower()

    # State overrides parent: detached / blank => no parent
    if is_blank_state(state) or st == "detached":
        return "No Parent"

    if parent is None:
        return "No Parent"
    p = parent.strip()
    if not p or p.lower() in ("none", "nan"):
        return "No Parent"

    return p


def build_parent_timeline(metrics: LogMetrics) -> None:
    """
    Build a time series of "effective parent" where:
      - state + parent are forward-filled over time
      - 'detached' or blank state => 'No Parent'
      - missing/none/nan parent => 'No Parent'

    Timeline includes all timestamps where we have RTT, loss, parent, state, or RSS info.
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

            # --- Packet loss / summary detection (any percentage) ---
            m_loss = loss_re.search(line_stripped)
            if m_loss:
                ts_loss_str = m_loss.group("ts")
                ts_loss = parse_timestamp(ts_loss_str)

                tx = int(m_loss.group("tx"))
                rx = int(m_loss.group("rx"))
                loss_pct = float(m_loss.group("loss"))

                # Accumulate totals for overall PDR
                metrics.total_tx += tx
                metrics.total_rx += rx

                # Keep timestamp for non-zero loss events (for vertical markers)
                if loss_pct > 0.0:
                    print(f"  [LOSS] Packet loss {loss_pct}% at {ts_loss_str}")
                    metrics.loss_timestamps.append(ts_loss)

            # --- RTT detection (only lines with Round-trip stats) ---
            m_rtt = rtt_re.search(line_stripped)
            if m_rtt:
                print(f"  [USE RTT] Line {line_no}: {line_stripped}")
                ts_str = m_rtt.group("ts")
                avg_str = m_rtt.group("avg")
                ts = parse_timestamp(ts_str)
                avg_ms = float(avg_str)
                metrics.rtt_timestamps.append(ts)
                metrics.rtt_avgs_ms.append(avg_ms)

            # --- Parent detection (RLOC16 from "Rloc:" lines) ---
            m_parent = parent_re.search(line_stripped)
            if m_parent:
                ts_str = m_parent.group("ts")
                rloc16 = m_parent.group("rloc")
                ts = parse_timestamp(ts_str)
                print(f"  [USE PARENT] Line {line_no}: {line_stripped}")
                print(f"               -> Parent RLOC16: {rloc16}")
                metrics.parent_timestamps.append(ts)
                metrics.parent_ids.append(rloc16)

            # --- State detection ---
            m_state = state_re.search(line_stripped)
            if m_state:
                ts_str = m_state.group("ts")
                state_str = m_state.group("state")
                ts = parse_timestamp(ts_str)
                state_norm = state_str.strip().lower()
                print(f"  [STATE] {state_norm} at {ts_str}")
                metrics.state_timestamps.append(ts)
                metrics.states.append(state_norm)

            # --- RSS detection for ping replies ---
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


def plot_rtt(
    ax,
    metrics: LogMetrics,
    rtt_mean_ms: Optional[float] = None,
    rtt_std_ms: Optional[float] = None,
) -> None:
    """
    Plot RTT on the provided Axes (no packet-loss here).
    Include number of RTT samples, and optionally mean and std in the title.
    """
    timestamps_rtt = metrics.rtt_timestamps
    avg_rtts_ms = metrics.rtt_avgs_ms
    n_rtt = len(avg_rtts_ms)

    if timestamps_rtt:
        ax.plot(
            timestamps_rtt,
            avg_rtts_ms,
            marker=".",
            linestyle="",
            label="RTT (ms)",
        )
        ax.legend()

    ax.set_ylabel("RTT (ms)")

    # Build title string
    title = "Ping to Parent Round-trip Time"
    suffix_parts = []

    if n_rtt > 0:
        suffix_parts.append(f"nRTT={n_rtt}")
    if rtt_mean_ms is not None and rtt_std_ms is not None:
        suffix_parts.append(f"avg={rtt_mean_ms:.1f} ms, std={rtt_std_ms:.1f} ms")

    if suffix_parts:
        title += " (" + ", ".join(suffix_parts) + ")"

    ax.set_title(title)
    ax.grid(True)

    if RTT_YLIM is not None:
        ax.set_ylim(*RTT_YLIM)


def plot_parents(
    ax,
    metrics: LogMetrics,
) -> None:
    """
    Plot parent over time using the *effective* parent.
    """
    if metrics.eff_parent_timestamps:
        parent_timestamps = metrics.eff_parent_timestamps
        parent_ids = metrics.eff_parents
    else:
        parent_timestamps = metrics.parent_timestamps
        parent_ids = metrics.parent_ids

    if not parent_timestamps:
        ax.set_title("Connected to Parent")
        ax.text(
            0.5,
            0.5,
            "No parent data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_yticks([])
        ax.grid(True)
        return

    unique_parents = sorted(set(parent_ids))
    parent_to_index = {p: i for i, p in enumerate(unique_parents)}
    y_values = [parent_to_index[p] for p in parent_ids]

    ax.scatter(parent_timestamps, y_values)
    ax.set_ylabel("Parent (RLOC16)")
    ax.set_yticks(range(len(unique_parents)))
    ax.set_yticklabels(unique_parents)
    ax.set_title("Connected to Parent")
    ax.grid(True)


def plot_rss_and_loss(
    ax,
    metrics: LogMetrics,
    overall_pdr: Optional[float] = None,
) -> None:
    """
    Plot RSS over time and packet-loss vertical lines on the same Axes.

    If overall_pdr is provided, include it in this subplot's title.
    Also include:
      - nRSS: number of RSS samples
      - average RSS
      - standard deviation of RSS
    when available.
    """
    timestamps_rss = metrics.rss_timestamps
    rss_values = metrics.rss_values
    loss_timestamps = metrics.loss_timestamps
    n_rss = len(rss_values)

    plotted_any = False

    # Packet loss vertical lines
    if loss_timestamps:
        for i, ts_loss in enumerate(loss_timestamps):
            ax.axvline(
                ts_loss,
                linestyle="--",
                color="red",
                alpha=0.7,
                label="Packet loss" if i == 0 else None,
            )
        plotted_any = True
        
    # RSS scatter
    if timestamps_rss:
        ax.plot(
            timestamps_rss,
            rss_values,
            marker=".",
            linestyle="",
            label="RTT RSS",
        )
        plotted_any = True

    if not timestamps_rss:
        # Make it clear there's no RSS samples, just loss markers (if any)
        ax.text(
            0.5,
            0.5,
            "No RSS data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    if plotted_any:
        ax.legend()

    ax.set_ylabel("RSS (dBm)")

    # --- RSS statistics for title ---
    rss_mean_dbm: Optional[float] = None
    rss_std_db: Optional[float] = None
    if rss_values:
        n = len(rss_values)
        rss_mean_dbm = sum(rss_values) / n
        if n > 1:
            var = sum((v - rss_mean_dbm) ** 2 for v in rss_values) / n  # population variance
            rss_std_db = var ** 0.5
        else:
            rss_std_db = 0.0

    # Build title string
    title = "Ping to Parent RSS & Packet Loss"

    suffix_parts = []
    if overall_pdr is not None:
        suffix_parts.append(f"PDR={overall_pdr:.1f}%")
    if n_rss > 0:
        suffix_parts.append(f"nRSS={n_rss}")
    if rss_mean_dbm is not None and rss_std_db is not None:
        suffix_parts.append(f"avgRSS={rss_mean_dbm:.1f} dBm, stdRSS={rss_std_db:.1f} dB")

    if suffix_parts:
        title += " (" + ", ".join(suffix_parts) + ")"

    ax.set_title(title)
    ax.grid(True)

    if RSS_YLIM is not None:
        ax.set_ylim(*RSS_YLIM)


def process_log_file(log_path: str, rtt_by_file: Dict[str, List[float]]) -> None:
    # Work out paths/labels for this file
    log_path_obj = Path(log_path)
    data_dir_path = Path(DATA_DIR)

    rel_log_path = log_path_obj.relative_to(data_dir_path)

    graph_dir = Path(GRAPHS_DIR) / rel_log_path.parent
    graph_dir.mkdir(parents=True, exist_ok=True)

    label_for_file = str(rel_log_path)

    metrics = parse_log_file(log_path)

    # --- Summaries / RTT collection ---
    if not metrics.rtt_timestamps and not metrics.loss_timestamps:
        print(f"[SUMMARY] {log_path}: 0 RTT samples and no packet loss.")
    else:
        print(f"[SUMMARY] {log_path}: {len(metrics.rtt_timestamps)} RTT samples parsed.")
        print(f"[SUMMARY] {log_path}: {len(metrics.loss_timestamps)} packet loss event(s).")
        if metrics.rtt_timestamps:
            rtt_by_file[label_for_file] = metrics.rtt_avgs_ms

    if metrics.rss_timestamps:
        print(f"[SUMMARY] {log_path}: {len(metrics.rss_timestamps)} ping RSS sample(s) found.")
    else:
        print(f"[SUMMARY] {log_path}: no ping RSS data (len={PING_REPLY_LEN}) found.")

    if metrics.parent_timestamps:
        print(f"[SUMMARY] {log_path}: {len(metrics.parent_timestamps)} parent sample(s) parsed.")
    else:
        print(f"[SUMMARY] {log_path}: no valid parent data (RLOC16) found.")

    # Overall packet delivery rate for the file
    if metrics.total_tx > 0:
        overall_pdr = 100.0 * metrics.total_rx / metrics.total_tx
        print(
            f"[SUMMARY] {log_path}: Overall packet delivery rate = "
            f"{overall_pdr:.2f}% ({metrics.total_rx}/{metrics.total_tx})"
        )
    else:
        overall_pdr = None
        print(f"[SUMMARY] {log_path}: No packet summary lines found for PDR.")

    # RTT statistics (mean and std) over this file
    if metrics.rtt_avgs_ms:
        vals = metrics.rtt_avgs_ms
        n = len(vals)
        rtt_mean_ms = sum(vals) / n
        if n > 1:
            var = sum((v - rtt_mean_ms) ** 2 for v in vals) / n  # population variance
            rtt_std_ms = var ** 0.5
        else:
            rtt_std_ms = 0.0
        print(
            f"[SUMMARY] {log_path}: RTT mean = {rtt_mean_ms:.2f} ms, "
            f"std = {rtt_std_ms:.2f} ms"
        )
    else:
        rtt_mean_ms = None
        rtt_std_ms = None

    # If there's absolutely nothing to plot, skip.
    if (
        not metrics.rtt_timestamps
        and not metrics.loss_timestamps
        and not metrics.rss_timestamps
        and not metrics.parent_timestamps
    ):
        print(f"[INFO] {log_path}: no time-series data to plot; skipping figure.")
        return

    # --- Combined figure (RTT, RSS+loss, Parent stacked) ---
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(12, 8),
    )
    ax_rtt, ax_rss, ax_parent = axes

    plot_rtt(ax=ax_rtt, metrics=metrics, rtt_mean_ms=rtt_mean_ms, rtt_std_ms=rtt_std_ms)
    plot_rss_and_loss(ax=ax_rss, metrics=metrics, overall_pdr=overall_pdr)
    plot_parents(ax=ax_parent, metrics=metrics)

    ax_parent.set_xlabel("Time")
    fig.autofmt_xdate(rotation=45)

    # Top title: only the file label; stats appear in subplots
    suptitle_text = label_for_file
    fig.suptitle(suptitle_text, y=0.98)   # only place where filename appears
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_name = log_path_obj.stem + "_timeseries.png"
    out_path = graph_dir / out_name
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[OK] Saved combined time-series graph for {label_for_file} -> {out_path}")


def main():
    data_dir_path = Path(DATA_DIR)
    pattern = os.path.join(DATA_DIR, "**", "*.log")
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
        print(f"[INFO] No .log files found in {DATA_DIR}")
        return

    print(f"[INFO] Found {len(log_files)} .log file(s) in {DATA_DIR} (excluding dot-directories):")
    for lf in log_files:
        print(f"  - {lf}")

    rtt_by_file: Dict[str, List[float]] = {}

    for log_path in log_files:
        process_log_file(log_path, rtt_by_file)

    if not rtt_by_file:
        print("[INFO] No RTT data collected; skipping RTT box plot.")
        return

    print("[INFO] Creating RTT box plot comparing all files...")

    labels = list(rtt_by_file.keys())
    data = [rtt_by_file[label] for label in labels]

    plt.figure()
    plt.boxplot(
        data,
        labels=labels,
        showfliers=False,
    )
    plt.ylabel("RTT (ms)")
    plt.title("Ping to Parent Round-trip Time per File")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    boxplot_path = Path(GRAPHS_DIR) / "all_files_rtt_boxplot.png"
    plt.savefig(boxplot_path)
    plt.close()

    print(f"[OK] Saved RTT box plot -> {boxplot_path}")


if __name__ == "__main__":
    main()
