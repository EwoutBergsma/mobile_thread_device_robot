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
# From your snippet: len:56 for echo reply.
PING_REPLY_LEN = 56

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

# Regex for ANY packet-loss line (0.0%, 10.0%, 100.0%, etc.)
# Example: [2025-12-04T17:52:44.956] 1 packets transmitted, 0 packets received. Packet loss = 100.0%.
loss_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]          # timestamp in brackets
    .*?Packet\ loss\s*=\s*
    (?P<loss>\d+(?:\.\d+)?)%    # e.g. 0.0, 10.0, 100.0
    """,
    re.VERBOSE
)

# Regex for parent lines, using RLOC16 as parent IDs
# Example: [2025-12-09T12:42:59.089] Rloc: c00
parent_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]      # timestamp in brackets
    .*?Rloc:\s*
    (?P<rloc>[0-9A-Fa-f]+)  # hex-only RLOC16 value
    """,
    re.VERBOSE
)

# Regex for OT state lines like:
# [ts] child
# [ts] detached
# [ts] router
# [ts] leader
state_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]              # timestamp
    .*?\b(?P<state>disabled|detached|child|router|leader)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Regex for *ping reply* RSS lines:
# Example:
# [2025-12-09T11:20:47.453] I(149610) OPENTHREAD:[I] MeshForwarder-:
#   Received IPv6 ICMP6 msg, len:56, chksum:c1d2, ecn:no, from:0x7000, sec:yes, prio:normal, rss:-90.0
#
# Or UDP variant:
#   Received IPv6 UDP msg, len:81, ...
#
# We don't care if it's UDP or ICMP6; we filter by len == PING_REPLY_LEN.
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

    # RSS samples (ONLY ping replies with len == PING_REPLY_LEN)
    rss_timestamps: List[datetime]
    rss_values: List[float]

    # State events
    state_timestamps: List[datetime]
    states: List[str]

    # Derived "effective parent" timeline (state + parent combined)
    eff_parent_timestamps: List[datetime]
    eff_parents: List[str]


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

    # Map timestamps -> parents / states
    parents_at_ts: Dict[datetime, List[str]] = defaultdict(list)
    for ts, pid in zip(metrics.parent_timestamps, metrics.parent_ids):
        parents_at_ts[ts].append(pid)

    states_at_ts: Dict[datetime, List[str]] = defaultdict(list)
    for ts, st in zip(metrics.state_timestamps, metrics.states):
        states_at_ts[ts].append(st)

    # Union of all timestamps where anything happens
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
        # Update state / parent if we have new info at this timestamp
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
    Read a log file and extract:
      - RTT timestamps and averages
      - packet-loss timestamps (loss > 0)
      - parent timestamps and RLOC16 values
      - OT state changes
      - RSS samples *only* for ping replies:
        MeshForwarder "Received IPv6 <type> msg" lines where len == PING_REPLY_LEN
      - derived effective-parent timeline (state + parent combined)
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
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line_stripped = line.rstrip("\n")

            # --- Packet loss detection (any loss != 0.0%) ---
            m_loss = loss_re.search(line_stripped)
            if m_loss:
                ts_loss_str = m_loss.group("ts")
                loss_pct_str = m_loss.group("loss")
                ts_loss = parse_timestamp(ts_loss_str)
                loss_pct = float(loss_pct_str)

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

            # --- State detection (child/detached/router/leader/disabled) ---
            m_state = state_re.search(line_stripped)
            if m_state:
                ts_str = m_state.group("ts")
                state_str = m_state.group("state")
                ts = parse_timestamp(ts_str)

                state_norm = state_str.strip().lower()
                print(f"  [STATE] {state_norm} at {ts_str}")

                metrics.state_timestamps.append(ts)
                metrics.states.append(state_norm)

            # --- RSS detection for ping replies (MeshForwarder "Received IPv6 <type> msg" lines) ---
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

    # Build effective-parent timeline once all events are collected
    build_parent_timeline(metrics)

    return metrics


def plot_rtt_and_loss(
    label_for_file: str,
    metrics: LogMetrics,
    out_path: Path,
) -> None:
    timestamps_rtt = metrics.rtt_timestamps
    avg_rtts_ms = metrics.rtt_avgs_ms
    loss_timestamps = metrics.loss_timestamps

    plt.figure()

    # Plot RTT as points (if any)
    if timestamps_rtt:
        plt.plot(
            timestamps_rtt,
            avg_rtts_ms,
            marker=".",
            linestyle="",
            label="RTT avg (ms)",
        )

    # Vertical dashed lines where packet loss > 0 (including 100%)
    if loss_timestamps:
        for i, ts_loss in enumerate(loss_timestamps):
            if i == 0:
                # First one gets the label for the legend
                plt.axvline(
                    ts_loss,
                    linestyle="--",
                    color="red",
                    alpha=0.7,
                    label="Packet loss",
                )
            else:
                plt.axvline(
                    ts_loss,
                    linestyle="--",
                    color="red",
                    alpha=0.7,
                )

    plt.xlabel("Time")
    plt.ylabel("RTT (ms)")
    plt.title(f"Ping RTT\n{label_for_file}")
    plt.grid(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_parents(
    label_for_file: str,
    metrics: LogMetrics,
    out_path: Path,
) -> None:
    """
    Plot parent over time using the *effective* parent:
      - 'detached' / blank state -> 'No Parent'
      - parent RLOC16 otherwise
    If the effective timeline is missing, falls back to raw parent events.
    """
    # Prefer the derived effective timeline
    if metrics.eff_parent_timestamps:
        parent_timestamps = metrics.eff_parent_timestamps
        parent_ids = metrics.eff_parents
    else:
        parent_timestamps = metrics.parent_timestamps
        parent_ids = metrics.parent_ids

    if not parent_timestamps:
        return

    unique_parents = sorted(set(parent_ids))
    parent_to_index = {p: i for i, p in enumerate(unique_parents)}
    y_values = [parent_to_index[p] for p in parent_ids]

    plt.figure()
    plt.scatter(parent_timestamps, y_values)
    plt.xlabel("Time")
    plt.ylabel("Parent (effective)")
    plt.yticks(range(len(unique_parents)), unique_parents)
    plt.title(f"Parent\n{label_for_file}")
    plt.grid(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rss(
    label_for_file: str,
    metrics: LogMetrics,
    out_path: Path,
) -> None:
    """
    Plot RSS over time using the extracted ping-reply RSS samples.
    """
    timestamps_rss = metrics.rss_timestamps
    rss_values = metrics.rss_values

    if not timestamps_rss:
        return

    plt.figure()
    plt.plot(
        timestamps_rss,
        rss_values,
        marker=".",
        linestyle="",
        label=f"Ping RSS (len={PING_REPLY_LEN})",
    )
    plt.xlabel("Time")
    plt.ylabel("RSS (dBm)")
    plt.title(f"Ping RSS over Time\n{label_for_file}")
    plt.grid(True)
    plt.xticks(rotation=45, ha="right")

    # Optional: invert y-axis so stronger (less negative) RSS is "higher" visually
    # plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def process_log_file(log_path: str, rtt_by_file: Dict[str, List[float]]) -> None:
    # Work out paths/labels for this file
    log_path_obj = Path(log_path)
    data_dir_path = Path(DATA_DIR)

    # path of this log relative to DATA_DIR (e.g. "subdir1/subdir2/file.log")
    rel_log_path = log_path_obj.relative_to(data_dir_path)

    # directory under GRAPHS_DIR that mirrors the data structure (e.g. "graphs/subdir1/subdir2")
    graph_dir = Path(GRAPHS_DIR) / rel_log_path.parent
    graph_dir.mkdir(parents=True, exist_ok=True)

    # label for plots/boxplot: use relative path to avoid collisions
    label_for_file = str(rel_log_path)

    metrics = parse_log_file(log_path)

    # --- RTT + packet-loss summary / data collection ---
    if not metrics.rtt_timestamps and not metrics.loss_timestamps:
        print(f"[SUMMARY] {log_path}: 0 RTT samples and no packet loss.")
    else:
        print(f"[SUMMARY] {log_path}: {len(metrics.rtt_timestamps)} RTT samples parsed.")
        print(f"[SUMMARY] {log_path}: {len(metrics.loss_timestamps)} packet loss event(s).")

        # Only include files with RTT for the boxplot
        if metrics.rtt_timestamps:
            rtt_by_file[label_for_file] = metrics.rtt_avgs_ms

        out_name = log_path_obj.stem + "_rtt.png"
        out_path = graph_dir / out_name

        plot_rtt_and_loss(
            label_for_file=label_for_file,
            metrics=metrics,
            out_path=out_path,
        )

        print(f"[OK] Saved RTT graph for {label_for_file} -> {out_path}")

    # --- RSS plotting (ping replies only) ---
    if metrics.rss_timestamps:
        rss_out_name = log_path_obj.stem + "_rss.png"
        rss_out_path = graph_dir / rss_out_name

        plot_rss(
            label_for_file=label_for_file,
            metrics=metrics,
            out_path=rss_out_path,
        )

        print(f"[OK] Saved RSS graph for {label_for_file} -> {rss_out_path}")
    else:
        print(f"[SUMMARY] {log_path}: no ping RSS data (len={PING_REPLY_LEN}) found.")

    # --- Parent plotting ---
    if not metrics.parent_timestamps:
        print(f"[SUMMARY] {log_path}: no valid parent data (RLOC16) found.")
        return

    print(f"[SUMMARY] {log_path}: {len(metrics.parent_timestamps)} parent sample(s) parsed.")

    parents_out_name = log_path_obj.stem + "_parents.png"
    parents_out_path = graph_dir / parents_out_name

    plot_parents(
        label_for_file=label_for_file,
        metrics=metrics,
        out_path=parents_out_path,
    )

    print(f"[OK] Saved parent graph for {label_for_file} -> {parents_out_path}")


def main():
    # Collect all .log files under DATA_DIR using glob, skipping any directory
    # that starts with '.' (same behavior as the previous os.walk version).
    data_dir_path = Path(DATA_DIR)
    pattern = os.path.join(DATA_DIR, "**", "*.log")
    all_candidates = glob(pattern, recursive=True)

    log_files: List[str] = []
    for path_str in all_candidates:
        p = Path(path_str)

        # Get parts relative to DATA_DIR, e.g. ("subdir1", "subdir2", "file.log")
        try:
            rel_parts = p.relative_to(data_dir_path).parts
        except ValueError:
            # Shouldn't happen for this pattern, but be safe
            continue

        # Check only directory parts (exclude the filename)
        dir_parts = rel_parts[:-1]
        if any(part.startswith(".") for part in dir_parts):
            # Skip any file that lives in a dot-directory
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
        showfliers=False,  # <- hide outliers from the boxplot
    )
    plt.ylabel("RTT (ms)")
    plt.title("Ping RTT Boxplot per File")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Keep the summary boxplot at the root of GRAPHS_DIR
    boxplot_path = Path(GRAPHS_DIR) / "all_files_rtt_boxplot.png"
    plt.savefig(boxplot_path)
    plt.close()

    print(f"[OK] Saved RTT box plot -> {boxplot_path}")


if __name__ == "__main__":
    main()
