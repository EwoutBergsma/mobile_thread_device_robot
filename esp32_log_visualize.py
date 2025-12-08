import re
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os

# Directories (relative, no leading slash)
DATA_DIR = "data"
GRAPHS_DIR = "graphs"

# Ensure graphs directory exists
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

# Regex for parent lines, only accept hex addresses as parent IDs
# Example: [2025-12-04T17:52:42.400] Ext Addr: 4649555e5f62298c
parent_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]      # timestamp in brackets
    .*?Ext\ Addr:\s*
    (?P<addr>[0-9A-Fa-f]+)  # hex-only address
    """,
    re.VERBOSE
)

# Helper to parse timestamps like 2025-12-04T17:52:42.396
def parse_timestamp(ts_str: str) -> datetime:
    try:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")


def process_log_file(log_path: str, rtt_by_file: dict):
    print(f"\n[PROCESS] Starting file: {log_path}")

    timestamps_rtt = []
    avg_rtts_ms = []

    # Track times where packet loss is not 0.0%
    loss_timestamps = []

    parent_timestamps = []
    parent_ids = []

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
                    print(f"  [LOSS] Non-zero packet loss {loss_pct}% at {ts_loss_str}")
                    loss_timestamps.append(ts_loss)

            # --- RTT detection (only lines with Round-trip stats) ---
            m_rtt = rtt_re.search(line_stripped)
            if m_rtt:
                print(f"  [USE RTT] Line {line_no}: {line_stripped}")

                ts_str = m_rtt.group("ts")
                avg_str = m_rtt.group("avg")

                ts = parse_timestamp(ts_str)
                avg_ms = float(avg_str)

                timestamps_rtt.append(ts)
                avg_rtts_ms.append(avg_ms)

            # --- Parent detection (hex-only Ext Addr:) ---
            m_parent = parent_re.search(line_stripped)
            if m_parent:
                ts_str = m_parent.group("ts")
                ext_addr = m_parent.group("addr")

                # Extra sanity checks if you want (e.g. length >= 8)
                if len(ext_addr) < 8:
                    print(f"  [SKIP PARENT] Line {line_no}: suspicious short addr '{ext_addr}'")
                    continue

                ts = parse_timestamp(ts_str)

                print(f"  [USE PARENT] Line {line_no}: {line_stripped}")
                print(f"               -> Parent Ext Addr: {ext_addr}")

                parent_timestamps.append(ts)
                parent_ids.append(ext_addr)

    # --- RTT + packet-loss plotting / data collection ---
    if not timestamps_rtt and not loss_timestamps:
        print(f"[SUMMARY] {log_path}: 0 RTT samples and no non-zero packet loss.")
    else:
        print(f"[SUMMARY] {log_path}: {len(timestamps_rtt)} RTT samples parsed.")
        print(f"[SUMMARY] {log_path}: {len(loss_timestamps)} non-zero packet loss event(s).")

        filename = Path(log_path).name

        # Only include files with RTT for the boxplot
        if timestamps_rtt:
            rtt_by_file[filename] = avg_rtts_ms

        plt.figure()

        # Plot RTT as points (if any)
        if timestamps_rtt:
            plt.plot(
                timestamps_rtt,
                avg_rtts_ms,
                marker=".",
                linestyle="",
                label="RTT avg (ms)"
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
                        label="Non-zero packet loss"
                    )
                else:
                    plt.axvline(
                        ts_loss,
                        linestyle="--",
                        color="red",
                        alpha=0.7
                    )

        plt.xlabel("Time")
        plt.ylabel("RTT (ms)")
        plt.title(f"Ping RTT\n{filename}")
        plt.grid(True)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.legend()

        out_name = Path(filename).stem + "_rtt.png"
        out_path = Path(GRAPHS_DIR) / out_name
        plt.savefig(out_path)
        plt.close()

        print(f"[OK] Saved RTT graph for {filename} -> {out_path}")

    # --- Parent plotting ---
    if not parent_timestamps:
        print(f"[SUMMARY] {log_path}: no valid parent data (Ext Addr) found.")
        return

    print(f"[SUMMARY] {log_path}: {len(parent_timestamps)} parent sample(s) parsed.")

    filename = Path(log_path).name
    unique_parents = sorted(set(parent_ids))
    parent_to_index = {p: i for i, p in enumerate(unique_parents)}
    y_values = [parent_to_index[p] for p in parent_ids]

    plt.figure()
    plt.scatter(parent_timestamps, y_values)
    plt.xlabel("Time")
    plt.ylabel("Parent")
    plt.yticks(range(len(unique_parents)), unique_parents)
    # title without "over time"
    plt.title(f"Parent\n{filename}")
    plt.grid(True)
    # make time diagonal for timeseries
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    parents_out_name = Path(filename).stem + "_parents.png"
    parents_out_path = Path(GRAPHS_DIR) / parents_out_name
    plt.savefig(parents_out_path)
    plt.close()

    print(f"[OK] Saved parent graph for {filename} -> {parents_out_path}")


def main():
    log_files = glob.glob(str(Path(DATA_DIR) / "*.log"))

    if not log_files:
        print(f"[INFO] No .log files found in {DATA_DIR}")
        return

    print(f"[INFO] Found {len(log_files)} .log file(s) in {DATA_DIR}:")
    for lf in log_files:
        print(f"  - {lf}")

    rtt_by_file = {}

    for log_path in log_files:
        process_log_file(log_path, rtt_by_file)

    if not rtt_by_file:
        print("[INFO] No RTT data collected; skipping RTT box plot.")
        return

    print("[INFO] Creating RTT box plot comparing all files...")

    labels = list(rtt_by_file.keys())
    data = [rtt_by_file[label] for label in labels]

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.ylabel("RTT (ms)")
    plt.title("Ping RTT Boxplot per File")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    boxplot_path = Path(GRAPHS_DIR) / "all_files_rtt_boxplot.png"
    plt.savefig(boxplot_path)
    plt.close()

    print(f"[OK] Saved RTT box plot -> {boxplot_path}")


if __name__ == "__main__":
    main()
