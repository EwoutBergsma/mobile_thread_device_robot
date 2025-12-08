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

# Regex for lines like:
# [2025-12-04T17:54:36.720] 1 packets transmitted, 1 packets received. Packet loss = 0.0%. Round-trip min/avg/max = 427/427.000/427 ms.
line_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]               # [2025-12-04T17:54:36.720]
    .*?Round-trip\ min/avg/max\ =\   # Round-trip min/avg/max =
    (?P<min>\d+(?:\.\d+)?)/          # min
    (?P<avg>\d+(?:\.\d+)?)/          # avg
    (?P<max>\d+(?:\.\d+)?)\ ms\.?    # max ms.
    """,
    re.VERBOSE
)

def process_log_file(log_path: str, rtt_by_file: dict):
    print(f"\n[PROCESS] Starting file: {log_path}")

    timestamps = []
    avg_rtts_ms = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            m = line_re.search(line)
            if not m:
                continue

            # Print the line that is actually used for processing
            # print(f"  [USE] Line {line_no}: {line.rstrip()}")

            ts_str = m.group("ts")
            avg_str = m.group("avg")

            # parse timestamp like 2025-12-04T17:54:36.720
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                # fallback if no milliseconds
                ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")

            avg_ms = float(avg_str)

            timestamps.append(ts)
            avg_rtts_ms.append(avg_ms)

    if not timestamps:
        print(f"[SKIP] No ping RTT lines found in {log_path}")
        return

    print(f"[SUMMARY] {log_path}: {len(timestamps)} RTT samples parsed.")

    # Store RTT samples for this file for the global box plot later
    filename = Path(log_path).name
    rtt_by_file[filename] = avg_rtts_ms

    # Build and save per-file plot
    plt.figure()
    # Only dots, no line
    plt.plot(timestamps, avg_rtts_ms, marker=".", linestyle="")

    plt.xlabel("Time")
    plt.ylabel("RTT (ms)")

    # Newline before the filename in the title, and no "over Time"
    plt.title(f"Ping RTT\n{filename}")

    plt.grid(True)
    plt.tight_layout()

    out_name = Path(filename).stem + "_rtt.png"
    out_path = Path(GRAPHS_DIR) / out_name
    plt.savefig(out_path)
    plt.close()

    print(f"[OK] Saved graph for {filename} -> {out_path}")

def main():
    log_files = glob.glob(str(Path(DATA_DIR) / "*.log"))

    if not log_files:
        print(f"[INFO] No .log files found in {DATA_DIR}")
        return

    print(f"[INFO] Found {len(log_files)} .log file(s) in {DATA_DIR}:")
    for lf in log_files:
        print(f"  - {lf}")

    # Collect RTT samples per file for box plot
    rtt_by_file = {}

    for log_path in log_files:
        process_log_file(log_path, rtt_by_file)

    # Create a box plot comparing all files
    if not rtt_by_file:
        print("[INFO] No RTT data collected; skipping box plot.")
        return

    print("[INFO] Creating box plot comparing all files...")

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

    print(f"[OK] Saved box plot -> {boxplot_path}")

if __name__ == "__main__":
    main()
