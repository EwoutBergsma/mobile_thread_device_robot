import re
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
import numpy as np

# Directories (relative)
DATA_DIR = "data"
GRAPHS_DIR = "graphs"

os.makedirs(GRAPHS_DIR, exist_ok=True)

# RTT lines
rtt_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]               
    .*?Round-trip\ min/avg/max\ =\   
    (?P<min>\d+(?:\.\d+)?)/          
    (?P<avg>\d+(?:\.\d+)?)/          
    (?P<max>\d+(?:\.\d+)?)\ ms\.?    
    """,
    re.VERBOSE,
)

# Parent lines (Ext Addr)
parent_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]      
    .*?Ext\ Addr:\s*
    (?P<addr>[0-9A-Fa-f]+)  
    """,
    re.VERBOSE,
)

# AttachState lines for parent search intervals
attach_re = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]      
    .*?AttachState\s+
    (?P<from_state>\S+)\s*->\s*(?P<to_state>\S+)
    """,
    re.VERBOSE,
)


def parse_timestamp(ts_str: str) -> datetime:
    try:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")


def process_log_file(log_path: str,
                     rtt_by_file: dict,
                     rtt_near_by_file: dict,
                     rtt_bg_by_file: dict):
    print(f"\n[PROCESS] Starting file: {log_path}")

    timestamps_rtt = []
    avg_rtts_ms = []

    parent_timestamps = []
    parent_ids = []

    # search intervals: list of (start_ts, end_ts)
    search_intervals = []
    search_active = False
    search_start_ts = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")

            # --- RTT detection ---
            m_rtt = rtt_re.search(line)
            if m_rtt:
                ts_str = m_rtt.group("ts")
                avg_str = m_rtt.group("avg")
                ts = parse_timestamp(ts_str)
                avg_ms = float(avg_str)

                timestamps_rtt.append(ts)
                avg_rtts_ms.append(avg_ms)

                print(f"  [USE RTT] Line {line_no}: {line}")

            # --- Parent detection ---
            m_parent = parent_re.search(line)
            if m_parent:
                ts_str = m_parent.group("ts")
                ext_addr = m_parent.group("addr")
                if len(ext_addr) < 8:
                    print(f"  [SKIP PARENT] Line {line_no}: suspicious short addr '{ext_addr}'")
                    continue

                ts = parse_timestamp(ts_str)
                parent_timestamps.append(ts)
                parent_ids.append(ext_addr)

                print(f"  [USE PARENT] Line {line_no}: {line}")
                print(f"               -> Parent Ext Addr: {ext_addr}")

            # --- Attach state detection for parent search intervals ---
            m_attach = attach_re.search(line)
            if m_attach:
                ts_str = m_attach.group("ts")
                from_state = m_attach.group("from_state")
                to_state = m_attach.group("to_state")
                ts = parse_timestamp(ts_str)

                print(f"  [ATTACH] Line {line_no}: {from_state} -> {to_state}")

                # Heuristic: search active from Start->ParentReq until ParentReq->Idle
                if to_state == "ParentReq" and not search_active:
                    search_active = True
                    search_start_ts = ts
                    print(f"    [SEARCH START] {ts}")

                if from_state == "ParentReq" and to_state == "Idle" and search_active:
                    search_intervals.append((search_start_ts, ts))
                    print(f"    [SEARCH END]   {ts} (interval {search_start_ts} -> {ts})")
                    search_active = False
                    search_start_ts = None

    # Close any unterminated search interval at last RTT timestamp (if present)
    if search_active and timestamps_rtt:
        end_ts = timestamps_rtt[-1]
        search_intervals.append((search_start_ts, end_ts))
        print(f"    [SEARCH END @EOF] {end_ts} (interval {search_start_ts} -> {end_ts})")
        search_active = False
        search_start_ts = None

    filename = Path(log_path).name

    # --- RTT stuff ---
    if not timestamps_rtt:
        print(f"[SUMMARY] {log_path}: 0 RTT samples parsed.")
    else:
        print(f"[SUMMARY] {log_path}: {len(timestamps_rtt)} RTT samples parsed.")
        rtt_by_file[filename] = avg_rtts_ms

        # Per-file RTT scatter
        plt.figure()
        plt.plot(timestamps_rtt, avg_rtts_ms, marker=".", linestyle="")
        plt.xlabel("Time")
        plt.ylabel("RTT (ms)")
        plt.title(f"Ping RTT\n{filename}")
        plt.grid(True)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path = Path(GRAPHS_DIR) / f"{Path(filename).stem}_rtt.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] Saved RTT graph for {filename} -> {out_path}")

    # --- classify RTTs as near-search vs background using intervals ---
    near_search_rtts = []
    background_rtts = []

    if timestamps_rtt and search_intervals:
        # sort intervals for safety
        search_intervals.sort(key=lambda ab: ab[0])

        def is_in_interval(ts):
            for start, end in search_intervals:
                if start <= ts <= end:
                    return True
                if ts < start:
                    # intervals are sorted; can stop early
                    return False
            return False

        for ts_rtt, rtt in zip(timestamps_rtt, avg_rtts_ms):
            if is_in_interval(ts_rtt):
                near_search_rtts.append(rtt)
            else:
                background_rtts.append(rtt)
    else:
        # no search intervals: everything is background
        background_rtts = avg_rtts_ms.copy() if timestamps_rtt else []

    if near_search_rtts:
        rtt_near_by_file[filename] = near_search_rtts
        print(f"[SUMMARY] {filename}: {len(near_search_rtts)} RTT samples NEAR parent search.")
    if background_rtts:
        rtt_bg_by_file[filename] = background_rtts
        print(f"[SUMMARY] {filename}: {len(background_rtts)} RTT samples in BACKGROUND.")

    # per-file boxplot: near-search vs background
    if near_search_rtts and background_rtts:
        plt.figure()
        plt.boxplot(
            [near_search_rtts, background_rtts],
            labels=["Near parent search", "Background"],
            showfliers=True,
        )
        plt.ylabel("RTT (ms)")
        plt.title(f"RTT vs Parent Search\n{filename}")
        plt.tight_layout()
        vs_path = Path(GRAPHS_DIR) / f"{Path(filename).stem}_rtt_search_vs_background.png"
        plt.savefig(vs_path)
        plt.close()
        print(f"[OK] Saved RTT vs parent-search boxplot for {filename} -> {vs_path}")

    # --- Parent plot ---
    if not parent_timestamps:
        print(f"[SUMMARY] {log_path}: no valid parent data (Ext Addr) found.")
        return

    print(f"[SUMMARY] {log_path}: {len(parent_timestamps)} parent sample(s) parsed.")
    unique_parents = sorted(set(parent_ids))
    parent_to_index = {p: i for i, p in enumerate(unique_parents)}
    y_values = [parent_to_index[p] for p in parent_ids]

    plt.figure()
    plt.scatter(parent_timestamps, y_values)
    plt.xlabel("Time")
    plt.ylabel("Parent")
    plt.yticks(range(len(unique_parents)), unique_parents)
    plt.title(f"Parent\n{filename}")
    plt.grid(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    parents_path = Path(GRAPHS_DIR) / f"{Path(filename).stem}_parents.png"
    plt.savefig(parents_path)
    plt.close()
    print(f"[OK] Saved parent graph for {filename} -> {parents_path}")


def main():
    log_files = glob.glob(str(Path(DATA_DIR) / "*.log"))

    if not log_files:
        print(f"[INFO] No .log files found in {DATA_DIR}")
        return

    print(f"[INFO] Found {len(log_files)} .log file(s) in {DATA_DIR}:")
    for lf in log_files:
        print(f"  - {lf}")

    rtt_by_file = {}
    rtt_near_by_file = {}
    rtt_bg_by_file = {}

    for log_path in log_files:
        process_log_file(log_path, rtt_by_file, rtt_near_by_file, rtt_bg_by_file)

    # ---- print stats ----
    if rtt_by_file:
        print("\n[STATS] RTT per file (all samples):")
        for fname, values in rtt_by_file.items():
            arr = np.array(values)
            print(
                f"  {fname}: "
                f"n={len(arr)}, "
                f"median={np.median(arr):.1f} ms, "
                f"mean={np.mean(arr):.1f} ms, "
                f"p90={np.percentile(arr, 90):.1f} ms, "
                f"max={np.max(arr):.1f} ms"
            )

    if rtt_near_by_file:
        print("\n[STATS] RTT NEAR parent search per file:")
        for fname, values in rtt_near_by_file.items():
            arr = np.array(values)
            print(
                f"  {fname}: "
                f"n={len(arr)}, "
                f"median={np.median(arr):.1f} ms, "
                f"mean={np.mean(arr):.1f} ms, "
                f"p90={np.percentile(arr, 90):.1f} ms, "
                f"max={np.max(arr):.1f} ms"
            )

    if rtt_bg_by_file:
        print("\n[STATS] RTT BACKGROUND per file:")
        for fname, values in rtt_bg_by_file.items():
            arr = np.array(values)
            print(
                f"  {fname}: "
                f"n={len(arr)}, "
                f"median={np.median(arr):.1f} ms, "
                f"mean={np.mean(arr):.1f} ms, "
                f"p90={np.percentile(arr, 90):.1f} ms, "
                f"max={np.max(arr):.1f} ms"
            )

    # ---- global RTT boxplot per file ----
    if not rtt_by_file:
        print("[INFO] No RTT data collected; skipping RTT box plot.")
        return

    print("\n[INFO] Creating RTT box plot comparing all files...")

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
