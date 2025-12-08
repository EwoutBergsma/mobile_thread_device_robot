import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


# ---------- Regexes ----------

# Match lines like:
# [2025-12-04T17:52:34.649] some text...
TIMESTAMP_RE = re.compile(r'^\[(?P<ts>[^\]]+)\]\s*(?P<msg>.*)$')

# Ping RTT, e.g. "time=239ms"
RTT_RE = re.compile(r'time=(?P<rtt>[0-9]+(?:\.[0-9]+)?)ms')

# RSS from MeshForwarder logs, e.g. "rss:-71.0"
RSS_RE = re.compile(r'rss:(?P<rss>-?[0-9]+(?:\.[0-9]+)?)')

# Parent command fields
PARENT_RLOC_RE = re.compile(r'Rloc:\s*([0-9a-fA-F]+)')
LQI_IN_RE = re.compile(r'Link Quality In:\s*([0-9]+)')
LQI_OUT_RE = re.compile(r'Link Quality Out:\s*([0-9]+)')
AGE_RE = re.compile(r'Age:\s*([0-9]+)')

# MAC counters from "ot counters mac"
TX_TOTAL_RE = re.compile(r'TxTotal:\s*([0-9]+)', re.IGNORECASE)
RX_TOTAL_RE = re.compile(r'RxTotal:\s*([0-9]+)', re.IGNORECASE)
TX_RETRY_RE = re.compile(r'TxRetry:\s*([0-9]+)', re.IGNORECASE)
TX_ERR_CCA_RE = re.compile(r'TxErrCca:\s*([0-9]+)', re.IGNORECASE)
RX_ERR_FCS_RE = re.compile(r'RxErrFcs:\s*([0-9]+)', re.IGNORECASE)


def parse_ot_log(path: Path) -> pd.DataFrame:

    records = []

    last_cmd = None
    current_parent_block = None
    current_counter_block = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            m = TIMESTAMP_RE.match(line)
            if not m:
                continue

            ts_str = m.group("ts")
            msg = m.group("msg").strip()

            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                # Ignore weird timestamps
                continue

            # Normalise potential command line
            cmd_candidate = msg
            if cmd_candidate.startswith(">"):
                cmd_candidate = cmd_candidate[1:].strip()

            # ----- Detect command start -----
            if cmd_candidate in ("state", "parent") or cmd_candidate.startswith("ot counters mac") or cmd_candidate.startswith("counters mac"):
                if cmd_candidate == "state":
                    last_cmd = "state"
                    continue
                elif cmd_candidate == "parent":
                    last_cmd = "parent"
                    current_parent_block = {
                        "timestamp": ts,
                        "parent_rloc16": None,
                        "lqi_in": None,
                        "lqi_out": None,
                        "age_s": None,
                    }
                    continue
                else:
                    # ot counters mac
                    last_cmd = "counters_mac"
                    current_counter_block = {
                        "timestamp": ts,
                        "tx_total": None,
                        "rx_total": None,
                        "tx_retry": None,
                        "tx_err_cca": None,
                        "rx_err_fcs": None,
                    }
                    continue

            # ----- Command terminator -----
            if msg == "Done":
                if last_cmd == "parent" and current_parent_block is not None and current_parent_block.get("parent_rloc16"):
                    records.append(current_parent_block.copy())
                    current_parent_block = None
                if last_cmd == "counters_mac" and current_counter_block is not None:
                    records.append(current_counter_block.copy())
                    current_counter_block = None
                last_cmd = None
                continue

            # ----- state command output -----
            if last_cmd == "state":
                # Lines like: "child", "detached"
                if msg and msg not in (">",):
                    records.append({"timestamp": ts, "state": msg})
                continue

            # ----- parent command output -----
            if last_cmd == "parent" and current_parent_block is not None:
                m_rloc = PARENT_RLOC_RE.search(msg)
                if m_rloc:
                    current_parent_block["parent_rloc16"] = m_rloc.group(1)

                m_lqi_in = LQI_IN_RE.search(msg)
                if m_lqi_in:
                    current_parent_block["lqi_in"] = int(m_lqi_in.group(1))

                m_lqi_out = LQI_OUT_RE.search(msg)
                if m_lqi_out:
                    current_parent_block["lqi_out"] = int(m_lqi_out.group(1))

                m_age = AGE_RE.search(msg)
                if m_age:
                    current_parent_block["age_s"] = int(m_age.group(1))

                continue

            # ----- MAC counters output -----
            if last_cmd == "counters_mac" and current_counter_block is not None:
                m_tx_total = TX_TOTAL_RE.search(msg)
                if m_tx_total:
                    current_counter_block["tx_total"] = int(m_tx_total.group(1))

                m_rx_total = RX_TOTAL_RE.search(msg)
                if m_rx_total:
                    current_counter_block["rx_total"] = int(m_rx_total.group(1))

                m_tx_retry = TX_RETRY_RE.search(msg)
                if m_tx_retry:
                    current_counter_block["tx_retry"] = int(m_tx_retry.group(1))

                m_tx_err = TX_ERR_CCA_RE.search(msg)
                if m_tx_err:
                    current_counter_block["tx_err_cca"] = int(m_tx_err.group(1))

                m_rx_err = RX_ERR_FCS_RE.search(msg)
                if m_rx_err:
                    current_counter_block["rx_err_fcs"] = int(m_rx_err.group(1))

                continue

            # ----- RTT / RSS from ping / MeshForwarder -----
            m_rtt = RTT_RE.search(msg)
            m_rss = RSS_RE.search(msg)
            if m_rtt or m_rss:
                rec = {"timestamp": ts}
                if m_rtt:
                    rec["rtt_ms"] = float(m_rtt.group("rtt"))
                if m_rss:
                    rec["rss_dbm"] = float(m_rss.group("rss"))
                records.append(rec)
                continue

    if not records:
        raise RuntimeError(f"No useful entries found in log: {path}")

    df = pd.DataFrame.from_records(records)

    # Merge multiple rows with same timestamp by taking first non-null per column
    def first_non_null(series: pd.Series):
        non_null = series.dropna()
        return non_null.iloc[0] if not non_null.empty else pd.NA

    df = (
        df.sort_values("timestamp")
          .groupby("timestamp", as_index=False)
          .agg(first_non_null)
    )

    # Forward-fill state and parent so background bands are continuous
    if "state" not in df.columns:
        df["state"] = pd.NA
    if "parent_rloc16" not in df.columns:
        df["parent_rloc16"] = pd.NA

    df["state"] = df["state"].ffill()
    df["parent_rloc16"] = df["parent_rloc16"].ffill().fillna("none")

    # Ensure numeric types where appropriate
    numeric_cols = [
        "lqi_in", "lqi_out", "age_s",
        "rtt_ms", "rss_dbm",
        "tx_total", "rx_total",
        "tx_retry", "tx_err_cca", "rx_err_fcs",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------- Small helpers ----------

def safe_get_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        obj = df[col]
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        return obj
    return pd.Series(dtype=float)


def has_data(s: pd.Series) -> bool:
    return isinstance(s, pd.Series) and not s.empty and s.notna().any()


def mark_parent_periods(ax, df: pd.DataFrame, alpha: float = 0.55, zorder: float = 0.02):

    if "timestamp" not in df.columns or df.empty:
        return

    def is_blank_state(v):
        if pd.isna(v):
            return True
        s = str(v).strip().lower()
        return s == "" or s == "blank"

    def effective_parent(row):
        parent = str(row.get("parent_rloc16", "none")).strip()
        st = str(row.get("state", "")).strip().lower()

        if is_blank_state(row.get("state")) or st == "detached":
            return "No Parent"
        if parent == "" or parent.lower() in ("none", "nan"):
            return "No Parent"
        return parent

    eff = df.apply(effective_parent, axis=1)
    uniq = list(pd.unique(eff))

    no_parent_color = "#f8c8cf"  # light pink
    real_parents = [u for u in uniq if u != "No Parent"]
    base = plt.get_cmap("tab20", max(len(real_parents), 1))

    color_map = {"No Parent": no_parent_color}
    for idx, p in enumerate(real_parents):
        color_map[p] = base(idx)

    cur = None
    start = None

    for idx, row in df.iterrows():
        cur_eff = eff.loc[idx]
        if cur_eff != cur:
            if cur is not None and start is not None:
                label = cur
                _, labels = ax.get_legend_handles_labels()
                ax.axvspan(
                    start,
                    row["timestamp"],
                    color=color_map.get(cur, "#dddddd"),
                    alpha=alpha,
                    zorder=zorder,
                    label=label if label not in labels else "",
                )
            start = row["timestamp"]
            cur = cur_eff

    if cur is not None and start is not None:
        label = cur
        _, labels = ax.get_legend_handles_labels()
        ax.axvspan(
            start,
            df["timestamp"].iloc[-1],
            color=color_map.get(cur, "#dddddd"),
            alpha=alpha,
            zorder=zorder,
            label=label if label not in labels else "",
        )


# ---------- Plotting helpers ----------

def _finalize_figure(fig, output_path: Path | None):
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------- Plotting functions ----------

def plot_lqi_rtt_rss(df: pd.DataFrame, title: str | None = None, output_path: Path | None = None):
    """
    LQI In/Out (blue/cyan) + RTT (red) + smoothed RSS (purple dashed),
    with parent background.
    """
    t = safe_get_series(df, "timestamp")
    lqi_in = safe_get_series(df, "lqi_in")
    lqi_out = safe_get_series(df, "lqi_out")
    rtt = safe_get_series(df, "rtt_ms")
    rss = safe_get_series(df, "rss_dbm")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # LQI on left axis
    if has_data(lqi_in):
        ax1.plot(t, lqi_in, label="LQI In", color="blue")
    if has_data(lqi_out):
        ax1.plot(t, lqi_out, label="LQI Out", color="cyan")

    if has_data(lqi_in) or has_data(lqi_out):
        ax1.set_ylabel("LQI (0–3)")
        ax1.set_ylim(-0.5, 3.5)
        ax1.tick_params(axis="y", labelcolor="blue")

    # RTT on right axis
    ax2 = ax1.twinx()
    if has_data(rtt):
        ax2.plot(t, rtt, label="RTT (ms)", color="red", alpha=0.7)
        ax2.set_ylabel("RTT (ms)")
        ax2.tick_params(axis="y", labelcolor="red")

    # Smoothed RSS on right axis
    if has_data(rss):
        tmp = pd.DataFrame({"timestamp": t, "rss_dbm": rss}).sort_values("timestamp")
        tmp["rss_smooth"] = tmp["rss_dbm"].rolling(
            window=30,  # 30-sample moving average
            min_periods=1,
            center=True,
        ).mean()

        ax2.plot(
            tmp["timestamp"],
            tmp["rss_smooth"],
            label="RSS (dBm, smoothed)",
            linestyle="--",
            color="purple",
            alpha=0.8,
        )

    # Background bands
    mark_parent_periods(ax1, df)

    if title:
        fig.suptitle(title + " – LQI/RTT/RSS")
    else:
        fig.suptitle("LQI, RTT, RSS with Parent Background")

    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = []
    labels = []
    seen = set()
    for h, l in list(zip(handles1 + handles2, labels1 + labels2)):
        if l and l not in seen:
            handles.append(h)
            labels.append(l)
            seen.add(l)
    if handles:
        fig.legend(handles, labels, loc="upper right")

    _finalize_figure(fig, output_path)


def plot_age(df: pd.DataFrame, title: str | None = None, output_path: Path | None = None):
    """Parent Age over time, with parent background."""
    t = safe_get_series(df, "timestamp")
    age = safe_get_series(df, "age_s")

    if not has_data(age):
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, age, label="Age (s)", color="tab:blue")

    mark_parent_periods(ax, df)

    ax.set_ylabel("Parent Age (s)")
    ax.set_title((title or "") + " – Parent Age over Time")
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    _finalize_figure(fig, output_path)


def plot_counters(df: pd.DataFrame, title: str | None = None, output_path: Path | None = None):

    t = safe_get_series(df, "timestamp")
    tx_total = safe_get_series(df, "tx_total")
    rx_total = safe_get_series(df, "rx_total")
    tx_err_cca = safe_get_series(df, "tx_err_cca")
    tx_retry = safe_get_series(df, "tx_retry")
    rx_err_fcs = safe_get_series(df, "rx_err_fcs")

    if not (has_data(tx_total) or has_data(rx_total) or has_data(tx_err_cca) or has_data(tx_retry) or has_data(rx_err_fcs)):
        return

    fig, ax3 = plt.subplots(figsize=(12, 6))

    if has_data(tx_total):
        ax3.plot(t, tx_total, label="TX Total")
    if has_data(rx_total):
        ax3.plot(t, rx_total, label="RX Total")

    ax3.set_ylabel("Total Packets")
    ax3.tick_params(axis="y", labelcolor="tab:blue")

    ax4 = ax3.twinx()
    if has_data(tx_err_cca):
        ax4.plot(t, tx_err_cca, label="TX Err CCA", color="orange")
    if has_data(tx_retry):
        ax4.plot(t, tx_retry, label="TX Retry", color="red")
    if has_data(rx_err_fcs):
        ax4.plot(t, rx_err_fcs, label="RX Err FCS", color="purple")

    ax4.set_ylabel("Error Counts")
    ax4.tick_params(axis="y", labelcolor="tab:orange")

    mark_parent_periods(ax3, df)

    fig.suptitle((title or "") + " – MAC Counters over Time")
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)

    handles1, labels1 = ax3.get_legend_handles_labels()
    handles2, labels2 = ax4.get_legend_handles_labels()
    handles = []
    labels = []
    seen = set()
    for h, l in list(zip(handles1 + handles2, labels1 + labels2)):
        if l and l not in seen:
            handles.append(h)
            labels.append(l)
            seen.add(l)
    if handles:
        fig.legend(handles, labels, loc="upper right")

    _finalize_figure(fig, output_path)


def plot_predictive(df: pd.DataFrame, title: str | None = None, output_path: Path | None = None):

    if "timestamp" not in df.columns or df.empty:
        return

    df_roll = df.set_index("timestamp")

    lqi_in = df_roll.get("lqi_in")
    rtt = df_roll.get("rtt_ms")

    if lqi_in is None and rtt is None:
        return

    window = 20  # samples
    lqi_mean = lqi_in.rolling(window=window, min_periods=5).mean() if lqi_in is not None else None
    lqi_std = lqi_in.rolling(window=window, min_periods=5).std() if lqi_in is not None else None
    rtt_mean = rtt.rolling(window=window, min_periods=5).mean() if rtt is not None else None

    t = df_roll.index

    fig, ax5 = plt.subplots(figsize=(12, 6))

    if lqi_mean is not None and has_data(lqi_mean):
        ax5.plot(t, lqi_mean, label="LQI Mean", color="blue")
    if lqi_std is not None and has_data(lqi_std):
        ax5.plot(t, lqi_std, label="LQI Std", color="cyan")

    ax5.set_ylabel("LQI Metrics")
    ax5.tick_params(axis="y", labelcolor="blue")

    ax6 = ax5.twinx()
    if rtt_mean is not None and has_data(rtt_mean):
        ax6.plot(t, rtt_mean, label="RTT Mean (ms)", color="red")

    ax6.set_ylabel("RTT Mean (ms)")
    ax6.tick_params(axis="y", labelcolor="red")

    # Use original df (not rolling) for parent shading (timestamps align enough)
    df_for_bg = df.copy()
    mark_parent_periods(ax5, df_for_bg)

    fig.suptitle((title or "") + " – Predictive Metrics")
    ax5.grid(True)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)

    handles1, labels1 = ax5.get_legend_handles_labels()
    handles2, labels2 = ax6.get_legend_handles_labels()
    handles = []
    labels = []
    seen = set()
    for h, l in list(zip(handles1 + handles2, labels1 + labels2)):
        if l and l not in seen:
            handles.append(h)
            labels.append(l)
            seen.add(l)
    if handles:
        fig.legend(handles, labels, loc="upper right")

    _finalize_figure(fig, output_path)


def plot_attachment_status(df: pd.DataFrame, title: str | None = None, output_path: Path | None = None):
    if "timestamp" not in df.columns or df.empty:
        return

    def get_status(row):
        state = str(row.get("state", "")).strip().lower()
        parent = str(row.get("parent_rloc16", "none")).strip().lower()

        if state == "" or state == "blank" or pd.isna(row.get("state")):
            return "Blank State"
        if state == "detached" or parent in ("none", "", "nan"):
            return "No Parent / Switching"
        return f"Attached to {parent}"

    df2 = df.copy()
    df2["status"] = df2.apply(get_status, axis=1)
    unique_status = list(pd.unique(df2["status"]))
    status_map = {s: i for i, s in enumerate(unique_status)}
    df2["status_num"] = df2["status"].map(status_map)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.step(df2["timestamp"], df2["status_num"], where="post", label="Attachment Status", color="blue")
    ax1.set_yticks(list(status_map.values()))
    ax1.set_yticklabels(list(status_map.keys()))
    ax1.set_ylabel("Attachment Status")
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)

    fig.suptitle((title or "") + " – Attachment Status")
    fig.legend(loc="upper right")

    _finalize_figure(fig, output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parse OT logs in a folder and generate graphs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing .log files (default: ./data)",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("graphs"),
        help="Directory to store generated graphs (default: ./graphs)",
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    graphs_dir: Path = args.graphs_dir

    if not data_dir.is_dir():
        print(f"[ERROR] Data directory does not exist or is not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Use glob to find all .log files
    log_files = sorted(data_dir.glob("*.log"))

    if not log_files:
        print(f"[WARN] No .log files found in {data_dir}")
        return

    for path in log_files:
        print(f"[INFO] Parsing log: {path}")
        try:
            df = parse_ot_log(path)
        except RuntimeError as e:
            print(f"[WARN] Skipping {path.name}: {e}")
            continue

        title = path.name
        stem = path.stem

        # Main LQI/RTT/RSS graph with parent background
        plot_lqi_rtt_rss(df, title=title, output_path=graphs_dir / f"{stem}_lqi_rtt_rss.png")

        # Other graphs
        plot_age(df, title=title, output_path=graphs_dir / f"{stem}_age.png")
        plot_counters(df, title=title, output_path=graphs_dir / f"{stem}_counters.png")
        plot_predictive(df, title=title, output_path=graphs_dir / f"{stem}_predictive.png")
        plot_attachment_status(df, title=title, output_path=graphs_dir / f"{stem}_attachment_status.png")

        print(f"[INFO] Finished {path.name}")


if __name__ == "__main__":
    main()
