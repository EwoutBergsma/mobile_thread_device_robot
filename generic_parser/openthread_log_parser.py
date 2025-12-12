import os
import re
from glob import glob
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, fields
from typing import List, Dict, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Length (in bytes) of the ping reply ICMPv6 message in the log.
PING_REPLY_LEN = 56

# Base directory: the directory where this script resides
SCRIPT_DIR = Path(__file__).resolve().parent

# Data directory (relative to this file)
DATA_DIR = SCRIPT_DIR / "data"

# Regex for round trip time of pings.
ping_rtt_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]
    .*?Round-trip\ min/avg/max\ =\
    (?P<min>\d+(?:\.\d+)?)/ 
    (?P<avg>\d+(?:\.\d+)?)/ 
    (?P<max>\d+(?:\.\d+)?)\ ms\.?
    """,
    re.VERBOSE,
)

# Regex for packet summary lines with packet loss.
ping_packet_loss_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]\s+
    (?P<tx>\d+)\s+packets\s+transmitted,\s+
    (?P<rx>\d+)\s+packets\s+received\.\s+
    Packet\ loss\s*=\s*
    (?P<loss>\d+(?:\.\d+)?)%
    """,
    re.VERBOSE,
)

# Regex for MAC frame tx failures due to NoAck.
mac_frame_tx_noack_failed_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]\s+
    .*?Mac-----------:\s+
    Frame\s+tx\s+attempt\s+\d+/\d+\s+
    failed,\s+error:\s*NoAck\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Regex for rss (dBm) of pings.
ping_rss_regex = re.compile(
    r"""
    ^\[(?P<ts>[^\]]+)\]\s+
    .*?MeshForwarder-:\s+Received\ IPv6\s+\S+\s+msg,\s+
    len:(?P<len>\d+),.*?
    rss:(?P<rss>-?\d+(?:\.\d+)?)
    """,
    re.VERBOSE,
)

# Example line: [2025-12-11T00:55:55.384] Rloc: 4000
cli_command_parent_rloc_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]
    .*?Rloc:\s*
    (?P<rloc>[0-9A-Fa-f]+)
    """,
    re.VERBOSE,
)

# Regex for OT state transition lines.
role_transition_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]
    .*?Role\s+
    (?P<from_state>disabled|detached|child|router|leader)
    \s*->\s*
    (?P<state>disabled|detached|child|router|leader)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Regex for node's own RLOC16 transitions.
node_transition_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]
    .*?RLOC16\s+
    (?P<old>[0-9A-Fa-f]+)
    \s*->\s*
    (?P<new>[0-9A-Fa-f]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# PPS: Backoff interval passed.
pps_backoff_interval_passed_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]\s+
    .*?Mle-----------:\s+
    PeriodicParentSearch:\s+
    Backoff\s+interval\s+passed\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# PPS: Parent RSS value.
pps_parent_rss_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]\s+
    .*?Mle-----------:\s+
    PeriodicParentSearch:\s+
    Parent\s+RSS\s+
    (?P<rss>-?\d+(?:\.\d+)?)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# PPS: RSS threshold met -> searching new parents.
pps_rss_new_parent_search_threshold_met_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]\s+
    .*?Mle-----------:\s+
    PeriodicParentSearch:\s+
    Parent\s+RSS\s+less\s+than\s+
    (?P<thresh>-?\d+(?:\.\d+)?)\s*,\s*
    searching\s+for\s+new\s+parents\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# PPS: (Re)starting timer for backoff interval.
pps_restart_timer_for_backoff_interval_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]\s+
    .*?Mle-----------:\s+
    PeriodicParentSearch:\s+
    \(Re\)starting\s+timer\s+for\s+backoff\s+interval\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Mle: Send Child ID Request (timestamp-only).
send_child_id_request_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]\s+
    .*?Mle-----------:\s+
    Send\s+Child\s+ID\s+Request\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Mle: Receive Child ID Response (timestamp + rloc16), e.g.:
# [2025-12-09T11:28:20.891] ... Mle-----------: Receive Child ID Response (...,0x0c00)
receive_child_id_response_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]\s+
    .*?Mle-----------:\s+
    Receive\s+Child\s+ID\s+Response\s*
    \(
        [^,]+,\s*                     # address part up to comma
        (?P<rloc>0x[0-9A-Fa-f]+)       # capture 0x0c00
    \)
    """,
    re.IGNORECASE | re.VERBOSE,
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
    """
    r = rloc.strip().lower()
    if r.startswith("0x"):
        r = r[2:]
    try:
        value = int(r, 16)
    except ValueError:
        return rloc.strip()
    return f"{value:04x}"


INVALID_RLOC16 = 0xFFFE
PARENT_ROUTER_MASK = 0xFC00


def _rloc16_to_int(rloc: str) -> Optional[int]:
    s = rloc.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    try:
        return int(s, 16)
    except ValueError:
        return None


def derive_parent_router_from_rloc16(node_rloc16: Optional[str]) -> Optional[str]:
    if node_rloc16 is None:
        return None

    v = _rloc16_to_int(node_rloc16)
    if v is None:
        return None

    if v == INVALID_RLOC16:
        return "No Parent"

    parent_router = v & PARENT_ROUTER_MASK
    return f"{parent_router:04x}"


@dataclass
class LogMetrics:
    ping_rtt_timestamps: List[datetime]
    ping_rtt_avg_ms: List[float]

    ping_packet_loss_timestamps: List[datetime]

    mac_frame_tx_noack_failed_timestamps: List[datetime]

    ping_rss_timestamps: List[datetime]
    ping_rss_dbm_values: List[float]

    role_from_transition_timestamps: List[datetime]
    role_from_transition_values: List[str]

    rloc16_from_transition_timestamps: List[datetime]
    rloc16_from_transition_values: List[str]

    parent_router_from_rloc16_transition_timestamps: List[datetime]
    parent_router_from_rloc16_transition_values: List[str]

    parent_rloc16_from_query_timestamps: List[datetime]
    parent_rloc16_from_query_values: List[str]

    pps_backoff_interval_passed_timestamps: List[datetime]

    pps_parent_rss_timestamps: List[datetime]
    pps_parent_rss_dbm_values: List[float]

    pps_rss_new_parent_search_threshold_met_timestamps: List[datetime]

    pps_restart_timer_for_backoff_interval_timestamps: List[datetime]

    send_child_id_request_timestamps: List[datetime]

    # Receive_Child_ID_Response: timestamps + rloc16 values (normalized, e.g., 0c00).
    receive_child_id_response_timestamps: List[datetime]
    receive_child_id_response_rloc16_values: List[str]

    total_ping_tx_packets: int
    total_ping_rx_packets: int


def build_parent_router_timeline(metrics: LogMetrics) -> None:
    if metrics.rloc16_from_transition_timestamps:
        out_ts: List[datetime] = []
        out_vals: List[str] = []
        for ts, node_rloc in zip(
            metrics.rloc16_from_transition_timestamps,
            metrics.rloc16_from_transition_values,
        ):
            parent_router = derive_parent_router_from_rloc16(node_rloc)
            if parent_router is None:
                continue
            out_ts.append(ts)
            out_vals.append(parent_router)

        metrics.parent_router_from_rloc16_transition_timestamps = out_ts
        metrics.parent_router_from_rloc16_transition_values = out_vals
        return

    metrics.parent_router_from_rloc16_transition_timestamps = []
    metrics.parent_router_from_rloc16_transition_values = []


def parse_log_file(log_path: str) -> LogMetrics:
    print(f"\n[PROCESS] Starting file: {log_path}")

    metrics = LogMetrics(
        ping_rtt_timestamps=[],
        ping_rtt_avg_ms=[],
        ping_packet_loss_timestamps=[],
        mac_frame_tx_noack_failed_timestamps=[],
        ping_rss_timestamps=[],
        ping_rss_dbm_values=[],
        role_from_transition_timestamps=[],
        role_from_transition_values=[],
        rloc16_from_transition_timestamps=[],
        rloc16_from_transition_values=[],
        parent_router_from_rloc16_transition_timestamps=[],
        parent_router_from_rloc16_transition_values=[],
        parent_rloc16_from_query_timestamps=[],
        parent_rloc16_from_query_values=[],
        pps_backoff_interval_passed_timestamps=[],
        pps_parent_rss_timestamps=[],
        pps_parent_rss_dbm_values=[],
        pps_rss_new_parent_search_threshold_met_timestamps=[],
        pps_restart_timer_for_backoff_interval_timestamps=[],
        send_child_id_request_timestamps=[],
        receive_child_id_response_timestamps=[],
        receive_child_id_response_rloc16_values=[],
        total_ping_tx_packets=0,
        total_ping_rx_packets=0,
    )

    first_log_ts: Optional[datetime] = None
    last_log_ts: Optional[datetime] = None
    first_state_seen = False
    first_node_rloc_seen = False

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line_stripped = line.rstrip("\n")

            # Global timestamp tracking (first/last in the whole log).
            m_ts_generic = re.match(r"\[(?P<ts>[^\]]+)\]", line_stripped)
            if m_ts_generic:
                ts_line = parse_timestamp(m_ts_generic.group("ts"))
                if first_log_ts is None or ts_line < first_log_ts:
                    first_log_ts = ts_line
                if last_log_ts is None or ts_line > last_log_ts:
                    last_log_ts = ts_line

            # Receive_Child_ID_Response (timestamp + rloc16).
            m_child_id_resp = receive_child_id_response_regex.search(line_stripped)
            if m_child_id_resp:
                ts_str = m_child_id_resp.group("ts")
                raw_rloc = m_child_id_resp.group("rloc")  # e.g. 0x0c00
                ts = parse_timestamp(ts_str)
                rloc16_norm = normalize_rloc16(raw_rloc)  # e.g. 0c00
                print(f"  [CHILD ID RESP] Receive Child ID Response rloc16={rloc16_norm} at {ts_str}")
                metrics.receive_child_id_response_timestamps.append(ts)
                metrics.receive_child_id_response_rloc16_values.append(rloc16_norm)

            # Send Child ID Request (timestamp-only).
            m_child_id_req = send_child_id_request_regex.search(line_stripped)
            if m_child_id_req:
                ts_str = m_child_id_req.group("ts")
                ts = parse_timestamp(ts_str)
                print(f"  [CHILD ID REQ] Send Child ID Request at {ts_str}")
                metrics.send_child_id_request_timestamps.append(ts)

            # PPS: Parent RSS value.
            m_pps_rss = pps_parent_rss_regex.search(line_stripped)
            if m_pps_rss:
                ts_str = m_pps_rss.group("ts")
                rss_str = m_pps_rss.group("rss")
                ts = parse_timestamp(ts_str)
                rss_val = float(rss_str)
                print(f"  [PPS RSS] Parent RSS {rss_val} dBm at {ts_str}")
                metrics.pps_parent_rss_timestamps.append(ts)
                metrics.pps_parent_rss_dbm_values.append(rss_val)

            # PPS: RSS threshold met (searching for new parents).
            m_pps_thresh = pps_rss_new_parent_search_threshold_met_regex.search(line_stripped)
            if m_pps_thresh:
                ts_str = m_pps_thresh.group("ts")
                ts = parse_timestamp(ts_str)
                thresh = m_pps_thresh.group("thresh")
                print(f"  [PPS THRESH] Parent RSS less than {thresh} -> searching new parents at {ts_str}")
                metrics.pps_rss_new_parent_search_threshold_met_timestamps.append(ts)

            # PPS: (Re)starting timer for backoff interval.
            m_pps_restart = pps_restart_timer_for_backoff_interval_regex.search(line_stripped)
            if m_pps_restart:
                ts_str = m_pps_restart.group("ts")
                ts = parse_timestamp(ts_str)
                print(f"  [PPS TIMER] (Re)starting timer for backoff interval at {ts_str}")
                metrics.pps_restart_timer_for_backoff_interval_timestamps.append(ts)

            # PPS: Backoff interval passed.
            m_pps_backoff = pps_backoff_interval_passed_regex.search(line_stripped)
            if m_pps_backoff:
                ts_str = m_pps_backoff.group("ts")
                ts = parse_timestamp(ts_str)
                print(f"  [PPS] Backoff interval passed at {ts_str}")
                metrics.pps_backoff_interval_passed_timestamps.append(ts)

            # Packet loss / summary detection (any percentage).
            m_loss = ping_packet_loss_regex.search(line_stripped)
            if m_loss:
                ts_loss_str = m_loss.group("ts")
                ts_loss = parse_timestamp(ts_loss_str)

                tx = int(m_loss.group("tx"))
                rx = int(m_loss.group("rx"))
                loss_pct = float(m_loss.group("loss"))

                metrics.total_ping_tx_packets += tx
                metrics.total_ping_rx_packets += rx

                if loss_pct > 0.0:
                    print(f"  [LOSS] Packet loss {loss_pct}% at {ts_loss_str}")
                    metrics.ping_packet_loss_timestamps.append(ts_loss)

            # MAC NoAck tx failures.
            m_noack = mac_frame_tx_noack_failed_regex.search(line_stripped)
            if m_noack:
                ts_str = m_noack.group("ts")
                ts = parse_timestamp(ts_str)
                print(f"  [NOACK] Frame tx failed (NoAck) at {ts_str}")
                metrics.mac_frame_tx_noack_failed_timestamps.append(ts)

            # RTT detection.
            m_rtt = ping_rtt_regex.search(line_stripped)
            if m_rtt:
                print(f"  [USE RTT] Line {line_no}: {line_stripped}")
                ts_str = m_rtt.group("ts")
                avg_str = m_rtt.group("avg")
                ts = parse_timestamp(ts_str)
                avg_ms = float(avg_str)
                metrics.ping_rtt_timestamps.append(ts)
                metrics.ping_rtt_avg_ms.append(avg_ms)

            # Parent detection (RLOC16 from "Rloc:" lines).
            m_parent = cli_command_parent_rloc_regex.search(line_stripped)
            if m_parent:
                ts_str = m_parent.group("ts")
                raw_rloc16 = m_parent.group("rloc")
                rloc16_norm = normalize_rloc16(raw_rloc16)

                p_int = _rloc16_to_int(rloc16_norm)
                if p_int == INVALID_RLOC16:
                    rloc16_val = "No Parent"
                else:
                    rloc16_val = rloc16_norm

                ts = parse_timestamp(ts_str)
                print(f"  [USE PARENT] Line {line_no}: {line_stripped}")
                print(f"               -> Parent RLOC16: {rloc16_val}")
                metrics.parent_rloc16_from_query_timestamps.append(ts)
                metrics.parent_rloc16_from_query_values.append(rloc16_val)

            # State transition detection (Role X -> Y).
            m_state = role_transition_regex.search(line_stripped)
            if m_state:
                ts_str = m_state.group("ts")
                from_state_str = m_state.group("from_state")
                state_str = m_state.group("state")
                ts = parse_timestamp(ts_str)
                state_norm = state_str.strip().lower()
                from_norm = from_state_str.strip().lower()
                print(f"  [STATE] {from_norm} -> {state_norm} at {ts_str}")

                if not first_state_seen:
                    first_state_seen = True
                    init_ts = first_log_ts if first_log_ts is not None else ts
                    metrics.role_from_transition_timestamps.append(init_ts)
                    metrics.role_from_transition_values.append(from_norm)

                metrics.role_from_transition_timestamps.append(ts)
                metrics.role_from_transition_values.append(state_norm)

            # Node RLOC16 transitions (RLOC16 old -> new).
            m_node_rloc = node_transition_regex.search(line_stripped)
            if m_node_rloc:
                ts_str = m_node_rloc.group("ts")
                old_rloc = m_node_rloc.group("old")
                new_rloc = m_node_rloc.group("new")
                ts = parse_timestamp(ts_str)
                new_norm = normalize_rloc16(new_rloc)
                old_norm = normalize_rloc16(old_rloc)
                print(f"  [NODE RLOC] {old_norm} -> {new_norm} at {ts_str}")

                if not first_node_rloc_seen:
                    first_node_rloc_seen = True
                    init_ts_rloc = first_log_ts if first_log_ts is not None else ts
                    metrics.rloc16_from_transition_timestamps.append(init_ts_rloc)
                    metrics.rloc16_from_transition_values.append(old_norm)

                metrics.rloc16_from_transition_timestamps.append(ts)
                metrics.rloc16_from_transition_values.append(new_norm)

            # RSS detection for ping replies.
            m_rss = ping_rss_regex.search(line_stripped)
            if m_rss:
                length = int(m_rss.group("len"))
                if length == PING_REPLY_LEN:
                    ts_str = m_rss.group("ts")
                    rss_str = m_rss.group("rss")
                    ts = parse_timestamp(ts_str)
                    rss_val = float(rss_str)
                    print(f"  [PING RSS] len={length} -> {rss_val} dBm at {ts_str}")
                    metrics.ping_rss_timestamps.append(ts)
                    metrics.ping_rss_dbm_values.append(rss_val)

    # Extend last role to the last known timestamp in the log.
    if metrics.role_from_transition_timestamps and last_log_ts is not None:
        last_state_ts = metrics.role_from_transition_timestamps[-1]
        last_state = metrics.role_from_transition_values[-1]
        if last_log_ts > last_state_ts:
            metrics.role_from_transition_timestamps.append(last_log_ts)
            metrics.role_from_transition_values.append(last_state)

    # Extend last node RLOC16 to the last known timestamp in the log.
    if metrics.rloc16_from_transition_timestamps and last_log_ts is not None:
        last_node_ts = metrics.rloc16_from_transition_timestamps[-1]
        last_node_val = metrics.rloc16_from_transition_values[-1]
        if last_log_ts > last_node_ts:
            metrics.rloc16_from_transition_timestamps.append(last_log_ts)
            metrics.rloc16_from_transition_values.append(last_node_val)

    build_parent_router_timeline(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Generic visualization
# ---------------------------------------------------------------------------

def _format_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, which="both", linestyle="--", alpha=0.3)


def _discover_time_series(metrics: LogMetrics) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []

    metric_values: Dict[str, Any] = {f.name: getattr(metrics, f.name) for f in fields(LogMetrics)}

    for fname, fval in metric_values.items():
        if not fname.endswith("_timestamps"):
            continue
        if not isinstance(fval, list) or not fval:
            continue

        timestamps = fval
        tokens = fname.split("_")[:-1]  # drop 'timestamps'
        root = "_".join(tokens) if tokens else fname[:-len("_timestamps")]
        root_label = root.replace("_", " ").title() if root else fname

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

            if (
                other_name.startswith(candidate_prefix)
                or other_name == candidate_exact
                or other_name == candidate_plural
            ):
                candidates.append({"name": other_name, "values": other_val})

        if candidates:
            for c in candidates:
                values = c["values"]
                non_none = next((v for v in values if v is not None), None)
                categorical = isinstance(non_none, str) if non_none is not None else False

                val_name = c["name"]
                if val_name.startswith(candidate_prefix):
                    suffix = val_name[len(candidate_prefix):]
                elif val_name in (candidate_exact, candidate_plural):
                    suffix = val_name
                else:
                    suffix = val_name
                pretty_suffix = suffix.replace("_", " ").title()

                label = f"{root_label} – {pretty_suffix}" if pretty_suffix and pretty_suffix.lower() != root_label.lower() else root_label

                series.append(
                    {
                        "timestamps": timestamps,
                        "values": values,
                        "label": label,
                        "categorical": categorical,
                    }
                )
        else:
            series.append(
                {
                    "timestamps": timestamps,
                    "values": None,
                    "label": f"{root_label} Events",
                    "categorical": False,
                }
            )

    return series


def visualize_metrics(metrics: LogMetrics, title: str = "OpenThread log metrics") -> None:
    series_list = _discover_time_series(metrics)

    if not series_list:
        print("[VIS] No metrics to visualize.")
        return

    nrows = len(series_list)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(12, 3 * nrows))
    if nrows == 1:
        axes = [axes]

    if metrics.total_ping_tx_packets > 0:
        pdr = 100.0 * metrics.total_ping_rx_packets / metrics.total_ping_tx_packets
        sup_title = (
            f"{title} – PDR: "
            f"{metrics.total_ping_rx_packets}/{metrics.total_ping_tx_packets} "
            f"({pdr:.2f}%)"
        )
    else:
        sup_title = title
    fig.suptitle(sup_title, fontsize=14)

    for ax, spec in zip(axes, series_list):
        ts = spec["timestamps"]
        values = spec["values"]
        label = spec["label"]
        categorical = spec["categorical"]

        if values is None:
            n = len(ts)
            for ts_i in ts:
                ax.axvline(ts_i, linestyle="--", alpha=0.5)
            ax.set_title(f"{label} (n={n})")
            ax.set_yticks([])
        else:
            if categorical:
                categories = sorted({v for v in values if v is not None})
                y_map = {cat: idx for idx, cat in enumerate(categories)}
                y_vals = [y_map.get(v, None) for v in values]

                x_plot = [t for t, y in zip(ts, y_vals) if y is not None]
                y_plot = [y for y in y_vals if y is not None]
                n = len(x_plot)

                if x_plot:
                    ax.step(x_plot, y_plot, where="post")
                ax.set_yticks(range(len(categories)))
                ax.set_yticklabels(categories)
                ax.set_ylabel(label)
                ax.set_title(f"{label} (n={n})")
            else:
                x_plot = [t for t, v in zip(ts, values) if v is not None]
                y_plot = [v for v in values if v is not None]
                n = len(x_plot)

                ax.scatter(x_plot, y_plot, s=8)
                ax.set_ylabel(label)
                ax.set_title(f"{label} (n={n})")

        _format_time_axis(ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main() -> None:
    data_dir_path = DATA_DIR

    pattern = str(DATA_DIR / "**" / "*.log")
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

    for log_path in log_files:
        metrics = parse_log_file(log_path)
        rel_label = str(Path(log_path).relative_to(data_dir_path))

        print("\n[SUMMARY]")
        print(f"  File:                                {rel_label}")
        print(f"  RTT samples:                         {len(metrics.ping_rtt_timestamps)}")
        print(f"  RSS samples:                         {len(metrics.ping_rss_timestamps)}")
        print(f"  Role-from-transition points:         {len(metrics.role_from_transition_timestamps)}")
        print(f"  Node RLOC16 points:                  {len(metrics.rloc16_from_transition_timestamps)}")
        print(f"  Parent RLOC16 queries:               {len(metrics.parent_rloc16_from_query_timestamps)}")
        print(f"  Parent router points:                {len(metrics.parent_router_from_rloc16_transition_timestamps)}")
        print(f"  MAC NoAck tx failures:               {len(metrics.mac_frame_tx_noack_failed_timestamps)}")
        print(f"  Loss events:                         {len(metrics.ping_packet_loss_timestamps)}")
        print(f"  PPS backoff passed events:           {len(metrics.pps_backoff_interval_passed_timestamps)}")
        print(f"  PPS Parent RSS samples:              {len(metrics.pps_parent_rss_timestamps)}")
        print(f"  PPS new-parent-search threshold met: {len(metrics.pps_rss_new_parent_search_threshold_met_timestamps)}")
        print(f"  PPS restart timer (backoff) events:  {len(metrics.pps_restart_timer_for_backoff_interval_timestamps)}")
        print(f"  Send Child ID Request events:        {len(metrics.send_child_id_request_timestamps)}")
        print(f"  Receive Child ID Response events:    {len(metrics.receive_child_id_response_timestamps)}")

        if metrics.total_ping_tx_packets > 0:
            pdr = 100.0 * metrics.total_ping_rx_packets / metrics.total_ping_tx_packets
            print(f"  Total TX:                            {metrics.total_ping_tx_packets}")
            print(f"  Total RX:                            {metrics.total_ping_rx_packets}")
            print(f"  Overall PDR:                         {pdr:.2f}%")
        else:
            print("  No packet summary lines found (TX/RX totals unavailable).")

        visualize_metrics(metrics, title=rel_label)


if __name__ == "__main__":
    main()
