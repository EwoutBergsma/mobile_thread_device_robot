import os
import re
from glob import glob
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, fields
from typing import List, Dict, Optional, Any
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Length (in bytes) of the ping reply ICMPv6 message in the log.
PING_REPLY_LEN = 56

# Base directory: the directory where this script resides
SCRIPT_DIR = Path(__file__).resolve().parent

# Data directory (relative to this file)
DATA_DIR = SCRIPT_DIR / "data"

# Regex for round trip time of pings.
# Example: [2025-12-11T00:56:16.332] 1 packets transmitted, 1 packets received. Packet loss = 0.0%. Round-trip min/avg/max = 55/55.000/55 ms.
ping_rtt_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]               # [2025-12-04T17:54:36.720]
    .*?Round-trip\ min/avg/max\ =\   # Round-trip min/avg/max =
    (?P<min>\d+(?:\.\d+)?)/          # min
    (?P<avg>\d+(?:\.\d+)?)/          # avg
    (?P<max>\d+(?:\.\d+)?)\ ms\.?    # max ms.
    """,
    re.VERBOSE,
)

# Regex for packet summary lines with packet loss (0.0%, 10.0%, 100.0%, etc.).
# Example: [2025-12-11T00:56:16.239] 1 packets transmitted, 0 packets received. Packet loss = 100.0%.
# Requires periodic pinging from the logged device.
ping_packet_loss_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]\s+                  # timestamp in brackets
    (?P<tx>\d+)\s+packets\s+transmitted,\s+
    (?P<rx>\d+)\s+packets\s+received\.\s+
    Packet\ loss\s*=\s*
    (?P<loss>\d+(?:\.\d+)?)%              # e.g. 0.0, 10.0, 100.0
    """,
    re.VERBOSE,
)

# Regex for rss (dBm) of pings.
# Example: [2025-12-10T23:30:32.356] I(19735589) OPENTHREAD:[I] MeshForwarder-: Received IPv6 ICMP6 msg, len:56, chksum:8e94, ecn:no, from:0x7000, sec:yes, prio:normal, rss:-101.0
# Requires periodic pinging from the logged device.
# Requires log level 4 or higher.
ping_rss_regex = re.compile(
    r"""
    ^\[(?P<ts>[^\]]+)\]\s+                      # timestamp at start of line
    .*?MeshForwarder-:\s+Received\ IPv6\s+\S+\s+msg,\s+
    len:(?P<len>\d+),.*?                        # capture len:NN
    rss:(?P<rss>-?\d+(?:\.\d+)?)                # capture rss:-90.0
    """,
    re.VERBOSE,
)

# Example line: [2025-12-11T00:55:55.384] Rloc: 4000
# Requires periodic "parent" command send to the cli of the child, when capturing logs.
cli_command_parent_rloc_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]      # timestamp in brackets
    .*?Rloc:\s*
    (?P<rloc>[0-9A-Fa-f]+)  # hex-only RLOC16 value
    """,
    re.VERBOSE,
)

# Regex for OT state transition lines, e.g.:
# Example: [2025-12-11T01:27:24.816] I(26748009) OPENTHREAD:[N] Mle-----------: Role child -> detached
# Requires log level 4 or higher.
role_transition_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]                  # timestamp
    .*?Role\s+
    (?P<from_state>disabled|detached|child|router|leader)
    \s*->\s*
    (?P<state>disabled|detached|child|router|leader)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Regex for node's own RLOC16 transitions, e.g.:
# Example: [2025-12-10T23:30:42.771] I(19745999) OPENTHREAD:[N] Mle-----------: RLOC16 70eb -> fffe
# Requires log level 4 or higher.
node_transition_regex = re.compile(
    r"""
    \[(?P<ts>[^\]]+)\]               # timestamp in brackets
    .*?RLOC16\s+
    (?P<old>[0-9A-Fa-f]+)            # old RLOC16
    \s*->\s*
    (?P<new>[0-9A-Fa-f]+)            # new RLOC16
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

    Examples:
        "c00"  -> "0c00"
        "0C00" -> "0c00"
        "0000" -> "0000"
    """
    r = rloc.strip().lower()
    if r.startswith("0x"):
        r = r[2:]
    try:
        value = int(r, 16)
    except ValueError:
        # If not valid hex, just return the stripped original.
        return rloc.strip()
    return f"{value:04x}"


# RLOC16 interpretation:
# - 0xfffe indicates no/invalid RLOC16 (treat as "No Parent")
# - Parent router RLOC16 can be derived by clearing the Child ID bits (low 10 bits).
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
    """
    Derive the parent router's RLOC16 from the node's RLOC16.

    If node RLOC16 is fffe, this indicates no parent.
    """
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
    # Ping RTT (round-trip time) statistics from ping summary lines.
    ping_rtt_timestamps: List[datetime]      # From ping summary lines with RTT stats.
    ping_rtt_avg_ms: List[float]             # Average RTT in milliseconds at each RTT timestamp.

    # Ping-level packet loss, based directly on ping summary lines.
    ping_packet_loss_timestamps: List[datetime]  # From ping summary lines that report non-zero packet loss.

    # RSSI measurements for received ping reply messages.
    ping_rss_timestamps: List[datetime]      # From ping RSS log lines for ICMPv6 replies.
    ping_rss_dbm_values: List[float]         # RSS values in dBm at the corresponding timestamps.

    # OpenThread role of the node over time.
    # These are derived from "Role <from> -> <to>" transition lines (explicit "->" transitions),
    # plus synthetic initial/final points added at the first/last log timestamps.
    role_from_transition_timestamps: List[datetime]
    role_from_transition_values: List[str]

    # Node's own RLOC16 (logical address) over time.
    # These are derived from "RLOC16 <old> -> <new>" transition lines (explicit "->" transitions),
    # plus synthetic initial/final points added at the first/last log timestamps.
    rloc16_from_transition_timestamps: List[datetime]
    rloc16_from_transition_values: List[str]

    # Parent router RLOC16 over time.
    # This is derived from the node's own RLOC16 transitions by extracting the parent router portion.
    parent_router_from_rloc16_transition_timestamps: List[datetime]
    parent_router_from_rloc16_transition_values: List[str]

    # Parent RLOC16 values obtained through the CLI "parent" command (no "->" transitions).
    parent_rloc16_from_query_timestamps: List[datetime] # From "Rloc: XXXX" CLI responses.
    parent_rloc16_from_query_values: List[str]         # Normalized parent RLOC16 value at each query timestamp.

    # Aggregate ping counters over the entire log, used to compute overall PDR.
    total_ping_tx_packets: int               # From ping summary lines (packets transmitted).
    total_ping_rx_packets: int               # From ping summary lines (packets received).


def build_parent_router_timeline(metrics: LogMetrics) -> None:
    """
    Build a time series of the parent router RLOC16.

    Primary source:
      - Node RLOC16 transitions ("RLOC16 old -> new"), because the parent router portion
        is encoded within the node's RLOC16.

    Fallback (if no node RLOC16 transitions exist):
      - Parent RLOC16 values obtained via periodic CLI "parent" queries ("Rloc: XXXX").
    """
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

    # Fallback: use CLI "parent" queries if present.
    if metrics.parent_rloc16_from_query_timestamps:
        out_ts = list(metrics.parent_rloc16_from_query_timestamps)
        out_vals: List[str] = []
        for p in metrics.parent_rloc16_from_query_values:
            p_norm = normalize_rloc16(p)
            p_int = _rloc16_to_int(p_norm)
            if p_int == INVALID_RLOC16:
                out_vals.append("No Parent")
            else:
                out_vals.append(p_norm)

        metrics.parent_router_from_rloc16_transition_timestamps = out_ts
        metrics.parent_router_from_rloc16_transition_values = out_vals
        return

    metrics.parent_router_from_rloc16_transition_timestamps = []
    metrics.parent_router_from_rloc16_transition_values = []


def parse_log_file(log_path: str) -> LogMetrics:
    """
    Read a log file and extract metrics.

    Additionally:
      - Record the initial role (from_state of the first "Role X -> Y" line)
        at the first timestamp seen in the log.
      - Extend the last role to the last timestamp seen in the log.
      - Record the initial node RLOC16 (old value of the first "RLOC16 A -> B"
        line) at the first timestamp seen in the log.
      - Extend the last node RLOC16 to the last timestamp seen in the log.
    """
    print(f"\n[PROCESS] Starting file: {log_path}")

    metrics = LogMetrics(
        ping_rtt_timestamps=[],
        ping_rtt_avg_ms=[],
        ping_packet_loss_timestamps=[],
        ping_rss_timestamps=[],
        ping_rss_dbm_values=[],
        role_from_transition_timestamps=[],
        role_from_transition_values=[],
        rloc16_from_transition_timestamps=[],
        rloc16_from_transition_values=[],
        parent_rloc16_from_query_timestamps=[],
        parent_rloc16_from_query_values=[],
        parent_router_from_rloc16_transition_timestamps=[],
        parent_router_from_rloc16_transition_values=[],
        total_ping_tx_packets=0,
        total_ping_rx_packets=0,
    )

    # Track first and last timestamps seen in the entire log file.
    first_log_ts: Optional[datetime] = None
    last_log_ts: Optional[datetime] = None

    # Flags to handle "initial" injection for state and node RLOC.
    first_state_seen = False
    first_node_rloc_seen = False

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line_stripped = line.rstrip("\n")

            # --- Global timestamp tracking (first/last in the whole log). ---
            m_ts_generic = re.match(r"\[(?P<ts>[^\]]+)\]", line_stripped)
            if m_ts_generic:
                ts_line = parse_timestamp(m_ts_generic.group("ts"))
                if first_log_ts is None or ts_line < first_log_ts:
                    first_log_ts = ts_line
                if last_log_ts is None or ts_line > last_log_ts:
                    last_log_ts = ts_line

            # --- Packet loss / summary detection (any percentage). ---
            m_loss = ping_packet_loss_regex.search(line_stripped)
            if m_loss:
                ts_loss_str = m_loss.group("ts")
                ts_loss = parse_timestamp(ts_loss_str)

                tx = int(m_loss.group("tx"))
                rx = int(m_loss.group("rx"))
                loss_pct = float(m_loss.group("loss"))

                # Accumulate totals for overall PDR.
                metrics.total_ping_tx_packets += tx
                metrics.total_ping_rx_packets += rx

                # Keep timestamp for non-zero loss events.
                if loss_pct > 0.0:
                    print(f"  [LOSS] Packet loss {loss_pct}% at {ts_loss_str}")
                    metrics.ping_packet_loss_timestamps.append(ts_loss)

            # --- RTT detection (only lines with Round-trip stats). ---
            m_rtt = ping_rtt_regex.search(line_stripped)
            if m_rtt:
                print(f"  [USE RTT] Line {line_no}: {line_stripped}")
                ts_str = m_rtt.group("ts")
                avg_str = m_rtt.group("avg")
                ts = parse_timestamp(ts_str)
                avg_ms = float(avg_str)
                metrics.ping_rtt_timestamps.append(ts)
                metrics.ping_rtt_avg_ms.append(avg_ms)

            # --- Parent detection (RLOC16 from "Rloc:" lines). ---
            m_parent = cli_command_parent_rloc_regex.search(line_stripped)
            if m_parent:
                ts_str = m_parent.group("ts")
                raw_rloc16 = m_parent.group("rloc")
                rloc16 = normalize_rloc16(raw_rloc16)
                ts = parse_timestamp(ts_str)
                print(f"  [USE PARENT] Line {line_no}: {line_stripped}")
                print(f"               -> Parent RLOC16: {rloc16}")
                metrics.parent_rloc16_from_query_timestamps.append(ts)
                metrics.parent_rloc16_from_query_values.append(rloc16)

            # --- State transition detection (Role X -> Y). ---
            m_state = role_transition_regex.search(line_stripped)
            if m_state:
                ts_str = m_state.group("ts")
                from_state_str = m_state.group("from_state")
                state_str = m_state.group("state")
                ts = parse_timestamp(ts_str)
                state_norm = state_str.strip().lower()
                from_norm = from_state_str.strip().lower()
                print(f"  [STATE] {from_norm} -> {state_norm} at {ts_str}")

                # For the first state transition, also record the "from" state at
                # the first timestamp in the log so that the initial role
                # (e.g. 'detached') appears in the plots.
                if not first_state_seen:
                    first_state_seen = True
                    init_ts = first_log_ts if first_log_ts is not None else ts
                    metrics.role_from_transition_timestamps.append(init_ts)
                    metrics.role_from_transition_values.append(from_norm)

                # Always record the destination state at its actual timestamp.
                metrics.role_from_transition_timestamps.append(ts)
                metrics.role_from_transition_values.append(state_norm)

            # --- Node RLOC16 transitions (RLOC16 old -> new). ---
            m_node_rloc = node_transition_regex.search(line_stripped)
            if m_node_rloc:
                ts_str = m_node_rloc.group("ts")
                old_rloc = m_node_rloc.group("old")
                new_rloc = m_node_rloc.group("new")
                ts = parse_timestamp(ts_str)
                new_norm = normalize_rloc16(new_rloc)
                old_norm = normalize_rloc16(old_rloc)
                print(f"  [NODE RLOC] {old_norm} -> {new_norm} at {ts_str}")

                # For the first node RLOC transition, also record the "old"
                # RLOC16 at the first timestamp in the log so the initial
                # RLOC16 (e.g. 403a) appears in the plots.
                if not first_node_rloc_seen:
                    first_node_rloc_seen = True
                    init_ts_rloc = first_log_ts if first_log_ts is not None else ts
                    metrics.rloc16_from_transition_timestamps.append(init_ts_rloc)
                    metrics.rloc16_from_transition_values.append(old_norm)

                # Always record the new RLOC at its actual timestamp.
                metrics.rloc16_from_transition_timestamps.append(ts)
                metrics.rloc16_from_transition_values.append(new_norm)

            # --- RSS detection for ping replies. ---
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

    # --- Extend last role to the last known timestamp in the log. ---
    if metrics.role_from_transition_timestamps and last_log_ts is not None:
        last_state_ts = metrics.role_from_transition_timestamps[-1]
        last_state = metrics.role_from_transition_values[-1]
        # Only add an extra point if the log actually continues beyond
        # the last transition timestamp.
        if last_log_ts > last_state_ts:
            metrics.role_from_transition_timestamps.append(last_log_ts)
            metrics.role_from_transition_values.append(last_state)

    # --- Extend last node RLOC16 to the last known timestamp in the log. ---
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
    """Helper to format datetime x-axis consistently."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, which="both", linestyle="--", alpha=0.3)


def _discover_time_series(metrics: LogMetrics) -> List[Dict[str, Any]]:
    """
    Inspect LogMetrics and automatically discover time series.

    Logic:
      - Any field ending with `_timestamps` and containing a non-empty list
        is treated as a time axis.
      - For its 'root' (everything before `_timestamps`), find list fields:
          * whose name starts with root + '_'
          * or whose name equals root
          * or whose name equals root + 's' (simple plural)
        and that have the same length as the timestamps list.
      - If at least one matching value list exists, each becomes a value
        series on that time axis; otherwise the root becomes an event-only
        series.
    """
    series: List[Dict[str, Any]] = []

    # Map field name -> value.
    metric_values: Dict[str, Any] = {
        f.name: getattr(metrics, f.name) for f in fields(LogMetrics)
    }

    for fname, fval in metric_values.items():
        if not fname.endswith("_timestamps"):
            continue
        if not isinstance(fval, list) or not fval:
            continue

        timestamps = fval

        # Root is everything before the final "_timestamps" token.
        tokens = fname.split("_")[:-1]  # drop 'timestamps'
        root = "_".join(tokens) if tokens else fname[:-len("_timestamps")]
        root_label = root.replace("_", " ").title() if root else fname

        # Candidate value names to match against.
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

            name_matches = (
                other_name.startswith(candidate_prefix)
                or other_name == candidate_exact
                or other_name == candidate_plural
            )
            if not name_matches:
                continue

            candidates.append({"name": other_name, "values": other_val})

        if candidates:
            for c in candidates:
                values = c["values"]
                # Determine if categorical (string) or numeric.
                non_none = next((v for v in values if v is not None), None)
                categorical = isinstance(non_none, str) if non_none is not None else False

                # Build a readable label: "Root – Suffix" where possible.
                val_name = c["name"]
                if val_name.startswith(candidate_prefix):
                    suffix = val_name[len(candidate_prefix):]
                elif val_name in (candidate_exact, candidate_plural):
                    suffix = val_name
                else:
                    suffix = val_name
                pretty_suffix = suffix.replace("_", " ").title()

                if pretty_suffix and pretty_suffix.lower() != root_label.lower():
                    label = f"{root_label} – {pretty_suffix}"
                else:
                    label = root_label

                series.append(
                    {
                        "timestamps": timestamps,
                        "values": values,
                        "label": label,
                        "categorical": categorical,
                    }
                )
        else:
            # Event-only series (no associated value list).
            label = f"{root_label} Events"
            series.append(
                {
                    "timestamps": timestamps,
                    "values": None,
                    "label": label,
                    "categorical": False,
                }
            )

    return series


def visualize_metrics(metrics: LogMetrics, title: str = "OpenThread log metrics") -> None:
    """
    Generic visualization of all parsed metrics.

    Every discovered time series (or event series) gets its own subplot:
      - numeric values: scatter only (no line)
      - categorical values: stepped series with discrete y-ticks
      - events (timestamps only): vertical lines vs. time
    """
    series_list = _discover_time_series(metrics)

    if not series_list:
        print("[VIS] No metrics to visualize.")
        return

    nrows = len(series_list)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(12, 3 * nrows))
    if nrows == 1:
        axes = [axes]

    # Overall PDR summary for the figure title.
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
            # Event-only series: vertical lines.
            for ts_i in ts:
                ax.axvline(ts_i, linestyle="--", alpha=0.5)
            ax.set_title(label)
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            if categorical:
                # Map categories to integer y values.
                categories = sorted({v for v in values if v is not None})
                y_map = {cat: idx for idx, cat in enumerate(categories)}
                y_vals = [y_map.get(v, None) for v in values]

                x_plot = [t for t, y in zip(ts, y_vals) if y is not None]
                y_plot = [y for y in y_vals if y is not None]

                if x_plot:
                    ax.step(x_plot, y_plot, where="post")
                ax.set_yticks(range(len(categories)))
                ax.set_yticklabels(categories)
                ax.set_ylabel(label)
                ax.set_title(label)
            else:
                # Numeric time series: scatter only, small markers.
                ax.scatter(ts, values, s=8)
                ax.set_ylabel(label)
                ax.set_title(label)

        _format_time_axis(ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point: glob log files, generic visualization, no boxplots
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir_path = DATA_DIR

    # Search for all .log files under data_dir_path (relative to this script)
    pattern = str(DATA_DIR / "**" / "*.log")
    all_candidates = glob(pattern, recursive=True)

    log_files: List[str] = []
    for path_str in all_candidates:
        p = Path(path_str)
        try:
            rel_parts = p.relative_to(data_dir_path).parts
        except ValueError:
            # Should not happen, but be defensive.
            continue

        dir_parts = rel_parts[:-1]
        if any(part.startswith(".") for part in dir_parts):
            # Skip dot-directories under data/
            continue

        log_files.append(path_str)

    if not log_files:
        print(f"[INFO] No .log files found in {data_dir_path}")
        return

    print(f"[INFO] Found {len(log_files)} .log file(s) in {data_dir_path} (excluding dot-directories):")
    for lf in log_files:
        print(f"  - {lf}")

    # Process each log file: parse, summarize, visualize
    for log_path in log_files:
        metrics = parse_log_file(log_path)
        rel_label = str(Path(log_path).relative_to(data_dir_path))

        print("\n[SUMMARY]")
        print(f"  File:                        {rel_label}")
        print(f"  RTT samples:                 {len(metrics.ping_rtt_timestamps)}")
        print(f"  RSS samples:                 {len(metrics.ping_rss_timestamps)}")
        print(f"  Role-from-transition points: {len(metrics.role_from_transition_timestamps)}")
        print(f"  Node RLOC16 points:          {len(metrics.rloc16_from_transition_timestamps)}")
        print(f"  Parent RLOC16 queries:       {len(metrics.parent_rloc16_from_query_timestamps)}")
        print(f"  Parent router points:        {len(metrics.parent_router_from_rloc16_transition_timestamps)}")
        print(f"  Loss events:                 {len(metrics.ping_packet_loss_timestamps)}")

        if metrics.total_ping_tx_packets > 0:
            pdr = 100.0 * metrics.total_ping_rx_packets / metrics.total_ping_tx_packets
            print(f"  Total TX:                    {metrics.total_ping_tx_packets}")
            print(f"  Total RX:                    {metrics.total_ping_rx_packets}")
            print(f"  Overall PDR:                 {pdr:.2f}%")
        else:
            print("  No packet summary lines found (TX/RX totals unavailable).")


        # Show generic visualization for this file.
        visualize_metrics(metrics, title=rel_label)


if __name__ == "__main__":
    main()
