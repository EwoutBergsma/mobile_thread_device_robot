#!/usr/bin/env python3
# Ping code heavily inspired by: https://github.com/kkSeaWater/n0rdic-halfconductor/blob/main/pyserial_esp.py

import datetime
import os
import threading
import time
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

import serial
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo


BAUDRATE = 2000000
SCAN_INTERVAL = 2.0  # seconds between rescans for new/removed devices
PING_INTERVAL = 1.0  # seconds between OT CLI pings to parent

# Number of times to send the initial startup command set (txpower/ipaddr)
STARTUP_CMD_ATTEMPTS = 5
STARTUP_CMD_RETRY_DELAY = 0.1  # seconds between repeated sends of the startup set

# ---------------------------------------------------------------------------
# Startup / periodic command sets
# ---------------------------------------------------------------------------

# Commands sent several times immediately after the serial port is opened.
# If you set this to an empty list, no startup commands will be sent.
STARTUP_COMMANDS = [
    b"txpower -12\r\n",  # set TX power on OpenThread CLI
    b"ipaddr\r\n",       # request IP addresses to learn mesh-local prefix
    b"log level 3\r\n"
]

# Base commands sent on every periodic tick (before optional ping).
# If you set this to an empty list, only the dynamic ping (when available)
# will be sent. If it is empty and there is no ping target, nothing is sent.
PERIODIC_BASE_COMMANDS = [
    b"txpower\r\n",  # query current txpower
    b"state\r\n",    # OT node state
    b"parent\r\n",   # parent info
]

# --- OpenThread parent / RLOC parsing helpers (adapted from pyserial_esp.py) ---


@dataclass
class TelemetryState:
    lock: threading.Lock = field(default_factory=threading.Lock)

    last_state: Optional[str] = None
    parent_rloc16: Optional[str] = None
    mesh_local_prefix: Optional[str] = None
    last_parent_ts: Optional[datetime.datetime] = None


# RLOC IPv6 address, just like Get-MyRlocAddr:
#   ^[0-9a-f:]+:0:ff:fe00:[0-9a-f]{1,4}$
RLOC_ADDR_RE = re.compile(
    r"\b([0-9a-fA-F:]+:0:ff:fe00:[0-9a-fA-F]{1,4})\b",
    re.IGNORECASE,
)

# Only match the simple OT CLI line from `parent`:
#   Rloc: 7000
PARENT_RLOC_RE = re.compile(
    r"^Rloc:\s+([0-9a-fA-F]{1,4})\s*$",
    re.IGNORECASE,
)

STATES = {"child", "router", "leader", "detached", "disabled"}
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;?]*[ -/]*[@-~]")


def build_parent_rloc_ipv6(prefix_ps_style: str, rloc16: str) -> str:
    """Build parent RLOC IPv6 EXACTLY like the PS script."""
    rloc16 = (rloc16 or "").strip()
    try:
        r = int(rloc16, 16)
        r_hex = f"{r:04x}"
    except ValueError:
        r_hex = rloc16.lower()
    return (prefix_ps_style + r_hex).lower()


def process_ot_line(ts_iso: str, line: str, tstate: TelemetryState) -> None:
    """
    Lightweight parser:
      - learn mesh-local prefix from a RLOC address
      - track state (child/router/leader/...)
      - track parent RLOC16 and freshness
    """
    stripped = line.strip()
    st_lower = stripped.lower()

    # --- learn mesh-local prefix from a RLOC address ---
    if tstate.mesh_local_prefix is None:
        m_addr = RLOC_ADDR_RE.search(line)
        if m_addr:
            my_rloc = m_addr.group(1).lower()
            prefix = re.sub(
                r"([0-9a-f:]+:0:ff:fe00:)[0-9a-f]{1,4}$",
                r"\1",
                my_rloc,
            )
            if prefix and prefix != my_rloc:
                with tstate.lock:
                    if tstate.mesh_local_prefix is None:
                        tstate.mesh_local_prefix = prefix
                        print(f"[parse] {ts_iso} mesh-local prefix -> {prefix}")

    # --- state (child/router/leader/detached/disabled) ---
    if st_lower in STATES:
        with tstate.lock:
            tstate.last_state = stripped
            # NOTE: we *do not* clear parent_rloc16 or last_parent_ts here anymore.
            # We keep the last-known parent so we can continue pinging even when
            # detached/disabled, per updated requirements.
        print(f"[parse] {ts_iso} state -> {stripped}")
        return

    # --- parent RLOC16 from 'parent' output ---
    m = PARENT_RLOC_RE.search(stripped)
    if m:
        r = m.group(1)
        try:
            r_hex = f"{int(r, 16):04x}"
        except ValueError:
            r_hex = r.lower()
        now = datetime.datetime.now()
        with tstate.lock:
            tstate.parent_rloc16 = r_hex
            tstate.last_parent_ts = now
        print(f"[parse] {ts_iso} parent RLOC16 -> {r_hex}")


def _sanitize_for_filename(text: str) -> str:
    """Make a safe-ish filename fragment from arbitrary text."""
    import re as _re

    text = text.strip().lower().replace(" ", "_")
    # Keep only a-z, 0-9, underscore, dash
    text = _re.sub(r"[^a-z0-9_\-]+", "", text)
    return text or "device"


class SerialLogger(threading.Thread):
    def __init__(
        self,
        port_info: ListPortInfo,
        baudrate: int = BAUDRATE,
        phase_offset: float = 0.0,
    ):
        super().__init__(daemon=True)
        self.port_info = port_info
        self.baudrate = baudrate
        self.stop_event = threading.Event()
        self.tstate = TelemetryState()  # per-device OT telemetry
        self.phase_offset = phase_offset

    def make_log_path(self) -> str:
        """
        Create a log filename in the current working directory.

        Example (Linux): uart_ttyacm0_20251203_142500.log
        """
        port_name = _sanitize_for_filename(self.port_info.device)
        serial_number = _sanitize_for_filename(
            getattr(self.port_info, "serial_number", "") or ""
        )
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        parts = [p for p in [port_name, serial_number, now] if p]
        filename = "uart_" + "_".join(parts) + ".log"

        # No folder: just in current directory
        return os.path.join(os.getcwd(), filename)

    def _send_periodic_commands(self, ser: serial.Serial) -> None:
        """
        Every PING_INTERVAL seconds, refresh basic OT telemetry and ping the
        parent RLOC address whenever it is known (regardless of current state).
        """
        with self.tstate.lock:
            mesh_prefix = self.tstate.mesh_local_prefix
            parent_rloc16 = self.tstate.parent_rloc16
            last_state = self.tstate.last_state
            last_parent_ts = self.tstate.last_parent_ts

        # Start with the configured periodic base commands.
        cmds = list(PERIODIC_BASE_COMMANDS)

        parent_ip = None

        # As soon as we know mesh-local prefix and parent RLOC16, keep pinging
        # that parent, independent of whether we are currently attached.
        if mesh_prefix and parent_rloc16:
            parent_ip = build_parent_rloc_ipv6(mesh_prefix, parent_rloc16)
            cmds.append(f"ping {parent_ip}\r\n".encode("ascii"))

        # If there is nothing to send (no base commands and no ping), exit early.
        if not cmds:
            return

        try:
            for cmd in cmds:
                ser.write(cmd)

            if parent_ip:
                print(f"[{self.port_info.device}] ping parent {parent_ip}")
        except serial.SerialException as exc:
            print(f"[{self.port_info.device}] ERROR: serial write failed: {exc}")

    def run(self) -> None:
        port_name = self.port_info.device
        log_path = self.make_log_path()

        try:
            # timeout shortened so we can send periodic commands accurately
            ser = serial.Serial(port_name, self.baudrate, timeout=0.1)
        except serial.SerialException as exc:
            print(f"[{port_name}] ERROR: could not open serial port: {exc}")
            return

        print(f"[{port_name}] Logging UART to {log_path}")
        print(
            f"[{port_name}] Will ping OT parent every {PING_INTERVAL:.1f}s when parent RLOC is known."
        )
        print(f"[{port_name}] Ping phase offset: {self.phase_offset:.3f}s")

        try:
            with ser, open(log_path, "a", encoding="utf-8") as log_file:
                # Send startup commands multiple times at connection for robustness.
                if STARTUP_COMMANDS:
                    try:
                        print(
                            f"[{port_name}] Sending startup OT CLI commands "
                            f"(x{STARTUP_CMD_ATTEMPTS})"
                        )
                        for attempt in range(1, STARTUP_CMD_ATTEMPTS + 1):
                            for cmd in STARTUP_COMMANDS:
                                ser.write(cmd)
                                ser.flush()
                            time.sleep(STARTUP_CMD_RETRY_DELAY)
                    except serial.SerialException as exc:
                        print(
                            f"[{port_name}] WARNING: could not send startup commands: {exc}"
                        )
                else:
                    print(
                        f"[{port_name}] No startup commands configured; "
                        "skipping initial send."
                    )

                # Start pings at a per-device offset so multiple devices are de-synchronized.
                next_ping_at = time.time() + self.phase_offset

                while not self.stop_event.is_set():
                    now = time.time()
                    if now >= next_ping_at:
                        self._send_periodic_commands(ser)

                        # Keep a strict 1 s period per device by advancing from the
                        # previous scheduled time, not from "now".
                        next_ping_at += PING_INTERVAL

                        # In case we were delayed and are still behind, catch up.
                        while next_ping_at <= now:
                            next_ping_at += PING_INTERVAL

                    try:
                        data = ser.readline()  # read until newline or timeout
                    except serial.SerialException as exc:
                        print(f"[{port_name}] Serial error / disconnected: {exc}")
                        break

                    if not data:
                        # timeout without data, just loop again to check stop_event / ping timer
                        continue

                    timestamp = datetime.datetime.now().isoformat(
                        timespec="milliseconds"
                    )

                    try:
                        text = data.decode("utf-8", errors="replace").rstrip(
                            "\r\n"
                        )
                        text = ANSI_ESCAPE_RE.sub("", text)
                    except Exception:
                        text = repr(data)

                    # log raw line
                    log_file.write(f"[{timestamp}] {text}\n")
                    log_file.flush()

                    # feed into OT parser to track parent + prefix
                    process_ot_line(timestamp, text, self.tstate)
        finally:
            print(f"[{port_name}] Logger stopped.")

    def stop(self) -> None:
        self.stop_event.set()


def scan_ports() -> Dict[str, ListPortInfo]:
    """
    Return likely USB-style serial ports on Linux.

    Linux: /dev/ttyUSB*, /dev/ttyACM*
    """
    ports: Dict[str, ListPortInfo] = {}

    for p in list_ports.comports():
        dev = p.device or ""

        # Linux: typical USB serial names
        if dev.startswith("/dev/ttyACM") or dev.startswith("/dev/ttyUSB"):
            ports[dev] = p

    return ports


def main():
    print("ESP32 auto UART logger with OT parent ping (Linux only)")
    print(f"Baudrate: {BAUDRATE}")
    print("Log files will be saved in the current directory.")
    print("Press Ctrl+C to stop.\n")

    active_loggers: Dict[str, SerialLogger] = {}

    try:
        while True:
            current_ports = scan_ports()

            # Determine per-device phases for all currently attached devices
            all_devs_sorted = sorted(current_ports.keys())
            total = len(all_devs_sorted)
            phase_map: Dict[str, float] = {}
            if total > 0:
                step = PING_INTERVAL / float(total)
                for idx, dev_name in enumerate(all_devs_sorted):
                    phase_map[dev_name] = idx * step

            # Start loggers for newly seen devices, with their assigned phase offset
            for dev_name, info in current_ports.items():
                if dev_name not in active_loggers:
                    phase_offset = phase_map.get(dev_name, 0.0)
                    print(
                        f"[{dev_name}] Starting logger with phase offset {phase_offset:.3f}s"
                    )
                    logger = SerialLogger(info, BAUDRATE, phase_offset=phase_offset)
                    active_loggers[dev_name] = logger
                    logger.start()

            # Stop loggers for devices that disappeared
            for dev_name in list(active_loggers.keys()):
                if dev_name not in current_ports:
                    logger = active_loggers.pop(dev_name)
                    print(
                        f"[{dev_name}] Device removed, stopping logger..."
                    )
                    logger.stop()

            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopping all loggers...")
    finally:
        # Signal all loggers to stop
        for logger in active_loggers.values():
            logger.stop()

        # Give threads a moment to exit
        for logger in active_loggers.values():
            logger.join(timeout=2.0)

        print("Done.")


if __name__ == "__main__":
    main()
