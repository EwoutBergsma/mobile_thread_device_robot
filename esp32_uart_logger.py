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


BAUDRATE = 115200
SCAN_INTERVAL = 2.0  # seconds between rescans for new/removed devices
PING_INTERVAL = 1.0  # seconds between OT CLI pings to parent


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

PARENT_RLOC_RE = re.compile(r"Rloc:\s*([0-9a-fA-F]{1,4})", re.IGNORECASE)
STATES = {"child", "router", "leader", "detached", "disabled"}
ANSI_ESCAPE_RE = re.compile(r'\x1B\[[0-9;?]*[ -/]*[@-~]')


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
                r'([0-9a-f:]+:0:ff:fe00:)[0-9a-f]{1,4}$',
                r'\1',
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
            if st_lower in ("detached", "disabled"):
                # no valid parent when detached/disabled
                tstate.parent_rloc16 = None
                tstate.last_parent_ts = None
        print(f"[parse] {ts_iso} state -> {stripped}")
        return

    # --- parent RLOC16 from 'parent' output ---
    m = PARENT_RLOC_RE.search(line)
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
    import re

    text = text.strip().lower().replace(" ", "_")
    # Keep only a-z, 0-9, underscore, dash
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    return text or "device"


class SerialLogger(threading.Thread):
    def __init__(self, port_info: ListPortInfo, baudrate: int = BAUDRATE):
        super().__init__(daemon=True)
        self.port_info = port_info
        self.baudrate = baudrate
        self.stop_event = threading.Event()
        self.tstate = TelemetryState()  # NEW: per-device OT telemetry

    def make_log_path(self) -> str:
        """
        Create a log filename in the current working directory.

        Example: uart_ttyacm0_20251203_142500.log
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
        Every PING_INTERVAL seconds, refresh basic OT telemetry and ping the parent
        (if we know its RLOC address).
        """
        with self.tstate.lock:
            mesh_prefix = self.tstate.mesh_local_prefix
            parent_rloc16 = self.tstate.parent_rloc16
            last_state = self.tstate.last_state
            last_parent_ts = self.tstate.last_parent_ts

        cmds = [b"state\r\n", b"parent\r\n"]

        parent_ip = None
        st_lower = (last_state or "").lower()

        # Only ping when we are attached (child/router/leader) and parent info is fresh.
        if (
            mesh_prefix
            and parent_rloc16
            and st_lower in ("child", "router", "leader")
            and last_parent_ts is not None
        ):
            now = datetime.datetime.now()
            if (now - last_parent_ts).total_seconds() <= max(3 * PING_INTERVAL, 3.0):
                parent_ip = build_parent_rloc_ipv6(mesh_prefix, parent_rloc16)
                cmds.append(f"ping {parent_ip}\r\n".encode("ascii"))

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
        print(f"[{port_name}] Will ping OT parent every {PING_INTERVAL:.1f}s when known.")

        try:
            with ser, open(log_path, "a", encoding="utf-8") as log_file:
                # Set txpower once at startup
                try:
                    print(f"[{port_name}] Setting txpower 0 dBm on OpenThread CLI")
                    ser.write(b"txpower 0\r\n")
                    time.sleep(0.1)
                except serial.SerialException as exc:
                    print(f"[{port_name}] WARNING: could not set txpower: {exc}")

                # Ask once for ipaddr so we can learn the mesh-local prefix.
                try:
                    ser.write(b"ipaddr\r\n")
                except serial.SerialException as exc:
                    print(f"[{port_name}] WARNING: could not request ipaddr: {exc}")

                next_ping_at = time.time()

                while not self.stop_event.is_set():
                    now = time.time()
                    if now >= next_ping_at:
                        self._send_periodic_commands(ser)
                        next_ping_at = now + PING_INTERVAL

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
                        text = data.decode("utf-8", errors="replace").rstrip("\r\n")
                        text = ANSI_ESCAPE_RE.sub("", text)
                    except Exception:
                        text = repr(data)

                    # log raw line
                    log_file.write(f"[{timestamp}] {text}\n")
                    log_file.flush()

                    # NEW: feed into OT parser to track parent + prefix
                    process_ot_line(timestamp, text, self.tstate)
        finally:
            print(f"[{port_name}] Logger stopped.")

    def stop(self) -> None:
        self.stop_event.set()


def scan_ports() -> Dict[str, ListPortInfo]:
    """
    Return all USB-style serial ports (/dev/ttyUSB* and /dev/ttyACM* on Linux).
    This will catch your ESP32 boards on the Raspberry Pi.
    """
    ports: Dict[str, ListPortInfo] = {}

    for p in list_ports.comports():
        dev = p.device or ""
        # On Raspberry Pi, ESP32s usually show up as ttyUSB* or ttyACM*.
        if dev.startswith("/dev/ttyACM") or dev.startswith("/dev/ttyUSB"):
            ports[dev] = p

    return ports


def main():
    print("ESP32 auto UART logger with OT parent ping")
    print(f"Baudrate: {BAUDRATE}")
    print("Log files will be saved in the current directory.")
    print("Press Ctrl+C to stop.\n")

    active_loggers: Dict[str, SerialLogger] = {}

    try:
        while True:
            current_ports = scan_ports()

            # Start loggers for newly seen devices
            for dev_name, info in current_ports.items():
                if dev_name not in active_loggers:
                    logger = SerialLogger(info, BAUDRATE)
                    active_loggers[dev_name] = logger
                    logger.start()

            # Stop loggers for devices that disappeared
            for dev_name in list(active_loggers.keys()):
                if dev_name not in current_ports:
                    logger = active_loggers.pop(dev_name)
                    print(f"[{dev_name}] Device removed, stopping logger...")
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
