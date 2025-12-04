#!/usr/bin/env python3
import datetime
import os
import threading
import time
from typing import Dict

import serial
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo


BAUDRATE = 115200
SCAN_INTERVAL = 2.0  # seconds between rescans for new/removed devices


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

    def run(self) -> None:
        port_name = self.port_info.device
        log_path = self.make_log_path()

        try:
            ser = serial.Serial(port_name, self.baudrate, timeout=1)
        except serial.SerialException as exc:
            print(f"[{port_name}] ERROR: could not open serial port: {exc}")
            return

        print(f"[{port_name}] Logging UART to {log_path}")

        try:
            with ser, open(log_path, "a", encoding="utf-8") as log_file:
                while not self.stop_event.is_set():
                    try:
                        data = ser.readline()  # read until newline or timeout
                    except serial.SerialException as exc:
                        print(f"[{port_name}] Serial error / disconnected: {exc}")
                        break

                    if not data:
                        # timeout without data, just loop again to check stop_event
                        continue

                    timestamp = datetime.datetime.now().isoformat(
                        timespec="milliseconds"
                    )

                    try:
                        text = data.decode("utf-8", errors="replace").rstrip("\r\n")
                    except Exception:
                        text = repr(data)

                    log_file.write(f"[{timestamp}] {text}\n")
                    log_file.flush()
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
    print("ESP32 auto UART logger")
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
