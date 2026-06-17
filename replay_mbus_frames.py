#!/usr/bin/env python3
"""
Replay captured M-Bus/DLMS HDLC frames from mbus_frames.log to a USB-TTL port.

The capture file is binary. Frames are delimited by HDLC flag bytes (0x7e);
newline bytes between captured frames are ignored, and newline bytes inside a
frame are preserved.

Usage:
    python3 replay_mbus_frames.py /dev/ttyUSB0
    python3 replay_mbus_frames.py /dev/cu.usbserial-0001 --all --interval 5
    python3 replay_mbus_frames.py COM3 --loop

Requires:
    pip install pyserial
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # pragma: no cover - depends on local environment
    serial = None


DEFAULT_BAUD = 2400
DEFAULT_INTERVAL_SECONDS = 0.0
DEFAULT_FRAME_GAP_SECONDS = 0.05
DEFAULT_WRITE_TIMEOUT_SECONDS = 10.0


def extract_hdlc_frames(data: bytes) -> list[bytes]:
    """Return HDLC frames using the HDLC format length field."""
    frames: list[bytes] = []
    pos = 0

    while True:
        start = data.find(b"\x7e", pos)
        if start == -1:
            break

        if start + 3 > len(data):
            break

        # IEC 62056 HDLC uses a 2-byte format field after the opening flag.
        # The low 11 bits are the frame length excluding the two flag bytes.
        fmt = (data[start + 1] << 8) | data[start + 2]
        declared_len = fmt & 0x07FF
        end = start + declared_len + 1

        if (data[start + 1] & 0xF0) != 0xA0 or end >= len(data):
            pos = start + 1
            continue

        if data[end] != 0x7e:
            pos = start + 1
            continue

        frame = data[start : end + 1]
        if len(frame) > 2:
            frames.append(frame)

        pos = end + 1

    return frames


def hdlc_info(frame: bytes) -> bytes:
    """Return the HDLC information field, without HCS/FCS."""
    body = frame[1:-1]
    pos = 2

    while pos < len(body) and (body[pos] & 1) == 0:
        pos += 1
    pos += 1

    while pos < len(body) and (body[pos] & 1) == 0:
        pos += 1
    pos += 1

    pos += 1  # control
    info = body[pos:-2]
    return info[2:] if len(info) >= 2 else b""


def gbt_block_info(frame: bytes) -> tuple[int, bool] | None:
    """Return (GBT block number, is_last_block), or None for non-GBT frames."""
    info = hdlc_info(frame)
    pos = 3 if info.startswith(b"\xe6\xe7\x00") else 0
    if pos >= len(info) or info[pos] != 0xE0:
        return None
    pos += 1
    if len(info) - pos < 5:
        return None

    control = info[pos]
    block_no = (info[pos + 1] << 8) | info[pos + 2]
    return block_no, bool(control & 0x80)


def group_gbt_frames(frames: list[bytes]) -> list[list[bytes]]:
    """Group consecutive frames that form one GBT notification."""
    groups: list[list[bytes]] = []
    current: list[bytes] = []
    last_block_no: int | None = None

    for frame in frames:
        gbt = gbt_block_info(frame)
        if gbt is None:
            if current:
                groups.append(current)
                current = []
                last_block_no = None
            groups.append([frame])
            continue

        block_no, is_last = gbt
        if block_no == 1 and current:
            groups.append(current)
            current = []
            last_block_no = None
        elif current and last_block_no is not None and block_no != last_block_no + 1:
            groups.append(current)
            current = []
            last_block_no = None

        current.append(frame)
        last_block_no = block_no
        if is_last:
            groups.append(current)
            current = []
            last_block_no = None

    if current:
        groups.append(current)
    return groups


def format_hex_preview(frame: bytes, max_bytes: int = 16) -> str:
    preview = " ".join(f"{b:02X}" for b in frame[:max_bytes])
    if len(frame) > max_bytes:
        preview += " ..."
    return preview


def list_ports() -> str:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return "  (no serial ports found)"
    return "\n".join(f"  {p.device} - {p.description}" for p in ports)


def outgoing_port(port: str) -> str:
    """Prefer macOS callout devices for scripts that initiate serial writes."""
    if not port.startswith("/dev/tty."):
        return port

    cu_port = "/dev/cu." + port[len("/dev/tty.") :]
    return cu_port if Path(cu_port).exists() else port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay mbus_frames.log frames to a USB-TTL serial adapter."
    )
    parser.add_argument("port", help="Serial port, e.g. /dev/ttyUSB0, COM3")
    parser.add_argument(
        "-i",
        "--input",
        default="mbus_frames.log",
        type=Path,
        help="Binary capture file to replay (default: mbus_frames.log)",
    )
    parser.add_argument(
        "-b",
        "--baud",
        default=DEFAULT_BAUD,
        type=int,
        help=f"Serial baud rate (default: {DEFAULT_BAUD})",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL_SECONDS,
        type=float,
        help=(
            "Delay between complete GBT push groups in seconds "
            f"(default: {DEFAULT_INTERVAL_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Replay every group from the capture once instead of only the first group",
    )
    parser.add_argument(
        "--frame-gap",
        default=DEFAULT_FRAME_GAP_SECONDS,
        type=float,
        help=(
            "Delay between HDLC frames inside one GBT push group in seconds "
            f"(default: {DEFAULT_FRAME_GAP_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--write-timeout",
        default=DEFAULT_WRITE_TIMEOUT_SECONDS,
        type=float,
        help=(
            "Serial write timeout in seconds; use 0 to disable pyserial's write "
            f"timeout (default: {DEFAULT_WRITE_TIMEOUT_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Replay the capture repeatedly until interrupted",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print frames without opening the serial port",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.interval < 0 or args.frame_gap < 0 or args.write_timeout < 0:
        print("Interval, frame gap, and write timeout must be >= 0", file=sys.stderr)
        return 2

    try:
        data = args.input.read_bytes()
    except OSError as exc:
        print(f"Could not read {args.input}: {exc}", file=sys.stderr)
        return 1

    frames = extract_hdlc_frames(data)
    if not frames:
        print(f"No HDLC frames found in {args.input}", file=sys.stderr)
        return 1
    groups = group_gbt_frames(frames)
    replay_groups = groups if args.all or args.loop else groups[:1]

    print(
        f"Loaded {len(frames)} frames in {len(groups)} group(s) from {args.input} "
        f"({sum(len(frame) for frame in frames)} bytes)."
    )
    if replay_groups != groups:
        print("Replaying first group only; pass --all to replay every captured group.")

    ser = None
    if not args.dry_run:
        if serial is None:
            print("pyserial is not installed. Install it with: pip install pyserial")
            return 1
        port = outgoing_port(args.port)
        if port != args.port:
            print(f"Using {port} for outgoing serial instead of {args.port}.")
        try:
            ser = serial.Serial()
            ser.port = port
            ser.baudrate = args.baud
            ser.bytesize = 8
            ser.parity = "N"
            ser.stopbits = 1
            ser.timeout = 1
            ser.write_timeout = args.write_timeout or None
            ser.xonxoff = False
            ser.rtscts = False
            ser.dsrdtr = False
            ser.dtr = False
            ser.rts = False
            ser.open()
        except serial.SerialException as exc:
            print(f"Could not open {args.port}: {exc}", file=sys.stderr)
            print("Available ports:", file=sys.stderr)
            print(list_ports(), file=sys.stderr)
            return 1

        ser.reset_output_buffer()
        print(f"Opened {port} @ {args.baud} 8N1.")
        if port == args.port and args.port.startswith("/dev/tty."):
            cu_port = "/dev/cu." + args.port[len("/dev/tty.") :]
            print(f"Note: on macOS, {cu_port} is usually better for outgoing serial.")

    sent = 0
    pass_no = 0

    try:
        while True:
            pass_no += 1
            for group_index, group in enumerate(replay_groups, start=1):
                print(
                    f"pass {pass_no}, group {group_index}/{len(replay_groups)}: "
                    f"{len(group)} frame(s)"
                )
                for frame_index, frame in enumerate(group, start=1):
                    label = f"  frame {frame_index}/{len(group)}"
                    print(f"{label}: {len(frame)} bytes  {format_hex_preview(frame)}")

                    if ser is not None:
                        try:
                            ser.write(frame)
                            ser.flush()
                        except serial.SerialTimeoutException as exc:
                            print(f"{label}: serial write timed out: {exc}", file=sys.stderr)
                            print(
                                "If this is macOS, try the matching /dev/cu.* port instead "
                                "of /dev/tty.*. Also check that the receiving device is "
                                "running and reading the UART.",
                                file=sys.stderr,
                            )
                            return 1
                    sent += 1

                    if args.frame_gap and frame_index < len(group):
                        time.sleep(args.frame_gap)

                if args.interval and (args.loop or group_index < len(replay_groups)):
                    time.sleep(args.interval)

            if not args.loop:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if ser is not None:
            ser.close()

    action = "Would replay" if args.dry_run else "Replayed"
    print(f"{action} {sent} frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
