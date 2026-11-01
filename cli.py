#!/usr/bin/env python3
"""CLI entry point for Persistent 2D Lidar Mapping System with YDLIDAR TG50."""

import os
import sys

# Auto-re-execute inside .venv if run with system python
if "VIRTUAL_ENV" not in os.environ:
    _venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
    if os.path.exists(_venv_py) and os.path.abspath(sys.executable) != os.path.abspath(_venv_py):
        os.execv(_venv_py, [_venv_py] + sys.argv)

import argparse
import asyncio
import logging
import math
import signal
import threading
import time
from typing import Any, Dict
import numpy as np
import uvicorn
import yaml

from driver.factory import create_lidar_source
from driver.fake_lidar import FakeLidar
from driver.interface import LidarSource, Scan
from driver.simulator import SimulationLidar, SimulatedEnvironment
from driver.tg50_serial import TG50SerialDriver
from persistence.ros_exporter import RosMapExporter
from slam.system import LidarSlamSystem
from ui.web_app import create_web_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_check_lidar(args):
    """Hardware diagnostic tool: tests connection, measures scan rate and point count."""
    port = args.port or "/dev/ttyUSB0"
    baud = args.baud or 512000
    print("\n=======================================================")
    print(f" YDLIDAR TG50 Hardware Bring-Up Diagnostic Check")
    print("=======================================================")
    print(f"Target Port: {port}")
    print(f"Baud Rate:   {baud} (Verified TG50 specification)")
    print(f"Protocol:    8N1, Header 0x55AA, Direct ToF mm\n")

    driver = TG50SerialDriver(port=port, baudrate=baud, auto_detect=True)
    try:
        driver.start()
        if not driver.is_connected():
            print(f"[FAIL] Could not open port {port}.")
            print("Troubleshooting steps:")
            print("  1. Check cable connection to PH2.0-5P adapter board.")
            print("  2. Ensure USB adapter is plugged into high-current USB 3.0 port (5V 1.0A+).")
            print("  3. Set Linux permissions: sudo usermod -aG dialout $USER && newgrp dialout")
            print(f"  4. Detected serial ports: {TG50SerialDriver.find_serial_ports()}")
            sys.exit(1)

        print("[OK] Serial connection established and motor started.")
        print("Collecting 10 sample revolutions...\n")

        scan_times = []
        point_counts = []
        ranges_all = []
        intensities_all = []

        last_time = time.time()
        collected = 0

        while collected < 10:
            scan = driver.get_scan(timeout=2.0)
            if scan is None:
                print(".", end="", flush=True)
                continue

            now = time.time()
            dt = now - last_time
            last_time = now
            collected += 1

            scan_times.append(dt)
            point_counts.append(len(scan.ranges_m))
            valid = scan.ranges_m > 0.05
            if np.any(valid):
                ranges_all.extend(scan.ranges_m[valid].tolist())
            if scan.intensities is not None:
                intensities_all.extend(scan.intensities.tolist())

            freq = 1.0 / dt if dt > 0 else 0.0
            print(f"  [Scan {collected}/10] Points: {len(scan.ranges_m):4d} | Frequency: {freq:4.1f} Hz")

        avg_freq = 1.0 / np.mean(scan_times)
        avg_pts = np.mean(point_counts)
        min_r = np.min(ranges_all) if ranges_all else 0.0
        max_r = np.max(ranges_all) if ranges_all else 0.0
        avg_int = np.mean(intensities_all) if intensities_all else 0.0

        print("\n---------------- Diagnostic Summary -------------------")
        print(f"Average Scan Frequency: {avg_freq:.2f} Hz (Expected TG50 nominal: 7.0 Hz)")
        print(f"Average Points / Rev:   {avg_pts:.0f} points (Sampling rate: ~{avg_pts * avg_freq / 1000:.1f} kHz)")
        print(f"Observed Range Extent:  {min_r:.2f} m to {max_r:.2f} m")
        print(f"Average Intensity:      {avg_int:.1f} / 255")
        print("Hardware Status:        READY & OPERATIONAL")
        print("=======================================================\n")

    finally:
        driver.stop()


def cmd_record(args):
    """Record raw scans to disk."""
    config = load_config(args.config)
    from recorder.scan_recorder import ScanRecorder

    output_path = args.output or f"recorded_scans_{int(time.time())}.npz"
    fmt = "jsonl" if output_path.endswith(".jsonl") else "npz"

    source = create_lidar_source(config)
    recorder = ScanRecorder(output_path, format=fmt)
    recorder.start()
    source.start()

    print(f"Recording raw scans from {source.describe()} to {output_path} (Ctrl+C to stop)...")
    count = 0
    try:
        while True:
            scan = source.get_scan(timeout=1.0)
            if scan is not None:
                recorder.record_scan(scan)
                count += 1
                if count % 10 == 0:
                    print(f"Recorded {count} scans...", end="\r", flush=True)
    except KeyboardInterrupt:
        print(f"\nStopped recording. Total scans saved: {count}")
    finally:
        source.stop()
        recorder.close()


def run_slam_loop(slam_system: LidarSlamSystem, source: LidarSource, app: Any, config: Dict[str, Any]):
    """Background SLAM processing loop."""
    source.start()
    ui_cfg = config.get("ui", {})
    fps = float(ui_cfg.get("stream_fps", 15))
    dt_target = 1.0 / fps

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received. Saving session...")
        try:
            slam_system.save_map("autosave_shutdown")
        except Exception:
            pass
        source.stop()
        sys.exit(0)

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        t0 = time.time()
        scan = source.get_scan(timeout=1.0)
        if scan is None:
            time.sleep(0.01)
            continue

        result = slam_system.process_scan(scan)

        # Periodic autosave
        if slam_system.map_manager.should_autosave():
            slam_system.save_map("autosave_periodic")

        # Prepare UI payload
        submaps_summary = []
        for s in slam_system.submaps:
            # Extract occupied cells in world coordinates for display
            prob = s.grid.get_probability_map()
            occ_mask = prob >= s.grid.occupied_thresh
            if np.any(occ_mask):
                rows, cols = np.where(occ_mask)
                cells_w = [list(s.grid.grid_to_world(c, r)) for c, r in zip(cols[::2], rows[::2])]
            else:
                cells_w = []

            submaps_summary.append({
                "id": s.submap_id,
                "is_active": (s == slam_system.active_submap),
                "is_unlinked": (s.session_id != "session_0"),
                "cells": cells_w,
            })

        payload = {
            "type": "telemetry",
            "pose": result["pose"].tolist(),
            "status": result["status"],
            "confidence": result["confidence"],
            "is_degenerate": result["is_degenerate"],
            "mean_residual": result.get("mean_residual", 0.02),
            "num_submaps": result["num_submaps"],
            "num_keyframes": result["num_keyframes"],
            "pts_world": result.get("pts_world", np.zeros((0, 2))).tolist(),
            "submaps": submaps_summary,
        }

        # Broadcast to web UI
        if hasattr(app.state, "broadcast_telemetry"):
            try:
                loop.run_until_complete(app.state.broadcast_telemetry(payload))
            except Exception:
                pass

        elapsed = time.time() - t0
        sleep_dur = dt_target - elapsed
        if sleep_dur > 0:
            time.sleep(sleep_dur)


def cmd_run(args):
    """Run full live SLAM with Lidar and Web UI."""
    config = load_config(args.config)
    if args.port:
        config["lidar"]["port"] = args.port
    if args.baud:
        config["lidar"]["baudrate"] = int(args.baud)

    slam_system = LidarSlamSystem(config)
    if args.load_map:
        slam_system.load_map(args.load_map)

    source = create_lidar_source(config)
    app = create_web_app(slam_system)

    # Start background SLAM thread
    slam_thread = threading.Thread(
        target=run_slam_loop,
        args=(slam_system, source, app, config),
        daemon=True,
    )
    slam_thread.start()

    host = config.get("ui", {}).get("web_host", "127.0.0.1")
    port = int(config.get("ui", {}).get("web_port", 8080))
    print(f"\n=======================================================")
    print(f" Persistent 2D Lidar SLAM System Running")
    print(f" Web UI: http://{host}:{port}")
    print(f" Lidar:  {source.describe()}")
    print("=======================================================\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")


def cmd_simulate(args):
    """Run multi-room simulation demo with Web UI."""
    config = load_config(args.config)
    config["lidar"]["driver_mode"] = "sim"
    slam_system = LidarSlamSystem(config)

    sim = SimulationLidar(num_beams=720, scan_rate_hz=10.0, noise_std_m=0.015)
    app = create_web_app(slam_system)

    # Robot movement trajectory simulating exploring Room A then passing into Room B
    def mover_thread():
        time.sleep(1.0)
        # Waypoints: Room A -> Doorway -> Room B
        path = [
            (2.5, 2.5, 0.0), (3.0, 2.5, 0.0), (3.5, 2.8, 0.1), (4.0, 2.5, 0.0),
            (5.0, 2.5, 0.0), (6.0, 2.6, 0.0), (7.0, 2.6, 0.0), (8.5, 3.0, 0.0),
            (10.0, 3.5, 0.1), (11.5, 4.0, 0.0), (12.0, 2.5, -0.2), (10.0, 2.5, 0.0),
            (7.5, 2.6, 0.0), (6.0, 2.6, 0.0), (4.0, 2.5, 0.0), (2.5, 2.5, 0.0)
        ]
        while True:
            for i in range(len(path) - 1):
                p0, p1 = path[i], path[i + 1]
                steps = 15
                for s in range(steps):
                    alpha = s / float(steps)
                    cx = p0[0] + (p1[0] - p0[0]) * alpha
                    cy = p0[1] + (p1[1] - p0[1]) * alpha
                    cth = p0[2] + (p1[2] - p0[2]) * alpha
                    sim.set_pose(cx, cy, cth)
                    time.sleep(0.1)

    t_move = threading.Thread(target=mover_thread, daemon=True)
    t_move.start()

    t_slam = threading.Thread(
        target=run_slam_loop,
        args=(slam_system, sim, app, config),
        daemon=True,
    )
    t_slam.start()

    host = config.get("ui", {}).get("web_host", "127.0.0.1")
    port = int(config.get("ui", {}).get("web_port", 8080))
    print(f"\n=======================================================")
    print(f" Multi-Room Simulation Demo Running")
    print(f" Web UI: http://{host}:{port}")
    print("=======================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def cmd_replay(args):
    """Replay recorded scan log."""
    config = load_config(args.config)
    slam_system = LidarSlamSystem(config)
    fake = FakeLidar(log_path=args.file, realtime=True)
    app = create_web_app(slam_system)

    t_slam = threading.Thread(
        target=run_slam_loop,
        args=(slam_system, fake, app, config),
        daemon=True,
    )
    t_slam.start()

    host = config.get("ui", {}).get("web_host", "127.0.0.1")
    port = int(config.get("ui", {}).get("web_port", 8080))
    print(f"\n=======================================================")
    print(f" Replay Log Running: {args.file}")
    print(f" Web UI: http://{host}:{port}")
    print("=======================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def cmd_export(args):
    """Export saved session to ROS PNG + YAML."""
    config = load_config(args.config)
    slam_system = LidarSlamSystem(config)
    slam_system.load_map(args.name)
    out_dir = args.output_dir or os.path.join(config.get("persistence", {}).get("maps_dir", "maps"), args.name)
    y_path, p_path = RosMapExporter.export(slam_system.global_grid, out_dir, args.name)
    print(f"Exported ROS map successfully:\n  YAML: {y_path}\n  PNG:  {p_path}")


def main():
    parser = argparse.ArgumentParser(description="Persistent 2D Lidar Mapping System with YDLIDAR TG50")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    subparsers = parser.add_subparsers(dest="command")

    # check-lidar
    p_check = subparsers.add_parser("check-lidar", help="Hardware diagnostic check")
    p_check.add_argument("--port", help="Serial port (/dev/ttyUSB0, COM3)")
    p_check.add_argument("--baud", type=int, default=512000, help="Baud rate (default: 512000)")

    # run
    p_run = subparsers.add_parser("run", help="Run live SLAM system")
    p_run.add_argument("--port", help="Serial port")
    p_run.add_argument("--baud", type=int, help="Baud rate")
    p_run.add_argument("--load-map", help="Load previously saved map for relocalization")

    # record
    p_rec = subparsers.add_parser("record", help="Record raw scans to disk")
    p_rec.add_argument("--output", "-o", help="Output path (.npz or .jsonl)")

    # replay
    p_rep = subparsers.add_parser("replay", help="Replay recorded scan log")
    p_rep.add_argument("file", help="Path to .npz or .jsonl file")

    # simulate
    subparsers.add_parser("simulate", help="Run multi-room simulation demo")

    # export
    p_exp = subparsers.add_parser("export", help="Export saved map to ROS map_server format")
    p_exp.add_argument("name", help="Name of saved map session")
    p_exp.add_argument("--output-dir", help="Destination folder")

    args = parser.parse_args()

    if args.command == "check-lidar":
        cmd_check_lidar(args)
    elif args.command == "record":
        cmd_record(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
