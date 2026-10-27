#!/usr/bin/env python3
"""
Invisible Geofence & Auto-Lock Security System for YDLIDAR X4
Monitors physical perimeter zones, filters static background furniture,
and triggers automatic screen locking, monitor blanking, and webcam intruder capture.
"""

import os
import sys
import math
import time
import shutil
import argparse
import subprocess
from datetime import datetime

# Optional OpenCV for webcam capture
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# YDLIDAR Driver
try:
    import ydlidar
    HAS_YDLIDAR = True
except ImportError:
    HAS_YDLIDAR = False


class GeofenceSecurity:
    def __init__(self, port="/dev/ttyUSB0", baud=128000,
                 min_dist=0.3, max_dist=2.5,
                 min_angle=-60.0, max_angle=60.0,
                 min_cluster_pts=6,
                 dry_run=False,
                 simulate=False,
                 webcam_id=0):
        self.port = port
        self.baud = baud
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_cluster_pts = min_cluster_pts
        self.dry_run = dry_run
        self.simulate = simulate
        self.webcam_id = webcam_id

        self.laser = None
        self.baseline_map = {}  # deg -> baseline distance in meters
        self.capture_dir = os.path.expanduser("~/Desktop/LIDAR_WS/security_captures")
        os.makedirs(self.capture_dir, exist_ok=True)
        self.log_file = os.path.join(self.capture_dir, "security_audit.log")

    def log(self, message):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        formatted = f"{timestamp} {message}"
        print(formatted)
        try:
            with open(self.log_file, "a") as f:
                f.write(formatted + "\n")
        except Exception:
            pass

    def initialize_hardware(self):
        if self.simulate:
            self.log("[SIMULATION] Simulation mode active - no hardware needed.")
            return True

        if not HAS_YDLIDAR:
            self.log("[-] YDLIDAR SDK python module not found. Falling back to simulation.")
            self.simulate = True
            return True

        self.log(f"[*] Initializing YDLIDAR on {self.port} @ {self.baud} baud...")
        ydlidar.os_init()
        self.laser = ydlidar.CYdLidar()
        self.laser.setlidaropt(ydlidar.LidarPropSerialPort, self.port)
        self.laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, self.baud)
        self.laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        self.laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        self.laser.setlidaropt(ydlidar.LidarPropScanFrequency, 7.0)
        self.laser.setlidaropt(ydlidar.LidarPropSampleRate, 5)
        self.laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
        self.laser.setlidaropt(ydlidar.LidarPropIntenstiy, False)
        self.laser.setlidaropt(ydlidar.LidarPropSupportMotorDtrCtrl, True)
        self.laser.setlidaropt(ydlidar.LidarPropFixedResolution, False)
        self.laser.setlidaropt(ydlidar.LidarPropAutoReconnect, True)

        if not self.laser.initialize() or not self.laser.turnOn():
            self.log("[-] Failed to initialize hardware. Falling back to simulation.")
            self.simulate = True
            return True

        self.log("[+] YDLIDAR initialized and motor spinning.")
        return True

    def calibrate_baseline(self, duration_sec=3.0):
        """Scans stationary room geometry to ignore walls, desks, and open doors."""
        self.log(f"[*] Learning static background baseline ({duration_sec:.1f}s)... Please do not move.")
        baseline = {}

        if self.simulate:
            # Synthetic room walls ~ 2.5m
            for deg in range(0, 360):
                baseline[deg] = 2.4 + 0.2 * math.cos(math.radians(deg * 2))
            self.baseline_map = baseline
            time.sleep(1.0)
            self.log(f"[+] Baseline calibration complete ({len(baseline)} azimuth bins recorded).")
            return

        scan = ydlidar.LaserScan()
        end_time = time.time() + duration_sec
        samples_collected = 0

        while time.time() < end_time and ydlidar.os_isOk():
            if self.laser.doProcessSimple(scan):
                samples_collected += 1
                for p in scan.points:
                    r = p.range
                    if 0.15 <= r <= 10.0:
                        deg = int(math.degrees(p.angle)) % 360
                        if deg not in baseline or r < baseline[deg]:
                            baseline[deg] = r
            time.sleep(0.005)

        self.baseline_map = baseline
        self.log(f"[+] Baseline calibration complete with {samples_collected} scans ({len(baseline)} degree bins).")

    def capture_intruder_photo(self):
        """Grabs an instant webcam snapshot of the room/intruder."""
        if not HAS_CV2:
            self.log("[!] OpenCV (cv2) not available; skipping photo capture.")
            return None

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"breach_{timestamp_str}.jpg"
        filepath = os.path.join(self.capture_dir, filename)

        try:
            cam = cv2.VideoCapture(self.webcam_id)
            if not cam.isOpened():
                self.log(f"[!] Could not open webcam index {self.webcam_id}.")
                return None

            # Warmup camera sensor for 2 frames
            for _ in range(2):
                cam.read()

            ret, frame = cam.read()
            cam.release()

            if ret and frame is not None:
                # Add timestamp watermark
                cv2.putText(frame, f"SECURITY BREACH: {timestamp_str}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(filepath, frame)
                self.log(f"[+] Intruder photo saved: {filepath}")
                return filepath
        except Exception as e:
            self.log(f"[!] Webcam capture error: {e}")

        return None

    def trigger_system_lockdown(self, dist, angle):
        """Executes OS-level lock, screen blanking, and alerts."""
        self.log(f"[ALERT] *** INTRUDER BREACH AT {dist:.2f}m ({angle:.1f}°) ***")

        # 1. Desktop Notification
        try:
            subprocess.run([
                "notify-send",
                "SECURITY PERIMETER BREACH",
                f"Intruder detected at {dist:.2f}m ({angle:.1f} deg)",
                "-u", "critical"
            ], check=False)
        except Exception:
            pass

        # 2. Capture Webcam Snapshot
        photo_path = self.capture_intruder_photo()

        # 3. Dry-Run Check
        if self.dry_run:
            self.log("[DRY-RUN] Screen lock and monitor blanking skipped (--dry-run).")
            return

        # 4. Blank Displays (DPMS Off)
        if shutil.which("hyprctl"):
            subprocess.run(["hyprctl", "dispatch", "dpms", "off"], check=False)
        elif shutil.which("xset"):
            subprocess.run(["xset", "dpms", "force", "off"], check=False)

        # 5. Lock Linux Session
        if shutil.which("hyprlock"):
            self.log("[ACTION] Executing hyprlock...")
            subprocess.Popen(["hyprlock"])
        elif shutil.which("xflock4"):
            self.log("[ACTION] Executing xflock4...")
            subprocess.Popen(["xflock4"])
        elif shutil.which("swaylock"):
            self.log("[ACTION] Executing swaylock...")
            subprocess.Popen(["swaylock"])
        else:
            self.log("[ACTION] Executing loginctl lock-session...")
            subprocess.Popen(["loginctl", "lock-session"])

    def arm_perimeter(self, cooldown_sec=10.0):
        """Continuously checks the active zone against the baseline map."""
        self.log("\n==============================================")
        self.log("      GEOFENCE PERIMETER MONITOR ARMED        ")
        self.log(f"  Zone: {self.min_dist:.2f}m to {self.max_dist:.2f}m | Angle: {self.min_angle:.0f}° to {self.max_angle:.0f}°")
        self.log("  Press Ctrl+C to disarm.")
        self.log("==============================================\n")

        scan = ydlidar.LaserScan() if HAS_YDLIDAR and not self.simulate else None
        last_trigger_time = 0.0
        sim_start_time = time.time()

        try:
            while True:
                intruder_cluster = []

                if self.simulate:
                    time.sleep(0.1)
                    elapsed = time.time() - sim_start_time
                    # Trigger simulated intrusion after 6 seconds
                    if elapsed > 6.0:
                        intruder_cluster = [(1.35, 12.0)] * 10
                else:
                    if not self.laser.doProcessSimple(scan):
                        time.sleep(0.001)
                        continue

                    for p in scan.points:
                        r = p.range
                        if self.min_dist <= r <= self.max_dist:
                            deg = math.degrees(p.angle)
                            # Normalize angle to [-180, +180]
                            if deg > 180.0:
                                deg -= 360.0

                            if self.min_angle <= deg <= self.max_angle:
                                deg_idx = int(deg) % 360
                                baseline_r = self.baseline_map.get(deg_idx, 99.0)

                                # Significant deviation from static wall/doorway (at least 20cm closer)
                                if (baseline_r - r) > 0.20:
                                    intruder_cluster.append((r, deg))

                # If a dense cluster of points is inside the geofence
                if len(intruder_cluster) >= self.min_cluster_pts:
                    now = time.time()
                    if now - last_trigger_time > cooldown_sec:
                        avg_dist = sum(pt[0] for pt in intruder_cluster) / len(intruder_cluster)
                        avg_deg = sum(pt[1] for pt in intruder_cluster) / len(intruder_cluster)
                        self.trigger_system_lockdown(avg_dist, avg_deg)
                        last_trigger_time = now

                        if self.simulate:
                            self.log("[SIMULATION] Simulation trigger cycle complete. Exiting...")
                            break
                    else:
                        sys.stdout.write(f"\r[ALERT ACTIVE - COOLDOWN] Cluster points: {len(intruder_cluster):2d}   ")
                        sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r[ARMED & SCANNING] Perimeter secure. (Monitoring {self.min_dist:.1f}m - {self.max_dist:.1f}m)   ")
                    sys.stdout.flush()

        except KeyboardInterrupt:
            self.log("\n[+] Disarming system...")
        finally:
            if self.laser:
                self.laser.turnOff()
                self.laser.disconnecting()
            self.log("[+] Geofence disarmed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="YDLIDAR Geofence & Auto-Lock Security")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port")
    parser.add_argument("--baud", type=int, default=128000, help="Baud rate (default: 128000)")
    parser.add_argument("--min-dist", type=float, default=0.35, help="Min distance in meters (default: 0.35)")
    parser.add_argument("--max-dist", type=float, default=2.5, help="Max distance in meters (default: 2.5)")
    parser.add_argument("--min-angle", type=float, default=-60.0, help="Min angle in degrees (default: -60)")
    parser.add_argument("--max-angle", type=float, default=60.0, help="Max angle in degrees (default: +60)")
    parser.add_argument("--countdown", type=int, default=5, help="Exit countdown seconds before arming (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="Log breach & capture webcam without locking screen")
    parser.add_argument("--simulate", action="store_true", help="Simulate hardware & intrusion for testing")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    args = parser.parse_args()

    security = GeofenceSecurity(
        port=args.port,
        baud=args.baud,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        dry_run=args.dry_run,
        simulate=args.simulate,
        webcam_id=args.camera
    )

    if not security.initialize_hardware():
        sys.exit(1)

    security.calibrate_baseline(duration_sec=3.0)

    if args.countdown > 0:
        print(f"\n[*] Arming in {args.countdown} seconds. Please step outside the perimeter...")
        for i in range(args.countdown, 0, -1):
            sys.stdout.write(f"\rArming in {i}... ")
            sys.stdout.flush()
            time.sleep(1.0)
        print("ARMED!\n")

    security.arm_perimeter()


if __name__ == "__main__":
    main()
