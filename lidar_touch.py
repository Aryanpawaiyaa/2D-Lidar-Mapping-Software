#!/usr/bin/env python3
"""
YDLIDAR X4 Live Reader & Test Script
Direct binding to /dev/ttyUSB0 in Single-Channel (one-way) mode.
"""

import sys
import time
import ydlidar


def main():
    print("==========================================")
    print("      YDLIDAR X4 Live Stream Reader       ")
    print("==========================================")

    # Initialize SDK OS environment
    ydlidar.os_init()

    # Configure Laser parameters for YDLIDAR X4
    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, "/dev/ttyUSB0")
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 7.0)
    laser.setlidaropt(ydlidar.LidarPropSampleRate, 5)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
    laser.setlidaropt(ydlidar.LidarPropIntenstiy, False)
    laser.setlidaropt(ydlidar.LidarPropSupportMotorDtrCtrl, True)

    # Initialize Driver
    ret = laser.initialize()
    if not ret:
        print("[!] Error: Failed to initialize LiDAR on /dev/ttyUSB0.")
        return 1

    print("[+] Successfully initialized driver. Turning on motor...")
    ret = laser.turnOn()
    if not ret:
        print("[!] Error: Failed to spin up LiDAR.")
        print("    Please check that external 5V power is supplied to the USB adapter.")
        laser.disconnecting()
        return 1

    print("[+] LiDAR is active. Streaming scans (Ctrl+C to quit)...")
    scan = ydlidar.LaserScan()

    try:
        scan_count = 0
        while ydlidar.os_isOk():
            r = laser.doProcessSimple(scan)
            if r:
                scan_count += 1
                freq = 1.0 / scan.config.scan_time if scan.config.scan_time > 0 else 0
                points = scan.points
                num_points = points.size()
                
                # Sample min/max distances in current scan
                valid_ranges = [p.range for p in points if p.range > 0.05]
                min_dist = min(valid_ranges) if valid_ranges else 0.0
                max_dist = max(valid_ranges) if valid_ranges else 0.0
                
                print(f"Scan #{scan_count:04d} | Points: {num_points:3d} | Freq: {freq:4.1f} Hz | Range: {min_dist:5.2f}m - {max_dist:5.2f}m")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[+] Interrupted by user. Shutting down...")
    finally:
        laser.turnOff()
        laser.disconnecting()
        print("[+] LiDAR disconnected cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
