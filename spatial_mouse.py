#!/usr/bin/env python3
"""
YDLIDAR Spatial Touch Interface (Virtual Trackpad & Mouse)
Translates 2D planar laser reflections into absolute screen coordinates via Linux uinput.
"""

import ydlidar
import math
import time
import sys
from evdev import UInput, ecodes as e, AbsInfo

# === 1. CALIBRATION & INTERACTION ZONE ===
# Based on sensor characteristics, the minimum range is ~0.23m.
# Place the LiDAR at the top-center of your desk/wall pointing toward you.
MIN_DIST = 0.24      # 24 cm (just outside the 0.23m blindspot)
MAX_DIST = 0.75      # 75 cm (depth of your active touch zone)
ANGLE_LIMIT = 50.0   # +/- 50 degrees left and right of center

# Screen Resolution (Adjust if your display isn't 1920x1080)
SCREEN_W = 1920
SCREEN_H = 1080

# Smoothing factor (0.1 = very smooth/slower, 0.8 = instant/jittery)
SMOOTHING = 0.35

# Physical bounding box in meters (mapped to screen edges)
X_BOUNDS = 0.35      # Hand moves 35cm left (-0.35) to 35cm right (+0.35)
Y_MIN = 0.25         # Top of screen (25cm away from sensor)
Y_MAX = 0.65         # Bottom of screen (65cm away from sensor)

def main():
    # Initialize Linux uinput Virtual Mouse
    cap = {
        e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
        e.EV_ABS: [
            (e.ABS_X, AbsInfo(value=0, min=0, max=SCREEN_W, fuzz=0, flat=0, resolution=0)),
            (e.ABS_Y, AbsInfo(value=0, min=0, max=SCREEN_H, fuzz=0, flat=0, resolution=0))
        ]
    }
    try:
        ui = UInput(cap, name='ydlidar-spatial-touch')
    except Exception as ex:
        print(f"[-] uinput error: {ex}. Run: sudo modprobe uinput && sudo chmod 666 /dev/uinput")
        sys.exit(1)

    # Initialize YDLIDAR
    ydlidar.os_init()
    laser = ydlidar.CYdLidar()
    
    # Matching your working hardware config
    laser.setlidaropt(ydlidar.LidarPropSerialPort, "/dev/ttyUSB0")
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 7.0)
    laser.setlidaropt(ydlidar.LidarPropSampleRate, 5)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
    laser.setlidaropt(ydlidar.LidarPropFixedResolution, False) # Helps prevent >5000 point overflow
    laser.setlidaropt(ydlidar.LidarPropAutoReconnect, True)
    laser.setlidaropt(ydlidar.LidarPropSupportMotorDtrCtrl, True)

    print("[+] Initializing YDLIDAR...")
    if not laser.initialize() or not laser.turnOn():
        print("[-] Failed to start LiDAR.")
        print("    Ensure secondary 5V power is connected and /dev/ttyUSB0 is accessible.")
        sys.exit(1)

    scan = ydlidar.LaserScan()
    smooth_x, smooth_y = SCREEN_W // 2, SCREEN_H // 2

    print("\n=== SPATIAL TOUCH INTERFACE ACTIVE ===")
    print(f"Zone: {MIN_DIST*100:.0f}cm to {MAX_DIST*100:.0f}cm in front of sensor.")
    print("Press Ctrl+C to stop.\n")

    try:
        while ydlidar.os_isOk():
            # Poll immediately without sleep to prevent serial buffer accumulation
            if laser.doProcessSimple(scan):
                valid_points = []

                for p in scan.points:
                    if MIN_DIST <= p.range <= MAX_DIST:
                        deg = math.degrees(p.angle)
                        # Normalize angle to -180 to +180
                        if deg > 180:
                            deg -= 360
                        
                        if -ANGLE_LIMIT <= deg <= ANGLE_LIMIT:
                            valid_points.append((p.range, math.radians(deg)))

                if valid_points:
                    # Find closest point (the tip of your hand)
                    valid_points.sort(key=lambda item: item[0])
                    min_r = valid_points[0][0]

                    # Cluster all points within 8cm of the closest point (your hand)
                    hand_cluster = [pt for pt in valid_points if pt[0] <= min_r + 0.08]

                    # Calculate average Cartesian (X, Y) centroid of your hand
                    avg_x = sum(r * math.sin(a) for r, a in hand_cluster) / len(hand_cluster)
                    avg_y = sum(r * math.cos(a) for r, a in hand_cluster) / len(hand_cluster)

                    # Map physical meters to screen pixels
                    norm_x = (avg_x + X_BOUNDS) / (2 * X_BOUNDS)
                    norm_y = (avg_y - Y_MIN) / (Y_MAX - Y_MIN)

                    target_x = int(max(0, min(1, norm_x)) * SCREEN_W)
                    target_y = int(max(0, min(1, norm_y)) * SCREEN_H)

                    # Apply Exponential Moving Average (EMA) smoothing
                    smooth_x = int(SMOOTHING * target_x + (1 - SMOOTHING) * smooth_x)
                    smooth_y = int(SMOOTHING * target_y + (1 - SMOOTHING) * smooth_y)

                    # Move OS cursor
                    ui.write(e.EV_ABS, e.ABS_X, smooth_x)
                    ui.write(e.EV_ABS, e.ABS_Y, smooth_y)
                    ui.syn()

                    sys.stdout.write(f"\r[TRACKING] Hand at Dist: {min_r:.2f}m | Cursor: ({smooth_x:4d}, {smooth_y:4d}) | Cluster pts: {len(hand_cluster):3d}   ")
                    sys.stdout.flush()
            else:
                # Tiny 1ms yield only if scan isn't ready
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[+] Shutting down...")
    finally:
        laser.turnOff()
        laser.disconnecting()
        ui.close()
        print("[+] Clean exit complete.")

if __name__ == "__main__":
    main()
