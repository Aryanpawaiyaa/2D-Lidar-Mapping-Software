import os
import sys
import time
import ydlidar

def main():
    print("=== YDLIDAR Hardware & SDK Test ===")
    ydlidar.os_init()

    ports = ydlidar.lidarPortList()
    port = "/dev/ydlidar"
    print(f"Detected ports:")
    for k, v in ports.items():
        print(f"  - {k}: {v}")

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

    ret = laser.initialize()
    if not ret:
        print("[-] Failed to initialize LiDAR.")
        return

    print("[+] LiDAR initialized successfully. Turning on motor and laser...")
    ret = laser.turnOn()
    if not ret:
        print("[-] Failed to turn on LiDAR. Ensure motor power / secondary USB port is connected.")
        laser.disconnecting()
        return

    print("[+] Streaming scans (Press Ctrl+C to stop)...")
    scan = ydlidar.LaserScan()
    try:
        while ydlidar.os_isOk():
            if laser.doProcessSimple(scan):
                hz = 1.0 / scan.config.scan_time if scan.config.scan_time > 0 else 0
                print(f"Scan received [{scan.stamp}]: {scan.points.size()} points, scanFreq: {hz:.1f} Hz")
            else:
                print("Waiting for scan...")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping LiDAR...")
    finally:
        laser.turnOff()
        laser.disconnecting()
        print("LiDAR disconnected.")

if __name__ == "__main__":
    main()
