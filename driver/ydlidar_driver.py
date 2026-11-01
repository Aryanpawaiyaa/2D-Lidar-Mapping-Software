"""Official YDLidar-SDK Python wrapper (Tier 1A).

Uses official named properties and enums:
- LidarPropLidarType: TYPE_TOF (0)
- LidarPropDeviceType: YDLIDAR_TYPE_SERIAL (0)
- LidarPropSerialBaudrate: 512000
- LidarPropIntensities: True
- LidarPropSingleChannel: False
"""

import logging
import math
import time
from typing import Optional
import numpy as np

from driver.interface import LidarSource, Scan

logger = logging.getLogger(__name__)

try:
    import ydlidar
    YDLIDAR_AVAILABLE = True
except ImportError:
    ydlidar = None
    YDLIDAR_AVAILABLE = False


class YDLidarSDKDriver(LidarSource):
    """Lidar source backed by official YDLidar-SDK python bindings."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 512000,
        sample_rate_khz: int = 20,
        scan_frequency_hz: float = 7.0,
        range_min_m: float = 0.05,
        range_max_m: float = 50.0,
        intensities: bool = True,
    ):
        if not YDLIDAR_AVAILABLE:
            raise ImportError(
                "Official 'ydlidar' Python bindings are not installed. "
                "Use TG50SerialDriver or build YDLidar-SDK."
            )

        self.port = port
        self.baudrate = baudrate
        self.sample_rate_khz = sample_rate_khz
        self.scan_frequency_hz = scan_frequency_hz
        self.range_min_m = range_min_m
        self.range_max_m = range_max_m
        self.intensities = intensities

        self._laser: Optional[ydlidar.CYdLidar] = None
        self._is_started = False

    def start(self) -> None:
        ydlidar.os_init()
        self._laser = ydlidar.CYdLidar()

        # Serial configuration
        self._laser.setlidaropt(ydlidar.LidarPropSerialPort, self.port)
        self._laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, self.baudrate)

        # Lidar model properties (TG series ToF configuration)
        # Always use named constants from SDK
        self._laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TOF)
        self._laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        self._laser.setlidaropt(ydlidar.LidarPropSampleRate, self.sample_rate_khz)
        self._laser.setlidaropt(ydlidar.LidarPropScanFrequency, float(self.scan_frequency_hz))

        # Scan limits
        self._laser.setlidaropt(ydlidar.LidarPropMinRange, float(self.range_min_m))
        self._laser.setlidaropt(ydlidar.LidarPropMaxRange, float(self.range_max_m))
        self._laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
        self._laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)

        # Intensity and channel mode
        self._laser.setlidaropt(ydlidar.LidarPropIntensities, self.intensities)
        self._laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)
        self._laser.setlidaropt(ydlidar.LidarPropAbnormalCheckCount, 4)

        if not self._laser.initialize():
            err = self._laser.DescribeError() if hasattr(self._laser, "DescribeError") else "Unknown error"
            raise RuntimeError(f"YDLidar initialization failed: {err}")

        if not self._laser.turnOn():
            raise RuntimeError("YDLidar failed to start motor/laser (turnOn failed).")

        self._is_started = True
        logger.info(f"YDLidarSDKDriver started on {self.port} at {self.baudrate} baud.")

    def get_scan(self, timeout: float = 1.0) -> Optional[Scan]:
        if not self._is_started or self._laser is None:
            return None

        laser_scan = ydlidar.LaserScan()
        if self._laser.doProcessSimple(laser_scan):
            pts = laser_scan.points
            n = len(pts)
            if n == 0:
                return None

            angles = np.zeros(n, dtype=np.float32)
            ranges = np.zeros(n, dtype=np.float32)
            intensities = np.zeros(n, dtype=np.float32) if self.intensities else None

            for i in range(n):
                p = pts[i]
                # SDK angle: verify radians vs degrees (SDK LaserPoint.angle is typically radians in modern SDK)
                ang = p.angle
                if abs(ang) > 2 * math.pi:  # degrees fallback
                    ang = math.radians(ang)
                # Normalize to [-pi, pi)
                angles[i] = (ang + np.pi) % (2 * np.pi) - np.pi
                ranges[i] = p.range
                if intensities is not None:
                    intensities[i] = p.intensity

            # Sort by angle
            sort_idx = np.argsort(angles)
            return Scan(
                angles_rad=angles[sort_idx],
                ranges_m=ranges[sort_idx],
                intensities=intensities[sort_idx] if intensities is not None else None,
                timestamp=time.time(),
            )
        return None

    def stop(self) -> None:
        if self._laser is not None:
            try:
                self._laser.turnOff()
                self._laser.disconnecting()
            except Exception:
                pass
            self._laser = None
        self._is_started = False
        logger.info("YDLidarSDKDriver stopped cleanly.")

    def is_connected(self) -> bool:
        return self._is_started and self._laser is not None

    def describe(self) -> str:
        return f"YDLidarSDKDriver(port='{self.port}', baud={self.baudrate})"

