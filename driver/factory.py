"""Driver factory for creating appropriate LidarSource instance."""

import logging
from typing import Any, Dict, Optional

from driver.fake_lidar import FakeLidar
from driver.interface import LidarSource
from driver.simulator import SimulationLidar, SimulatedEnvironment
from driver.tg50_serial import TG50SerialDriver
from driver.ydlidar_driver import YDLIDAR_AVAILABLE, YDLidarSDKDriver

logger = logging.getLogger(__name__)


def create_lidar_source(config: Dict[str, Any], replay_file: Optional[str] = None) -> LidarSource:
    """Create and return a configured LidarSource."""
    lidar_cfg = config.get("lidar", {})
    mode = lidar_cfg.get("driver_mode", "auto").lower()

    if replay_file:
        logger.info(f"Using FakeLidar with replay file: {replay_file}")
        return FakeLidar(log_path=replay_file, realtime=True)

    if mode == "sim":
        logger.info("Instantiating SimulationLidar...")
        return SimulationLidar(
            scan_rate_hz=float(lidar_cfg.get("scan_frequency_hz", 7.0)),
            range_min_m=float(lidar_cfg.get("range_min_m", 0.05)),
            range_max_m=float(lidar_cfg.get("range_max_m", 50.0)),
        )

    if mode == "fake":
        log_path = lidar_cfg.get("replay_file")
        if not log_path:
            raise ValueError("driver_mode 'fake' requires 'replay_file' in config.yaml")
        return FakeLidar(log_path=log_path)

    if mode == "sdk":
        if not YDLIDAR_AVAILABLE:
            raise RuntimeError("YDLidar Python SDK is not installed, but mode 'sdk' was requested.")
        return YDLidarSDKDriver(
            port=lidar_cfg.get("port", "/dev/ttyUSB0"),
            baudrate=int(lidar_cfg.get("baudrate", 512000)),
            sample_rate_khz=int(lidar_cfg.get("sample_rate_khz", 20)),
            scan_frequency_hz=float(lidar_cfg.get("scan_frequency_hz", 7.0)),
            range_min_m=float(lidar_cfg.get("range_min_m", 0.05)),
            range_max_m=float(lidar_cfg.get("range_max_m", 50.0)),
            intensities=bool(lidar_cfg.get("enable_intensity", True)),
        )

    if mode == "serial":
        return TG50SerialDriver(
            port=lidar_cfg.get("port", "/dev/ttyUSB0"),
            baudrate=int(lidar_cfg.get("baudrate", 512000)),
            auto_detect=bool(lidar_cfg.get("auto_detect_ports", True)),
            range_min_m=float(lidar_cfg.get("range_min_m", 0.05)),
            range_max_m=float(lidar_cfg.get("range_max_m", 50.0)),
        )

    # mode == 'auto'
    if YDLIDAR_AVAILABLE:
        logger.info("Auto-driver: Found official ydlidar Python bindings. Using YDLidarSDKDriver.")
        try:
            return YDLidarSDKDriver(
                port=lidar_cfg.get("port", "/dev/ttyUSB0"),
                baudrate=int(lidar_cfg.get("baudrate", 512000)),
                sample_rate_khz=int(lidar_cfg.get("sample_rate_khz", 20)),
                scan_frequency_hz=float(lidar_cfg.get("scan_frequency_hz", 7.0)),
                range_min_m=float(lidar_cfg.get("range_min_m", 0.05)),
                range_max_m=float(lidar_cfg.get("range_max_m", 50.0)),
                intensities=bool(lidar_cfg.get("enable_intensity", True)),
            )
        except Exception as e:
            logger.warning(f"Could not initialize YDLidarSDKDriver ({e}). Falling back to pure-Python TG50SerialDriver.")

    logger.info("Auto-driver: Using pure-Python TG50SerialDriver.")
    return TG50SerialDriver(
        port=lidar_cfg.get("port", "/dev/ttyUSB0"),
        baudrate=int(lidar_cfg.get("baudrate", 512000)),
        auto_detect=bool(lidar_cfg.get("auto_detect_ports", True)),
        range_min_m=float(lidar_cfg.get("range_min_m", 0.05)),
        range_max_m=float(lidar_cfg.get("range_max_m", 50.0)),
    )

