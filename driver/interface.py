"""Abstract Lidar interface and standard Scan data representation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Scan:
    """Standard 2D Lidar Scan representation.
    
    Coordinates follow ROS REP-103 convention:
      - Coordinate system: Right-handed (X forward, Y left, Z up).
      - Angles: Radians, CCW positive. 0 rad is along the +X axis (forward),
        +pi/2 rad is along +Y (left), -pi/2 rad is along -Y (right).
      - Ranges: Meters (float32).
      - Intensities: Scaled signal strength (0-255) or None.
      - Timestamp: Epoch seconds (float).
    """
    angles_rad: np.ndarray        # shape (N,), float32
    ranges_m: np.ndarray          # shape (N,), float32
    intensities: Optional[np.ndarray] = None  # shape (N,), float32 or None
    timestamp: float = 0.0

    def __post_init__(self):
        self.angles_rad = np.ascontiguousarray(self.angles_rad, dtype=np.float32)
        self.ranges_m = np.ascontiguousarray(self.ranges_m, dtype=np.float32)
        if self.intensities is not None:
            self.intensities = np.ascontiguousarray(self.intensities, dtype=np.float32)

    def to_cartesian(self, min_range: float = 0.05, max_range: float = 50.0) -> np.ndarray:
        """Convert valid range returns to 2D local Cartesian coordinates (x, y).
        
        Returns:
            np.ndarray of shape (M, 2), float32 points in sensor frame.
        """
        valid_mask = (
            np.isfinite(self.ranges_m) &
            (self.ranges_m >= min_range) &
            (self.ranges_m <= max_range)
        )
        ranges = self.ranges_m[valid_mask]
        angles = self.angles_rad[valid_mask]
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        return np.column_stack((xs, ys)).astype(np.float32)


class LidarSource(ABC):
    """Abstract interface for any 2D lidar data source."""

    @abstractmethod
    def start(self) -> None:
        """Start the lidar connection and spinning motor."""
        pass

    @abstractmethod
    def get_scan(self, timeout: float = 1.0) -> Optional[Scan]:
        """Fetch the next available full 360-degree scan.
        
        Args:
            timeout: Maximum seconds to wait for a scan.
            
        Returns:
            Scan object if successful, or None on timeout/disconnection.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop scanning and shut down hardware cleanly."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if connection is currently active."""
        pass

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of this source."""
        pass

