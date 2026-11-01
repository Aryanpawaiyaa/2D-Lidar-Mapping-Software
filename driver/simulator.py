"""High-performance 2D raycasting simulation for algorithm development without hardware."""

import time
from typing import List, Optional, Tuple
import numpy as np

from driver.interface import LidarSource, Scan


class LineSegment:
    """2D line segment representing a wall or boundary."""
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.p1 = np.array([x1, y1], dtype=np.float32)
        self.p2 = np.array([x2, y2], dtype=np.float32)


class SimulatedEnvironment:
    """2D multi-room synthetic environment with Room A, doorway, and Room B."""

    def __init__(self):
        self.walls: List[LineSegment] = []
        self._build_default_multi_room()

    def _build_default_multi_room(self) -> None:
        """Construct Room A [0 to 6, 0 to 5], Doorway [6 to 8, 2 to 3.2], and Room B [8 to 14, 0 to 6]."""
        # Room A walls
        # South wall: (0,0) to (6,0)
        self.add_wall(0, 0, 6, 0)
        # West wall: (0,0) to (0,5)
        self.add_wall(0, 0, 0, 5)
        # North wall: (0,5) to (6,5)
        self.add_wall(0, 5, 6, 5)
        # East wall with doorway: (6,0) to (6,2.0) and (6,3.2) to (6,5)
        self.add_wall(6, 0, 6, 2.0)
        self.add_wall(6, 3.2, 6, 5.0)

        # Corridor walls connecting Room A and Room B
        # South corridor wall: (6,2.0) to (8,2.0)
        self.add_wall(6, 2.0, 8, 2.0)
        # North corridor wall: (6,3.2) to (8,3.2)
        self.add_wall(6, 3.2, 8, 3.2)

        # Room B walls
        # West wall with doorway: (8,0) to (8,2.0) and (8,3.2) to (8,6.0)
        self.add_wall(8, 0, 8, 2.0)
        self.add_wall(8, 3.2, 8, 6.0)
        # South wall: (8,0) to (14,0)
        self.add_wall(8, 0, 14, 0)
        # East wall: (14,0) to (14,6.0)
        self.add_wall(14, 0, 14, 6.0)
        # North wall: (8,6.0) to (14,6.0)
        self.add_wall(8, 6.0, 14, 6.0)

        # Some internal furniture / obstacles in Room A
        # Pillar 1 in Room A: (2, 2) to (2.5, 2.5) box
        self.add_box(2.0, 2.0, 0.5, 0.5)

        # Internal obstacles in Room B
        # Table in Room B: (10.5, 3.0) to (12.0, 4.0) box
        self.add_box(10.5, 3.0, 1.5, 1.0)

    def add_wall(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.walls.append(LineSegment(x1, y1, x2, y2))

    def add_box(self, x: float, y: float, w: float, h: float) -> None:
        self.add_wall(x, y, x + w, y)
        self.add_wall(x + w, y, x + w, y + h)
        self.add_wall(x + w, y + h, x, y + h)
        self.add_wall(x, y + h, x, y)


class SimulationLidar(LidarSource):
    """Raycasting Lidar simulator generating realistic 360-degree scans."""

    def __init__(
        self,
        env: Optional[SimulatedEnvironment] = None,
        num_beams: int = 720,
        scan_rate_hz: float = 7.0,
        range_min_m: float = 0.05,
        range_max_m: float = 50.0,
        noise_std_m: float = 0.015,
        dropout_rate: float = 0.005,
    ):
        self.env = env or SimulatedEnvironment()
        self.num_beams = num_beams
        self.scan_rate_hz = scan_rate_hz
        self.range_min_m = range_min_m
        self.range_max_m = range_max_m
        self.noise_std_m = noise_std_m
        self.dropout_rate = dropout_rate

        # Simulated robot state (x, y, yaw in radians)
        self.pose = np.array([2.5, 2.5, 0.0], dtype=np.float32)
        self._is_started = False
        self._last_scan_time = 0.0
        self._step_dt = 1.0 / self.scan_rate_hz

        # Precompute base angles from -pi to +pi (CCW)
        self.base_angles = np.linspace(-np.pi, np.pi, self.num_beams, endpoint=False, dtype=np.float32)

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        """Update the simulated robot pose."""
        self.pose = np.array([x, y, yaw], dtype=np.float32)

    def move_by(self, dx: float, dy: float, dyaw: float) -> None:
        """Increment current pose in world coordinates."""
        self.pose[0] += dx
        self.pose[1] += dy
        self.pose[2] = (self.pose[2] + dyaw + np.pi) % (2 * np.pi) - np.pi

    def start(self) -> None:
        self._is_started = True
        self._last_scan_time = time.time()

    def get_scan(self, timeout: float = 1.0) -> Optional[Scan]:
        if not self._is_started:
            return None

        # Raycast each beam against environment walls
        # Ray origin
        ox, oy = self.pose[0], self.pose[1]
        sensor_yaw = self.pose[2]

        world_angles = self.base_angles + sensor_yaw
        cos_a = np.cos(world_angles)
        sin_a = np.sin(world_angles)

        ranges = np.full(self.num_beams, self.range_max_m, dtype=np.float32)
        intensities = np.full(self.num_beams, 200.0, dtype=np.float32)

        # Vectorized segment intersection:
        # Ray: P = O + t * D  (t >= 0)
        # Wall segment: Q = A + u * (B - A)  (0 <= u <= 1)
        # Solve: O + t * D = A + u * (B - A)
        # (B - A)_perp dot (O + t*D - A) = 0 => t = ((A - O) x (B - A)) / (D x (B - A))
        # u = ((A - O) x D) / (D x (B - A))
        for wall in self.env.walls:
            ax, ay = wall.p1[0], wall.p1[1]
            bx, by = wall.p2[0], wall.p2[1]
            vx, vy = bx - ax, by - ay  # wall vector

            # D x V = cos_a * vy - sin_a * vx
            denom = cos_a * vy - sin_a * vx
            # Non-parallel condition
            valid = np.abs(denom) > 1e-6

            dx_ao = ax - ox
            dy_ao = ay - oy

            # numerator for t: (A - O) x V = dx_ao * vy - dy_ao * vx
            t_num = dx_ao * vy - dy_ao * vx
            # numerator for u: (A - O) x D = dx_ao * sin_a - dy_ao * cos_a
            u_num = dx_ao * sin_a - dy_ao * cos_a

            t = np.full(self.num_beams, -1.0, dtype=np.float32)
            u = np.full(self.num_beams, -1.0, dtype=np.float32)
            np.divide(t_num, denom, out=t, where=valid)
            np.divide(u_num, denom, out=u, where=valid)

            hit_mask = valid & (t >= self.range_min_m) & (t <= self.range_max_m) & (u >= 0.0) & (u <= 1.0)
            ranges = np.where(hit_mask & (t < ranges), t, ranges)

        # Add Gaussian noise
        if self.noise_std_m > 0:
            noise = np.random.normal(0, self.noise_std_m, size=ranges.shape).astype(np.float32)
            # Only add noise to valid hits, not max-range misses
            has_hit = ranges < (self.range_max_m - 0.1)
            ranges[has_hit] = np.clip(ranges[has_hit] + noise[has_hit], self.range_min_m, self.range_max_m)

        # Simulate occasional beam dropouts
        if self.dropout_rate > 0:
            dropouts = np.random.rand(self.num_beams) < self.dropout_rate
            ranges[dropouts] = 0.0  # Invalid zero return

        now = time.time()
        self._last_scan_time = now

        return Scan(
            angles_rad=self.base_angles.copy(),
            ranges_m=ranges,
            intensities=intensities,
            timestamp=now,
        )

    def stop(self) -> None:
        self._is_started = False

    def is_connected(self) -> bool:
        return self._is_started

    def describe(self) -> str:
        return f"SimulationLidar(beams={self.num_beams}, rate={self.scan_rate_hz}Hz, pose=({self.pose[0]:.2f}, {self.pose[1]:.2f}))"
