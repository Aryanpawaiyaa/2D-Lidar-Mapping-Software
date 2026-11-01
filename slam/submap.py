"""Submap management: bounded local occupancy grids and keyframe management."""

import math
from typing import List, Optional
import numpy as np

from slam.occupancy_grid import OccupancyGrid


class Submap:
    """A bounded local occupancy grid representing a spatial region."""

    def __init__(
        self,
        submap_id: int,
        session_id: str,
        origin_pose: np.ndarray,
        resolution_m: float = 0.05,
        extent_m: float = 14.0,
    ):
        self.submap_id = submap_id
        self.session_id = session_id
        # Origin pose in world/session frame: [x, y, theta]
        self.origin_pose = origin_pose.copy().astype(np.float64)
        self.resolution = resolution_m
        self.extent = extent_m

        # Local grid centered around origin_pose
        self.grid = OccupancyGrid(
            resolution_m=resolution_m,
            width_m=extent_m,
            height_m=extent_m,
            origin_x_m=self.origin_pose[0] - extent_m / 2.0,
            origin_y_m=self.origin_pose[1] - extent_m / 2.0,
        )

        self.keyframe_ids: List[int] = []
        self.is_finished: bool = False
        self.num_points_mapped: int = 0

    def add_keyframe(
        self,
        keyframe_id: int,
        sensor_pose: np.ndarray,
        points_sensor: np.ndarray,
        max_ray_range: float = 25.0,
    ) -> None:
        """Integrate a keyframe's scan points into this submap's occupancy grid."""
        self.keyframe_ids.append(keyframe_id)

        # Transform sensor points to world coordinates
        x, y, theta = sensor_pose[0], sensor_pose[1], sensor_pose[2]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        px_w = cos_t * points_sensor[:, 0] - sin_t * points_sensor[:, 1] + x
        py_w = sin_t * points_sensor[:, 0] + cos_t * points_sensor[:, 1] + y
        pts_world = np.column_stack((px_w, py_w)).astype(np.float32)

        self.grid.update_scan(
            sensor_x=float(x),
            sensor_y=float(y),
            hit_points_world=pts_world,
            max_ray_range=max_ray_range,
        )
        self.num_points_mapped += len(points_sensor)

    def finish(self) -> None:
        """Freeze submap and precompute distance field."""
        self.is_finished = True
        self.grid.compute_distance_field()

