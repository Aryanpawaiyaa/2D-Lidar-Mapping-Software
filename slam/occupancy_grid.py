"""Occupancy Grid Map with log-odds Bayesian updates and Euclidean Distance Field."""

import math
from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt


class OccupancyGrid:
    """2D Log-Odds Occupancy Grid Map."""

    def __init__(
        self,
        resolution_m: float = 0.05,
        width_m: float = 20.0,
        height_m: float = 20.0,
        origin_x_m: float = -10.0,
        origin_y_m: float = -10.0,
        p_occ: float = 0.75,
        p_free: float = 0.35,
        log_odds_min: float = -4.0,
        log_odds_max: float = 4.0,
        occupied_thresh: float = 0.65,
        free_thresh: float = 0.25,
    ):
        self.resolution = float(resolution_m)
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.origin_x = float(origin_x_m)
        self.origin_y = float(origin_y_m)

        self.cols = int(round(self.width_m / self.resolution))
        self.rows = int(round(self.height_m / self.resolution))

        self.log_odds_occ = float(np.log(p_occ / (1.0 - p_occ)))
        self.log_odds_free = float(np.log(p_free / (1.0 - p_free)))
        self.log_odds_min = float(log_odds_min)
        self.log_odds_max = float(log_odds_max)

        self.occupied_thresh = float(occupied_thresh)
        self.free_thresh = float(free_thresh)
        self.log_odds_thresh_occ = float(np.log(occupied_thresh / (1.0 - occupied_thresh)))

        # 2D Grid array storing log-odds: shape (rows, cols), float32
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Cached distance field and gradients for scan matching
        self._edt: Optional[np.ndarray] = None
        self._grad_x: Optional[np.ndarray] = None
        self._grad_y: Optional[np.ndarray] = None
        self._cache_dirty = True

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid indices (col, row)."""
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid indices (col, row) to world coordinates (cell center in meters)."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def is_in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def update_scan(
        self,
        sensor_x: float,
        sensor_y: float,
        hit_points_world: np.ndarray,
        max_ray_range: float = 25.0,
    ) -> None:
        """Update occupancy grid using raycasting from sensor to each hit point."""
        c0, r0 = self.world_to_grid(sensor_x, sensor_y)
        if not self.is_in_bounds(c0, r0):
            # If sensor is outside current bounds, expand or skip
            return

        c_endpoints = np.floor((hit_points_world[:, 0] - self.origin_x) / self.resolution).astype(np.int32)
        r_endpoints = np.floor((hit_points_world[:, 1] - self.origin_y) / self.resolution).astype(np.int32)

        # Rasterize rays using Bresenham lines
        for c1, r1 in zip(c_endpoints, r_endpoints):
            # Compute ray length
            dx = (c1 - c0) * self.resolution
            dy = (r1 - r0) * self.resolution
            dist = math.hypot(dx, dy)
            if dist > max_ray_range:
                # Clip to max range
                scale = max_ray_range / dist
                c1 = int(round(c0 + (c1 - c0) * scale))
                r1 = int(round(r0 + (r1 - r0) * scale))
                is_endpoint_hit = False
            else:
                is_endpoint_hit = True

            line_cols, line_rows = self._bresenham_line(c0, r0, c1, r1)
            if len(line_cols) == 0:
                continue

            # Free space along the ray
            free_cols = line_cols[:-1] if is_endpoint_hit else line_cols
            free_rows = line_rows[:-1] if is_endpoint_hit else line_rows

            valid_free = (
                (free_cols >= 0) & (free_cols < self.cols) &
                (free_rows >= 0) & (free_rows < self.rows)
            )
            fc = free_cols[valid_free]
            fr = free_rows[valid_free]
            self.grid[fr, fc] = np.maximum(self.grid[fr, fc] + self.log_odds_free, self.log_odds_min)

            # Hit at endpoint
            if is_endpoint_hit and self.is_in_bounds(c1, r1):
                self.grid[r1, c1] = np.minimum(self.grid[r1, c1] + self.log_odds_occ, self.log_odds_max)

        self._cache_dirty = True

    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fast Bresenham line generation returning arrays of x and y coordinates."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        n = max(dx, dy) + 1
        x_pts = np.empty(n, dtype=np.int32)
        y_pts = np.empty(n, dtype=np.int32)

        cur_x, cur_y = x0, y0
        for i in range(n):
            x_pts[i] = cur_x
            y_pts[i] = cur_y
            if cur_x == x1 and cur_y == y1:
                return x_pts[: i + 1], y_pts[: i + 1]
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cur_x += sx
            if e2 < dx:
                err += dx
                cur_y += sy

        return x_pts, y_pts

    def compute_distance_field(self) -> None:
        """Compute Euclidean Distance Transform (EDT) and gradients from obstacle cells."""
        if not self._cache_dirty and self._edt is not None:
            return

        # Obstacle mask: cells with log-odds >= log_odds_thresh_occ
        occupied_mask = self.grid >= self.log_odds_thresh_occ

        if not np.any(occupied_mask):
            # No obstacles yet; default distance field
            self._edt = np.full_like(self.grid, fill_value=10.0)
            self._grad_x = np.zeros_like(self.grid)
            self._grad_y = np.zeros_like(self.grid)
        else:
            # distance_transform_edt computes distance to nearest zero, so invert mask:
            # non-obstacles are True, obstacles are False
            # sampling specifies distance per cell in meters
            self._edt = distance_transform_edt(~occupied_mask, sampling=self.resolution).astype(np.float32)

            # Compute numerical gradients with central differences: d(dist)/dx, d(dist)/dy
            # Note: rows index Y, cols index X
            gy, gx = np.gradient(self._edt, self.resolution)
            self._grad_x = gx.astype(np.float32)
            self._grad_y = gy.astype(np.float32)

        self._cache_dirty = False

    def get_probability_map(self) -> np.ndarray:
        """Convert log-odds grid to probability in [0, 1]."""
        # p = 1 / (1 + exp(-l))
        return 1.0 / (1.0 + np.exp(-self.grid))

    def get_distance_field(self) -> np.ndarray:
        self.compute_distance_field()
        assert self._edt is not None
        return self._edt

    def query_distance_and_gradient(
        self, points_world: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Query distance and gradients for world points via bilinear interpolation.
        
        Args:
            points_world: shape (N, 2), points in world coordinates (meters).
            
        Returns:
            dists: shape (N,) distance to nearest obstacle in meters.
            grad_x: shape (N,) d(dist)/dx.
            grad_y: shape (N,) d(dist)/dy.
            valid_mask: shape (N,) bool mask indicating points within grid bounds.
        """
        self.compute_distance_field()
        assert self._edt is not None and self._grad_x is not None and self._grad_y is not None

        # Continuous grid coordinates
        col_f = (points_world[:, 0] - self.origin_x) / self.resolution - 0.5
        row_f = (points_world[:, 1] - self.origin_y) / self.resolution - 0.5

        valid = (
            (col_f >= 0.0) & (col_f < (self.cols - 1)) &
            (row_f >= 0.0) & (row_f < (self.rows - 1))
        )

        n = len(points_world)
        dists = np.full(n, 10.0, dtype=np.float32)
        gx = np.zeros(n, dtype=np.float32)
        gy = np.zeros(n, dtype=np.float32)

        if not np.any(valid):
            return dists, gx, gy, valid

        c0 = np.floor(col_f[valid]).astype(np.int32)
        r0 = np.floor(row_f[valid]).astype(np.int32)
        c1 = c0 + 1
        r1 = r0 + 1

        wc = col_f[valid] - c0
        wr = row_f[valid] - r0

        # Bilinear weights
        w00 = (1.0 - wc) * (1.0 - wr)
        w10 = wc * (1.0 - wr)
        w01 = (1.0 - wc) * wr
        w11 = wc * wr

        # Interpolate distance
        dists[valid] = (
            w00 * self._edt[r0, c0] +
            w10 * self._edt[r0, c1] +
            w01 * self._edt[r1, c0] +
            w11 * self._edt[r1, c1]
        )

        # Interpolate gradients
        gx[valid] = (
            w00 * self._grad_x[r0, c0] +
            w10 * self._grad_x[r0, c1] +
            w01 * self._grad_x[r0, c0] +
            w11 * self._grad_x[r1, c1]
        )
        gy[valid] = (
            w00 * self._grad_y[r0, c0] +
            w10 * self._grad_y[r0, c1] +
            w01 * self._grad_y[r0, c0] +
            w11 * self._grad_y[r1, c1]
        )

        return dists, gx, gy, valid

