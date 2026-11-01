"""Scan preprocessor: range filtering, downsampling, and degeneracy detection."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from driver.interface import Scan


@dataclass
class PreprocessedScan:
    """Filtered and conditioned scan ready for scan matching and mapping."""
    points_sensor: np.ndarray       # shape (N, 2), local Cartesian (x, y)
    angles_rad: np.ndarray          # shape (N,)
    ranges_m: np.ndarray            # shape (N,)
    intensities: Optional[np.ndarray] # shape (N,) or None
    timestamp: float
    is_degenerate: bool = False
    degeneracy_direction: Optional[np.ndarray] = None  # unit vector along unconstrained direction


class ScanPreprocessor:
    """Preprocesses raw Lidar scans."""

    def __init__(
        self,
        range_min_m: float = 0.05,
        range_max_m: float = 25.0,
        angular_downsample_deg: float = 0.25,
        outlier_min_neighbors: int = 2,
        outlier_radius_m: float = 0.15,
        min_eigenvalue_ratio: float = 0.03,
    ):
        self.range_min_m = range_min_m
        self.range_max_m = range_max_m
        self.angular_downsample_rad = np.radians(angular_downsample_deg)
        self.outlier_min_neighbors = outlier_min_neighbors
        self.outlier_radius_m = outlier_radius_m
        self.min_eigenvalue_ratio = min_eigenvalue_ratio

    def process(self, raw_scan: Scan) -> Optional[PreprocessedScan]:
        """Filter and process raw scan into clean points and detect degeneracy."""
        # 1. Range validity mask
        valid = (
            np.isfinite(raw_scan.ranges_m)
            & (raw_scan.ranges_m >= self.range_min_m)
            & (raw_scan.ranges_m <= self.range_max_m)
        )
        if np.sum(valid) < 15:
            return None

        angles = raw_scan.angles_rad[valid]
        ranges = raw_scan.ranges_m[valid]
        intensities = raw_scan.intensities[valid] if raw_scan.intensities is not None else None

        # 2. Angular downsampling (if requested)
        if self.angular_downsample_rad > 0:
            # Bin angles into discrete buckets
            bin_indices = np.round(angles / self.angular_downsample_rad).astype(np.int32)
            _, unique_idx = np.unique(bin_indices, return_index=True)
            angles = angles[unique_idx]
            ranges = ranges[unique_idx]
            if intensities is not None:
                intensities = intensities[unique_idx]

        # 3. Convert to Cartesian sensor coordinates (x, y)
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.column_stack((xs, ys)).astype(np.float32)

        # 4. Outlier removal (speckle filter for dust/multipath returns)
        if self.outlier_min_neighbors > 0 and len(points) > 10:
            keep_mask = self._filter_outliers(points)
            points = points[keep_mask]
            angles = angles[keep_mask]
            ranges = ranges[keep_mask]
            if intensities is not None:
                intensities = intensities[keep_mask]

        if len(points) < 15:
            return None

        # 5. Degeneracy check (corridor / collinear points detection via PCA)
        is_degenerate, degen_dir = self._check_geometric_degeneracy(points)

        return PreprocessedScan(
            points_sensor=points,
            angles_rad=angles,
            ranges_m=ranges,
            intensities=intensities,
            timestamp=raw_scan.timestamp,
            is_degenerate=is_degenerate,
            degeneracy_direction=degen_dir,
        )

    def _filter_outliers(self, points: np.ndarray) -> np.ndarray:
        """Simple fast spatial outlier filter based on neighbor counts."""
        # For small point clouds (hundreds of points), compute pairwise distance or chunked neighbor test
        # Fast consecutive distance filter:
        # A true wall return has adjacent angular points close in Euclidean distance
        dists_next = np.hypot(points[1:, 0] - points[:-1, 0], points[1:, 1] - points[:-1, 1])
        dists_prev = np.empty_like(points[:, 0])
        dists_prev[0] = 999.0
        dists_prev[1:] = dists_next
        dists_next_full = np.empty_like(points[:, 0])
        dists_next_full[:-1] = dists_next
        dists_next_full[-1] = 999.0

        has_near_neighbor = (dists_prev < self.outlier_radius_m) | (dists_next_full < self.outlier_radius_m)
        return has_near_neighbor

    def _check_geometric_degeneracy(self, points: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Check if scan points are degenerate (e.g. collinear along a single wall)."""
        centered = points - np.mean(points, axis=0)
        cov = (centered.T @ centered) / len(points)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues ascending
        idx = np.argsort(eigvals)
        lambda_min, lambda_max = eigvals[idx[0]], eigvals[idx[1]]

        ratio = lambda_min / max(1e-6, lambda_max)
        if ratio < self.min_eigenvalue_ratio:
            # Low information along the principal line
            primary_direction = eigvecs[:, idx[1]]
            return True, primary_direction

        return False, None

