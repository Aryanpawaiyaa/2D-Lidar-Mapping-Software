"""Global Multi-Resolution Correlative Scan Matcher for relocalization across saved maps."""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import numpy as np

from slam.occupancy_grid import OccupancyGrid
from slam.scan_matcher import LikelihoodFieldScanMatcher, MatchResult
from slam.submap import Submap


@dataclass
class RelocalizationResult:
    """Outcome of global relocalization search."""
    success: bool
    pose: np.ndarray             # shape (3,), [x, y, theta] in target map frame
    confidence_score: float      # [0.0, 1.0]
    matched_submap_id: int
    mean_residual_m: float


class CorrelativeGlobalMatcher:
    """Coarse-to-fine correlative scan matcher searching over full map extent."""

    def __init__(
        self,
        coarse_res_m: float = 0.20,
        coarse_angle_step_deg: float = 3.0,
        min_confidence_score: float = 0.70,
    ):
        self.coarse_res = coarse_res_m
        self.coarse_angle_step = np.radians(coarse_angle_step_deg)
        self.min_confidence = min_confidence_score
        self.fine_matcher = LikelihoodFieldScanMatcher(max_iterations=25, huber_delta_m=0.10)

    def search_submap(
        self, submap: Submap, points_sensor: np.ndarray, top_k: int = 5
    ) -> List[Tuple[float, np.ndarray]]:
        """Fast FFT-accelerated correlative scan search across a submap's probability field.
        
        Returns:
            List of (score, candidate_pose) sorted by best (highest) score.
        """
        import scipy.signal
        grid = submap.grid
        grid.compute_distance_field()

        dist_field = grid.get_distance_field()
        # Likelihood field: peak (1.0) at obstacle cells
        likelihood = np.exp(-dist_field / (2.0 * grid.resolution)).astype(np.float32)

        res = grid.resolution
        # Filter points within submap extent radius so kernel fits within submap grid bounds
        max_rad = submap.extent * 0.45
        rad_mask = np.hypot(points_sensor[:, 0], points_sensor[:, 1]) <= max_rad
        filtered_points = points_sensor[rad_mask]
        sub_pts = filtered_points[::3] if len(filtered_points) > 120 else filtered_points

        if len(sub_pts) < 15:
            return []

        candidates: List[Tuple[float, np.ndarray]] = []
        thetas = np.linspace(-np.pi, np.pi, 36, endpoint=False)

        for th in thetas:
            cos_t, sin_t = math.cos(th), math.sin(th)
            rx = cos_t * sub_pts[:, 0] - sin_t * sub_pts[:, 1]
            ry = sin_t * sub_pts[:, 0] + cos_t * sub_pts[:, 1]

            dc = np.round(rx / res).astype(int)
            dr = np.round(ry / res).astype(int)

            min_dc = min(0, int(np.min(dc)))
            max_dc = max(0, int(np.max(dc)))
            min_dr = min(0, int(np.min(dr)))
            max_dr = max(0, int(np.max(dr)))

            W = max_dc - min_dc + 1
            H = max_dr - min_dr + 1

            if W >= grid.cols or H >= grid.rows:
                continue

            K = np.zeros((H, W), dtype=np.float32)
            K[dr - min_dr, dc - min_dc] = 1.0

            corr = scipy.signal.correlate(likelihood, K, mode="valid", method="fft")
            max_idx = np.unravel_index(np.argmax(corr), corr.shape)
            score = float(corr[max_idx])

            sensor_r = max_idx[0] - min_dr
            sensor_c = max_idx[1] - min_dc
            wx, wy = grid.grid_to_world(sensor_c, sensor_r)

            candidates.append((score, np.array([wx, wy, th], dtype=np.float64)))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[:top_k]

    def relocalize(
        self,
        submaps: List[Submap],
        points_sensor: np.ndarray,
        global_grid: Optional[OccupancyGrid] = None,
    ) -> RelocalizationResult:
        """Search across all saved submaps to find the true sensor pose."""
        best_result: Optional[MatchResult] = None
        best_submap_id = -1

        for submap in submaps:
            top_candidates = self.search_submap(submap, points_sensor, top_k=3)

            for _, cand_pose in top_candidates:
                # Refine with fine Gauss-Newton matcher on submap grid
                refined = self.fine_matcher.match(
                    grid=submap.grid,
                    points_sensor=points_sensor,
                    initial_pose=cand_pose,
                    use_coarse_search=False,
                )

                if (
                    best_result is None
                    or refined.confidence_score > best_result.confidence_score
                ):
                    best_result = refined
                    best_submap_id = submap.submap_id

        if best_result is not None and best_result.confidence_score >= self.min_confidence:
            return RelocalizationResult(
                success=True,
                pose=best_result.pose,
                confidence_score=best_result.confidence_score,
                matched_submap_id=best_submap_id,
                mean_residual_m=best_result.mean_residual_m,
            )

        return RelocalizationResult(
            success=False,
            pose=best_result.pose if best_result else np.zeros(3),
            confidence_score=best_result.confidence_score if best_result else 0.0,
            matched_submap_id=best_submap_id,
            mean_residual_m=best_result.mean_residual_m if best_result else 999.0,
        )
