"""Scan Matcher: Robust Likelihood Field Gauss-Newton with Huber loss and Coarse Correlative Search."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np

from slam.occupancy_grid import OccupancyGrid


@dataclass
class MatchResult:
    """Result of scan-to-submap matching."""
    pose: np.ndarray             # shape (3,), [x, y, theta]
    converged: bool
    iterations: int
    mean_residual_m: float
    confidence_score: float      # [0.0, 1.0]
    hessian: np.ndarray          # shape (3, 3)
    is_degenerate: bool
    min_eigenvalue: float


class LikelihoodFieldScanMatcher:
    """Robust 2D Scan-to-Submap Matcher."""

    def __init__(
        self,
        max_iterations: int = 25,
        convergence_tol: float = 1e-4,
        huber_delta_m: float = 0.10,
        min_hessian_eigenvalue: float = 15.0,
        search_window_xy_m: float = 0.25,
        search_window_theta_rad: float = 0.20,
    ):
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.huber_delta = huber_delta_m
        self.min_hessian_eigenvalue = min_hessian_eigenvalue
        self.search_window_xy = search_window_xy_m
        self.search_window_theta = search_window_theta_rad

    def _compute_cost(self, grid: OccupancyGrid, points_sensor: np.ndarray, pose: np.ndarray) -> float:
        x, y, theta = pose[0], pose[1], pose[2]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        px_w = cos_t * points_sensor[:, 0] - sin_t * points_sensor[:, 1] + x
        py_w = sin_t * points_sensor[:, 0] + cos_t * points_sensor[:, 1] + y
        pts_world = np.column_stack((px_w, py_w)).astype(np.float32)
        dists, _, _, valid = grid.query_distance_and_gradient(pts_world)
        if not np.any(valid):
            return 1e6
        r = dists[valid]
        huber = np.where(r <= self.huber_delta, 0.5 * r**2, self.huber_delta * (r - 0.5 * self.huber_delta))
        return float(np.sum(huber))

    def match(
        self,
        grid: OccupancyGrid,
        points_sensor: np.ndarray,
        initial_pose: np.ndarray,
        use_coarse_search: bool = False,
    ) -> MatchResult:
        """Align local sensor points against the occupancy grid's distance field.
        
        Args:
            grid: OccupancyGrid with populated obstacles and distance field.
            points_sensor: shape (N, 2), points in sensor local coordinates.
            initial_pose: shape (3,), [x, y, theta].
            use_coarse_search: If True, first perform coarse grid search around initial_pose.
            
        Returns:
            MatchResult containing refined pose, confidence, and degeneracy metrics.
        """
        pose = initial_pose.copy().astype(np.float64)

        if use_coarse_search:
            pose = self._coarse_correlative_search(grid, points_sensor, pose)

        converged = False
        final_hessian = np.eye(3, dtype=np.float64)
        min_eig = 0.0
        current_cost = self._compute_cost(grid, points_sensor, pose)

        for it in range(self.max_iterations):
            # Transform points to world coordinates using current candidate pose
            x, y, theta = pose[0], pose[1], pose[2]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            px_w = cos_t * points_sensor[:, 0] - sin_t * points_sensor[:, 1] + x
            py_w = sin_t * points_sensor[:, 0] + cos_t * points_sensor[:, 1] + y
            pts_world = np.column_stack((px_w, py_w)).astype(np.float32)

            dists, gx, gy, valid = grid.query_distance_and_gradient(pts_world)
            if np.sum(valid) < 15:
                break

            r = dists[valid]  # residuals (meters)
            val_gx = gx[valid]
            val_gy = gy[valid]
            p_sx = points_sensor[valid, 0]
            p_sy = points_sensor[valid, 1]

            # Jacobian J_i = [d(dist)/dx, d(dist)/dy, d(dist)/dtheta]
            d_rot_x = -sin_t * p_sx - cos_t * p_sy
            d_rot_y = cos_t * p_sx - sin_t * p_sy
            d_theta = val_gx * d_rot_x + val_gy * d_rot_y

            J = np.column_stack((val_gx, val_gy, d_theta))  # shape (M, 3)

            # Huber Loss weights
            abs_r = np.abs(r)
            weights = np.ones_like(r)
            outlier_mask = abs_r > self.huber_delta
            weights[outlier_mask] = self.huber_delta / (abs_r[outlier_mask] + 1e-6)

            # Normal equations: (J^T W J) delta_p = - J^T W r
            W_diag = np.sqrt(weights)
            JW = J * W_diag[:, None]
            rw = r * W_diag

            H = JW.T @ JW  # shape (3, 3)
            g = JW.T @ rw  # shape (3,)

            # Levenberg-Marquardt regularization
            H_reg = H + 1e-3 * np.eye(3)

            try:
                delta_p = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                break

            # Backtracking line search to ensure monotonic cost reduction
            alpha = 1.0
            accepted = False
            for _ in range(5):
                cand_pose = pose.copy()
                cand_pose[0] += alpha * delta_p[0]
                cand_pose[1] += alpha * delta_p[1]
                cand_pose[2] = (cand_pose[2] + alpha * delta_p[2] + math.pi) % (2 * math.pi) - math.pi
                cand_cost = self._compute_cost(grid, points_sensor, cand_pose)
                if cand_cost < current_cost:
                    pose = cand_pose
                    current_cost = cand_cost
                    accepted = True
                    break
                alpha *= 0.5

            step_norm = math.hypot(alpha * delta_p[0], alpha * delta_p[1])
            step_rot = abs(alpha * delta_p[2])
            final_hessian = H

            if step_norm < 1e-3 and step_rot < 1e-3:
                converged = True
                break

        # Compute final metrics
        x, y, theta = pose[0], pose[1], pose[2]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        px_w = cos_t * points_sensor[:, 0] - sin_t * points_sensor[:, 1] + x
        py_w = sin_t * points_sensor[:, 0] + cos_t * points_sensor[:, 1] + y
        pts_world = np.column_stack((px_w, py_w)).astype(np.float32)

        dists, _, _, valid = grid.query_distance_and_gradient(pts_world)
        mean_res = float(np.mean(dists[valid])) if np.any(valid) else 999.0

        # Confidence: percentage of points within 1.5 grid cells of obstacle
        match_thresh = 1.5 * grid.resolution
        close_points = np.sum(dists[valid] <= match_thresh)
        confidence = float(close_points / max(1, len(points_sensor)))

        # Degeneracy check from Hessian
        eigvals = np.linalg.eigvalsh(final_hessian)
        min_eig = float(np.min(eigvals))
        is_degenerate = min_eig < self.min_hessian_eigenvalue

        return MatchResult(
            pose=pose,
            converged=converged,
            iterations=it + 1,
            mean_residual_m=mean_res,
            confidence_score=confidence,
            hessian=final_hessian,
            is_degenerate=is_degenerate,
            min_eigenvalue=min_eig,
        )

    def _coarse_correlative_search(
        self, grid: OccupancyGrid, points_sensor: np.ndarray, prior_pose: np.ndarray
    ) -> np.ndarray:
        """Coarse grid search over a 3D box (x, y, theta) around prior pose."""
        step_xy = max(0.05, grid.resolution)
        step_th = np.radians(3.0)

        xs = np.arange(
            prior_pose[0] - self.search_window_xy,
            prior_pose[0] + self.search_window_xy + 1e-4,
            step_xy,
        )
        ys = np.arange(
            prior_pose[1] - self.search_window_xy,
            prior_pose[1] + self.search_window_xy + 1e-4,
            step_xy,
        )
        thetas = np.arange(
            prior_pose[2] - self.search_window_theta,
            prior_pose[2] + self.search_window_theta + 1e-4,
            step_th,
        )

        best_score = float("inf")
        best_pose = prior_pose.copy()

        # Subsample points for coarse evaluation speed
        sub_pts = points_sensor[::4] if len(points_sensor) > 100 else points_sensor

        for th in thetas:
            cos_t = math.cos(th)
            sin_t = math.sin(th)
            rx = cos_t * sub_pts[:, 0] - sin_t * sub_pts[:, 1]
            ry = sin_t * sub_pts[:, 0] + cos_t * sub_pts[:, 1]

            for x_cand in xs:
                for y_cand in ys:
                    cand_pts_world = np.column_stack((rx + x_cand, ry + y_cand)).astype(np.float32)
                    dists, _, _, valid = grid.query_distance_and_gradient(cand_pts_world)
                    if not np.any(valid):
                        continue
                    # Huber score
                    res = dists[valid]
                    score = np.sum(np.minimum(res, 0.20))
                    if score < best_score:
                        best_score = score
                        best_pose = np.array([x_cand, y_cand, th])

        return best_pose
