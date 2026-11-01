"""Unified 2D SLAM System integrating scan matching, submaps, pose graph, and multi-room persistence."""

import logging
import math
import time
from typing import Any, Dict, List, Optional
import numpy as np

from driver.interface import Scan
from persistence.map_manager import MapManager
from persistence.ros_exporter import RosMapExporter
from relocalization.session_merger import MappingStatus, MultiRoomSessionManager
from slam.occupancy_grid import OccupancyGrid
from slam.pose_graph import PoseGraph
from slam.preprocessor import ScanPreprocessor
from slam.scan_matcher import LikelihoodFieldScanMatcher, MatchResult
from slam.submap import Submap

logger = logging.getLogger(__name__)


class LidarSlamSystem:
    """Full Persistent 2D Lidar SLAM System."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        slam_cfg = config.get("slam", {})
        grid_cfg = config.get("occupancy_grid", {})
        lidar_cfg = config.get("lidar", {})
        reloc_cfg = config.get("relocalization", {})
        pers_cfg = config.get("persistence", {})

        # Components
        self.preprocessor = ScanPreprocessor(
            range_min_m=float(lidar_cfg.get("range_min_m", 0.05)),
            range_max_m=float(lidar_cfg.get("clip_range_max_m", 25.0)),
            angular_downsample_deg=float(lidar_cfg.get("angular_downsample_deg", 0.25)),
            outlier_min_neighbors=int(lidar_cfg.get("outlier_min_neighbors", 2)),
            outlier_radius_m=float(lidar_cfg.get("outlier_neighbor_radius_m", 0.15)),
        )

        self.scan_matcher = LikelihoodFieldScanMatcher(
            max_iterations=int(slam_cfg.get("max_iterations", 25)),
            convergence_tol=float(slam_cfg.get("convergence_tol", 1e-4)),
            huber_delta_m=float(slam_cfg.get("huber_delta_m", 0.10)),
            min_hessian_eigenvalue=float(slam_cfg.get("min_hessian_eigenvalue", 15.0)),
            search_window_xy_m=float(slam_cfg.get("search_window_xy_m", 0.25)),
            search_window_theta_rad=float(slam_cfg.get("search_window_theta_rad", 0.20)),
        )

        self.map_manager = MapManager(
            maps_root_dir=pers_cfg.get("maps_dir", "maps"),
            autosave_interval_sec=float(pers_cfg.get("autosave_interval_sec", 15.0)),
        )

        self.session_manager = MultiRoomSessionManager(
            min_reloc_confidence=float(reloc_cfg.get("min_confidence_score", 0.70)),
            min_merge_confidence=float(reloc_cfg.get("merge_confidence_score", 0.75)),
            consecutive_matches_needed=int(reloc_cfg.get("consecutive_matches_needed", 3)),
        )

        # SLAM parameters
        self.min_translation_kf = float(slam_cfg.get("min_translation_kf_m", 0.15))
        self.min_rotation_kf = float(slam_cfg.get("min_rotation_kf_rad", 0.10))
        self.submap_max_keyframes = int(slam_cfg.get("submap_max_keyframes", 30))
        self.submap_max_extent = float(slam_cfg.get("submap_max_extent_m", 12.0))
        self.resolution = float(grid_cfg.get("resolution_m", 0.05))

        # SLAM state
        self.current_pose = np.zeros(3, dtype=np.float64)  # [x, y, theta]
        self.last_keyframe_pose = np.zeros(3, dtype=np.float64)
        self.current_session_id = "session_0"
        self.submaps: List[Submap] = []
        self.active_submap: Optional[Submap] = None
        self.pose_graph = PoseGraph()
        self.global_grid = OccupancyGrid(resolution_m=self.resolution)

        self.keyframe_count = 0
        self.node_count = 0
        self.keyframes_data: Dict[int, np.ndarray] = {}
        self._initial_startup_scans: List[np.ndarray] = []
        self._is_relocalized = False

        self._init_first_submap()

    def _init_first_submap(self) -> None:
        """Create initial Submap and PoseGraph root."""
        submap_id = len(self.submaps)
        self.active_submap = Submap(
            submap_id=submap_id,
            session_id=self.current_session_id,
            origin_pose=self.current_pose,
            resolution_m=self.resolution,
            extent_m=self.submap_max_extent,
        )
        self.submaps.append(self.active_submap)

        # Root node in pose graph
        self.pose_graph.add_node(
            node_id=self.node_count,
            pose=self.current_pose,
            session_id=self.current_session_id,
            submap_id=submap_id,
            timestamp=time.time(),
        )
        self.node_count += 1

    def process_scan(self, raw_scan: Scan) -> Dict[str, Any]:
        """Main SLAM update step for an incoming raw Lidar scan."""
        prep_scan = self.preprocessor.process(raw_scan)
        if prep_scan is None:
            return {
                "pose": self.current_pose.copy(),
                "status": self.session_manager.status.value,
                "confidence": 0.0,
                "is_degenerate": False,
            }

        pts_sensor = prep_scan.points_sensor

        # Startup relocalization check if loaded with a saved map
        if self.session_manager.status == MappingStatus.STARTUP_LOCALIZING and not self._is_relocalized:
            self._initial_startup_scans.append(pts_sensor)
            if len(self._initial_startup_scans) >= 2:
                saved_sms = [s for s in self.submaps if s.is_finished or s.num_points_mapped > 0]
                if not saved_sms:
                    saved_sms = self.submaps

                success, found_pose, conf = self.session_manager.attempt_startup_relocalization(
                    saved_submaps=saved_sms,
                    initial_scans_points=self._initial_startup_scans,
                )
                if success:
                    self.current_pose = found_pose.copy()
                    self.last_keyframe_pose = found_pose.copy()
                    self._is_relocalized = True
                    # Set active submap to the one with closest origin
                    best_sm = min(
                        saved_sms,
                        key=lambda s: math.hypot(s.origin_pose[0] - found_pose[0], s.origin_pose[1] - found_pose[1]),
                    )
                    self.active_submap = best_sm
                    logger.info(f"Relocalized inside Submap {best_sm.submap_id} at pose {self.current_pose}")
                else:
                    # Sensor is in an unmapped room! Start unlinked Session B
                    self._start_new_unlinked_session("session_1")

            return {
                "pose": self.current_pose.copy(),
                "status": self.session_manager.status.value,
                "confidence": 0.0,
                "is_degenerate": False,
            }

        # Pose Tracking via Scan-to-Submap Matching
        assert self.active_submap is not None
        match_result = self.scan_matcher.match(
            grid=self.active_submap.grid,
            points_sensor=pts_sensor,
            initial_pose=self.current_pose,
            use_coarse_search=prep_scan.is_degenerate or (self.session_manager.status == MappingStatus.KIDNAPPED_RECOVERY),
        )

        # Handle tracking health
        if match_result.confidence_score < 0.25 and self.active_submap.num_points_mapped > 500:
            logger.warning(f"Tracking lost! Low confidence: {match_result.confidence_score:.2f}")
            self.session_manager.status = MappingStatus.KIDNAPPED_RECOVERY
        else:
            if self.session_manager.status == MappingStatus.KIDNAPPED_RECOVERY:
                self.session_manager.status = MappingStatus.TRACKING_EXISTING

            # Update current estimated pose
            self.current_pose = match_result.pose.copy()

        # Check for cross-room merge if in an unlinked room
        if self.session_manager.status == MappingStatus.NEW_SESSION_UNLINKED:
            saved_sms = [s for s in self.submaps if s.session_id != self.current_session_id]
            merged = self.session_manager.check_for_cross_room_merge(
                active_submap_b=self.active_submap,
                saved_submaps_a=saved_sms,
                pose_graph=self.pose_graph,
                current_sensor_points=pts_sensor,
            )
            if merged:
                # Update global grid
                self.global_grid = self.map_manager.render_composite_global_grid(self.submaps, self.pose_graph)

        # Keyframe decision
        dx = self.current_pose[0] - self.last_keyframe_pose[0]
        dy = self.current_pose[1] - self.last_keyframe_pose[1]
        dist_moved = math.hypot(dx, dy)
        rot_moved = abs((self.current_pose[2] - self.last_keyframe_pose[2] + math.pi) % (2 * math.pi) - math.pi)

        is_keyframe = (
            (dist_moved >= self.min_translation_kf)
            or (rot_moved >= self.min_rotation_kf)
            or (self.keyframe_count == 0)
        )

        if is_keyframe and (not match_result.is_degenerate or self.keyframe_count == 0):
            self._add_keyframe(pts_sensor, raw_scan.timestamp)

        # Transform scan points to world for visualization
        cos_t = math.cos(self.current_pose[2])
        sin_t = math.sin(self.current_pose[2])
        pts_w = np.column_stack((
            cos_t * pts_sensor[:, 0] - sin_t * pts_sensor[:, 1] + self.current_pose[0],
            sin_t * pts_sensor[:, 0] + cos_t * pts_sensor[:, 1] + self.current_pose[1],
        )).astype(np.float32)

        return {
            "pose": self.current_pose.copy(),
            "status": self.session_manager.status.value,
            "confidence": match_result.confidence_score,
            "is_degenerate": match_result.is_degenerate or prep_scan.is_degenerate,
            "mean_residual": match_result.mean_residual_m,
            "num_submaps": len(self.submaps),
            "num_keyframes": self.keyframe_count,
            "active_submap_id": self.active_submap.submap_id,
            "pts_world": pts_w,
        }

    def _add_keyframe(self, pts_sensor: np.ndarray, timestamp: float) -> None:
        """Spawn and record a new keyframe in the active submap and pose graph."""
        assert self.active_submap is not None
        kf_id = self.keyframe_count
        self.keyframe_count += 1
        self.keyframes_data[kf_id] = pts_sensor.copy()

        # Add node to pose graph
        node_id = self.node_count
        self.node_count += 1
        self.pose_graph.add_node(
            node_id=node_id,
            pose=self.current_pose,
            session_id=self.current_session_id,
            submap_id=self.active_submap.submap_id,
            timestamp=timestamp,
        )

        # Add sequential odometry constraint edge
        prev_node_id = node_id - 1
        if prev_node_id >= 0 and prev_node_id in self.pose_graph.nodes:
            prev_p = self.pose_graph.nodes[prev_node_id].pose
            # Relative transform in prev frame: R^T (curr - prev)
            cos_p = math.cos(prev_p[2])
            sin_p = math.sin(prev_p[2])
            dx_w = self.current_pose[0] - prev_p[0]
            dy_w = self.current_pose[1] - prev_p[1]
            rel_dx = cos_p * dx_w + sin_p * dy_w
            rel_dy = -sin_p * dx_w + cos_p * dy_w
            rel_dth = (self.current_pose[2] - prev_p[2] + math.pi) % (2 * math.pi) - math.pi
            rel_pose = np.array([rel_dx, rel_dy, rel_dth])

            self.pose_graph.add_edge(prev_node_id, node_id, rel_pose)

        # Update active submap occupancy grid
        self.active_submap.add_keyframe(
            keyframe_id=kf_id,
            sensor_pose=self.current_pose,
            points_sensor=pts_sensor,
        )
        self.last_keyframe_pose = self.current_pose.copy()

        # Check if submap should freeze and spawn next submap
        submap_dist = math.hypot(
            self.current_pose[0] - self.active_submap.origin_pose[0],
            self.current_pose[1] - self.active_submap.origin_pose[1],
        )
        if len(self.active_submap.keyframe_ids) >= self.submap_max_keyframes or submap_dist >= (self.submap_max_extent * 0.7):
            self._spawn_next_submap()

    def _spawn_next_submap(self) -> None:
        """Freeze current submap and create overlapping subsequent submap."""
        assert self.active_submap is not None
        self.active_submap.finish()

        new_submap_id = len(self.submaps)
        self.active_submap = Submap(
            submap_id=new_submap_id,
            session_id=self.current_session_id,
            origin_pose=self.current_pose,
            resolution_m=self.resolution,
            extent_m=self.submap_max_extent,
        )
        self.submaps.append(self.active_submap)
        logger.info(f"Spawned new Submap {new_submap_id} at pose {self.current_pose}.")

    def _start_new_unlinked_session(self, new_session_id: str) -> None:
        """Start an independent coordinate frame session for an unmapped room."""
        logger.info(f"Starting unlinked multi-room session '{new_session_id}'...")
        self.current_session_id = new_session_id
        self.session_manager.status = MappingStatus.NEW_SESSION_UNLINKED
        # Reset local frame pose to origin (0, 0, 0) for Room B
        self.current_pose = np.zeros(3, dtype=np.float64)
        self.last_keyframe_pose = np.zeros(3, dtype=np.float64)

        if self.active_submap is not None:
            self.active_submap.finish()

        self._init_first_submap()

    def save_map(self, map_name: str) -> str:
        """Save current full SLAM state to named map folder."""
        if self.active_submap is not None:
            self.active_submap.grid.compute_distance_field()

        self.global_grid = self.map_manager.render_composite_global_grid(self.submaps, self.pose_graph)
        return self.map_manager.save_session(
            map_name=map_name,
            submaps=self.submaps,
            pose_graph=self.pose_graph,
            global_grid=self.global_grid,
            keyframes_data=self.keyframes_data,
        )

    def load_map(self, map_name: str) -> None:
        """Load saved map from disk and set status to STARTUP_LOCALIZING."""
        submaps, pose_graph, global_grid, kf_data = self.map_manager.load_session(map_name)

        self.submaps = submaps
        self.pose_graph = pose_graph
        self.global_grid = global_grid
        self.keyframes_data = kf_data
        self.keyframe_count = max(list(kf_data.keys())) + 1 if kf_data else 0
        self.node_count = max(list(pose_graph.nodes.keys())) + 1 if pose_graph.nodes else 0

        self.session_manager.status = MappingStatus.STARTUP_LOCALIZING
        self._is_relocalized = False
        self._initial_startup_scans.clear()

        # Point active submap to last saved submap
        self.active_submap = self.submaps[-1] if self.submaps else None
        logger.info(f"Loaded map '{map_name}'. Ready for relocalization.")
