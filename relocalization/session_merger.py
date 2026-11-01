"""Multi-room session coordinator and overlap-driven map merger."""

from dataclasses import dataclass
from enum import Enum
import logging
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from relocalization.correlative_matcher import CorrelativeGlobalMatcher
from slam.occupancy_grid import OccupancyGrid
from slam.pose_graph import PoseGraph
from slam.scan_matcher import LikelihoodFieldScanMatcher, MatchResult
from slam.submap import Submap

logger = logging.getLogger(__name__)


class MappingStatus(str, Enum):
    STARTUP_LOCALIZING = "STARTUP_LOCALIZING"
    TRACKING_EXISTING = "TRACKING_EXISTING"
    NEW_SESSION_UNLINKED = "NEW_SESSION_UNLINKED"
    MERGED_TRACKING = "MERGED_TRACKING"
    KIDNAPPED_RECOVERY = "KIDNAPPED_RECOVERY"


@dataclass
class OverlapCandidate:
    source_submap_id: int        # Room B submap
    target_submap_id: int        # Room A submap
    relative_pose: np.ndarray    # [dx, dy, dtheta]
    confidence_score: float
    timestamp: float


class MultiRoomSessionManager:
    """Manages multi-room sessions, room isolation, and overlap-based map merging."""

    def __init__(
        self,
        min_reloc_confidence: float = 0.70,
        min_merge_confidence: float = 0.75,
        consecutive_matches_needed: int = 3,
        trans_agreement_tol_m: float = 0.20,
        rot_agreement_tol_rad: float = 0.10,
    ):
        self.min_reloc_confidence = min_reloc_confidence
        self.min_merge_confidence = min_merge_confidence
        self.consecutive_matches_needed = consecutive_matches_needed
        self.trans_agreement_tol = trans_agreement_tol_m
        self.rot_agreement_tol = rot_agreement_tol_rad

        self.global_matcher = CorrelativeGlobalMatcher(
            coarse_res_m=0.20,
            coarse_angle_step_deg=3.0,
            min_confidence_score=min_reloc_confidence,
        )
        self.status = MappingStatus.TRACKING_EXISTING
        self._candidate_history: List[OverlapCandidate] = []
        self._merged_sessions: set = set()

    def attempt_startup_relocalization(
        self,
        saved_submaps: List[Submap],
        initial_scans_points: List[np.ndarray],
    ) -> Tuple[bool, np.ndarray, float]:
        """Try to relocalize inside saved maps upon startup.
        
        Args:
            saved_submaps: Submaps loaded from disk (e.g. Room A).
            initial_scans_points: 1-3 initial scan point sets from current position.
            
        Returns:
            Tuple of (success, pose, confidence).
        """
        if not saved_submaps or not initial_scans_points:
            self.status = MappingStatus.NEW_SESSION_UNLINKED
            return False, np.zeros(3), 0.0

        # Combine initial scans for higher point density
        pts_combined = np.vstack(initial_scans_points)
        result = self.global_matcher.relocalize(saved_submaps, pts_combined)

        if result.success and result.confidence_score >= self.min_reloc_confidence:
            self.status = MappingStatus.TRACKING_EXISTING
            logger.info(
                f"Relocalization successful! Found pose {result.pose} with confidence {result.confidence_score:.2f}"
            )
            return True, result.pose, result.confidence_score
        else:
            self.status = MappingStatus.NEW_SESSION_UNLINKED
            logger.info(
                f"Relocalization failed (confidence {result.confidence_score:.2f} < {self.min_reloc_confidence}). "
                f"Device is in a new/unmapped room. Starting unlinked session."
            )
            return False, np.zeros(3), result.confidence_score

    def check_for_cross_room_merge(
        self,
        active_submap_b: Submap,
        saved_submaps_a: List[Submap],
        pose_graph: PoseGraph,
        current_sensor_points: np.ndarray,
    ) -> bool:
        """Test if the current room (Room B) overlaps with previously mapped room (Room A).
        
        Requires `consecutive_matches_needed` agreeing matches to prevent false loop closures.
        
        Returns:
            True if merged, False otherwise.
        """
        if self.status == MappingStatus.MERGED_TRACKING:
            return False  # Already merged

        if not saved_submaps_a:
            return False

        # Attempt to match current scan against all Room A submaps
        best_cand: Optional[OverlapCandidate] = None

        for submap_a in saved_submaps_a:
            if submap_a.session_id == active_submap_b.session_id:
                continue

            top_candidates = self.global_matcher.search_submap(submap_a, current_sensor_points, top_k=2)
            for _, cand_pose in top_candidates:
                refined = self.global_matcher.fine_matcher.match(
                    grid=submap_a.grid,
                    points_sensor=current_sensor_points,
                    initial_pose=cand_pose,
                )
                if refined.confidence_score >= self.min_merge_confidence:
                    if best_cand is None or refined.confidence_score > best_cand.confidence_score:
                        # Compute relative transformation from submap_a to active_submap_b
                        # refined.pose is sensor pose in submap_a frame
                        best_cand = OverlapCandidate(
                            source_submap_id=active_submap_b.submap_id,
                            target_submap_id=submap_a.submap_id,
                            relative_pose=refined.pose,
                            confidence_score=refined.confidence_score,
                            timestamp=0.0,
                        )

        if best_cand is None:
            self._candidate_history.clear()
            return False

        # Check consistency with previous candidates
        if len(self._candidate_history) > 0:
            last = self._candidate_history[-1]
            dx = abs(best_cand.relative_pose[0] - last.relative_pose[0])
            dy = abs(best_cand.relative_pose[1] - last.relative_pose[1])
            dth = abs((best_cand.relative_pose[2] - last.relative_pose[2] + math.pi) % (2 * math.pi) - math.pi)

            if math.hypot(dx, dy) < self.trans_agreement_tol and dth < self.rot_agreement_tol:
                self._candidate_history.append(best_cand)
            else:
                # Reset consistency chain on disagreement
                self._candidate_history = [best_cand]
        else:
            self._candidate_history.append(best_cand)

        logger.info(
            f"Cross-room overlap candidate: confidence={best_cand.confidence_score:.2f}, "
            f"consecutive_count={len(self._candidate_history)}/{self.consecutive_matches_needed}"
        )

        # Trigger merge when consecutive consistency is met
        if len(self._candidate_history) >= self.consecutive_matches_needed:
            self._execute_merge(
                cand=best_cand,
                active_submap_b=active_submap_b,
                saved_submaps_a=saved_submaps_a,
                pose_graph=pose_graph,
            )
            self._candidate_history.clear()
            self.status = MappingStatus.MERGED_TRACKING
            return True

        return False

    def _execute_merge(
        self,
        cand: OverlapCandidate,
        active_submap_b: Submap,
        saved_submaps_a: List[Submap],
        pose_graph: PoseGraph,
    ) -> None:
        """Add inter-room edge, run PGO, and align Room B coordinates into Room A."""
        logger.info(
            f"Executing cross-room map merge between Room B (Submap {cand.source_submap_id}) "
            f"and Room A (Submap {cand.target_submap_id})."
        )

        # Find a node in submap A and a node in submap B to connect
        node_a_id = None
        for sm in saved_submaps_a:
            if sm.submap_id == cand.target_submap_id and len(sm.keyframe_ids) > 0:
                node_a_id = sm.keyframe_ids[0]
                break

        node_b_id = active_submap_b.keyframe_ids[-1] if len(active_submap_b.keyframe_ids) > 0 else None

        if node_a_id is not None and node_b_id is not None:
            # Add inter-session loop closure edge
            pose_graph.add_edge(
                from_id=node_a_id,
                to_id=node_b_id,
                relative_pose=cand.relative_pose,
                info_matrix=np.diag([200.0, 200.0, 1000.0]),
                is_loop_closure=True,
            )

            # Optimize pose graph
            pose_graph.optimize(max_iterations=25, fixed_node_id=0)

            # Update submap B's origin pose from optimized node
            if node_b_id in pose_graph.nodes:
                active_submap_b.origin_pose = pose_graph.nodes[node_b_id].pose.copy()

            self._merged_sessions.add(active_submap_b.session_id)
            logger.info("Pose graph optimized. Coordinate frames linked successfully.")
