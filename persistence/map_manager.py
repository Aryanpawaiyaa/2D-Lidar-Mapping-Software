"""Map persistence manager: save/load full session state and handle autosaves."""

import io
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from persistence.atomic_io import atomic_write_bytes, atomic_write_text
from persistence.ros_exporter import RosMapExporter
from slam.occupancy_grid import OccupancyGrid
from slam.pose_graph import PoseEdge, PoseGraph, PoseNode
from slam.submap import Submap

logger = logging.getLogger(__name__)


class MapManager:
    """Manages saving, loading, and autosaving of multi-room SLAM sessions."""

    def __init__(self, maps_root_dir: str = "maps", autosave_interval_sec: float = 15.0):
        self.maps_root = maps_root_dir
        self.autosave_interval = autosave_interval_sec
        self._last_autosave_time = time.time()
        os.makedirs(self.maps_root, exist_ok=True)

    def save_session(
        self,
        map_name: str,
        submaps: List[Submap],
        pose_graph: PoseGraph,
        global_grid: OccupancyGrid,
        keyframes_data: Optional[Dict[int, np.ndarray]] = None,
    ) -> str:
        """Save complete SLAM session state atomically to disk.
        
        Args:
            map_name: Unique name or folder for the map.
            submaps: List of active and finished Submaps.
            pose_graph: PoseGraph containing all nodes and edges.
            global_grid: Rendered composite global occupancy grid.
            keyframes_data: Optional dict mapping keyframe_id -> points_sensor.
            
        Returns:
            Absolute path to saved map directory.
        """
        map_dir = os.path.join(self.maps_root, map_name)
        os.makedirs(map_dir, exist_ok=True)

        # 1. Prepare session metadata and pose graph JSON
        nodes_list = []
        for nid, node in sorted(pose_graph.nodes.items()):
            nodes_list.append({
                "node_id": node.node_id,
                "pose": node.pose.tolist(),
                "session_id": node.session_id,
                "submap_id": node.submap_id,
                "timestamp": node.timestamp,
            })

        edges_list = []
        for edge in pose_graph.edges:
            edges_list.append({
                "from_id": edge.from_id,
                "to_id": edge.to_id,
                "relative_pose": edge.relative_pose.tolist(),
                "info_matrix": edge.info_matrix.tolist(),
                "is_loop_closure": edge.is_loop_closure,
            })

        submaps_meta = []
        submap_grids_dict = {}
        for s in submaps:
            submaps_meta.append({
                "submap_id": s.submap_id,
                "session_id": s.session_id,
                "origin_pose": s.origin_pose.tolist(),
                "resolution_m": s.resolution,
                "extent_m": s.extent,
                "origin_x_m": s.grid.origin_x,
                "origin_y_m": s.grid.origin_y,
                "is_finished": s.is_finished,
                "keyframe_ids": s.keyframe_ids,
            })
            submap_grids_dict[f"grid_{s.submap_id}"] = s.grid.grid

        state_doc = {
            "version": "1.0.0",
            "timestamp": time.time(),
            "map_name": map_name,
            "num_submaps": len(submaps),
            "num_nodes": len(nodes_list),
            "num_edges": len(edges_list),
            "submaps": submaps_meta,
            "nodes": nodes_list,
            "edges": edges_list,
        }

        # 2. Save JSON state atomically
        json_path = os.path.join(map_dir, "session_state.json")
        atomic_write_text(json_path, json.dumps(state_doc, indent=2))

        # 3. Save Submap grids array atomically
        submaps_npz_path = os.path.join(map_dir, "submaps.npz")
        buf = io.BytesIO()
        np.savez_compressed(buf, **submap_grids_dict)
        atomic_write_bytes(submaps_npz_path, buf.getvalue())

        # 4. Save Keyframe raw points (if available)
        if keyframes_data:
            kf_dict = {f"kf_{k}": v for k, v in keyframes_data.items()}
            kf_npz_path = os.path.join(map_dir, "keyframes.npz")
            buf_kf = io.BytesIO()
            np.savez_compressed(buf_kf, **kf_dict)
            atomic_write_bytes(kf_npz_path, buf_kf.getvalue())

        # 5. Export ROS map_server compatible PNG + YAML
        RosMapExporter.export(
            grid=global_grid,
            output_dir=map_dir,
            map_name="map",
        )

        logger.info(f"Saved session '{map_name}' successfully to {map_dir}")
        self._last_autosave_time = time.time()
        return map_dir

    def load_session(
        self, map_name: str
    ) -> Tuple[List[Submap], PoseGraph, OccupancyGrid, Dict[int, np.ndarray]]:
        """Load complete saved SLAM session from disk.
        
        Args:
            map_name: Name or folder of the map to load.
            
        Returns:
            Tuple of (submaps, pose_graph, global_grid, keyframes_dict).
        """
        map_dir = os.path.join(self.maps_root, map_name)
        json_path = os.path.join(map_dir, "session_state.json")
        submaps_npz_path = os.path.join(map_dir, "submaps.npz")

        if not os.path.exists(json_path) or not os.path.exists(submaps_npz_path):
            raise FileNotFoundError(f"Map session files not found in: {map_dir}")

        with open(json_path, "r", encoding="utf-8") as f:
            state_doc = json.load(f)

        submap_grids = np.load(submaps_npz_path)

        # Reconstruct Submaps
        submaps: List[Submap] = []
        for sm_meta in state_doc["submaps"]:
            s = Submap(
                submap_id=sm_meta["submap_id"],
                session_id=sm_meta["session_id"],
                origin_pose=np.array(sm_meta["origin_pose"], dtype=np.float64),
                resolution_m=sm_meta["resolution_m"],
                extent_m=sm_meta["extent_m"],
            )
            s.is_finished = sm_meta.get("is_finished", True)
            s.keyframe_ids = sm_meta.get("keyframe_ids", [])

            grid_key = f"grid_{s.submap_id}"
            if grid_key in submap_grids:
                s.grid.grid = submap_grids[grid_key].copy()
                s.grid.compute_distance_field()

            submaps.append(s)

        # Reconstruct PoseGraph
        pose_graph = PoseGraph()
        for n in state_doc["nodes"]:
            pose_graph.add_node(
                node_id=n["node_id"],
                pose=np.array(n["pose"], dtype=np.float64),
                session_id=n.get("session_id", "session_0"),
                submap_id=n.get("submap_id", 0),
                timestamp=n.get("timestamp", 0.0),
            )

        for e in state_doc["edges"]:
            pose_graph.add_edge(
                from_id=e["from_id"],
                to_id=e["to_id"],
                relative_pose=np.array(e["relative_pose"], dtype=np.float64),
                info_matrix=np.array(e["info_matrix"], dtype=np.float64),
                is_loop_closure=e.get("is_loop_closure", False),
            )

        # Load keyframes data if present
        keyframes_data: Dict[int, np.ndarray] = {}
        kf_npz_path = os.path.join(map_dir, "keyframes.npz")
        if os.path.exists(kf_npz_path):
            kf_loaded = np.load(kf_npz_path)
            for k, arr in kf_loaded.items():
                if k.startswith("kf_"):
                    kid = int(k.split("_")[1])
                    keyframes_data[kid] = arr

        # Re-render composite global occupancy grid
        global_grid = self.render_composite_global_grid(submaps, pose_graph)

        logger.info(f"Loaded session '{map_name}' ({len(submaps)} submaps, {len(pose_graph.nodes)} nodes).")
        return submaps, pose_graph, global_grid, keyframes_data

    @staticmethod
    def render_composite_global_grid(
        submaps: List[Submap], pose_graph: PoseGraph, resolution_m: float = 0.05
    ) -> OccupancyGrid:
        """Composite all submaps into a coherent global occupancy grid using optimized poses."""
        if not submaps:
            return OccupancyGrid(resolution_m=resolution_m)

        # Determine global bounding box from all submap poses
        all_x_min = min(s.origin_pose[0] - s.extent / 2.0 for s in submaps)
        all_x_max = max(s.origin_pose[0] + s.extent / 2.0 for s in submaps)
        all_y_min = min(s.origin_pose[1] - s.extent / 2.0 for s in submaps)
        all_y_max = max(s.origin_pose[1] + s.extent / 2.0 for s in submaps)

        margin = 2.0
        width_m = (all_x_max - all_x_min) + 2 * margin
        height_m = (all_y_max - all_y_min) + 2 * margin
        origin_x = all_x_min - margin
        origin_y = all_y_min - margin

        global_grid = OccupancyGrid(
            resolution_m=resolution_m,
            width_m=width_m,
            height_m=height_m,
            origin_x_m=origin_x,
            origin_y_m=origin_y,
        )

        for s in submaps:
            # Map submap cells to global grid
            prob = s.grid.get_probability_map()
            for r in range(s.grid.rows):
                for c in range(s.grid.cols):
                    p = prob[r, c]
                    # Only copy known cells (free or occupied)
                    if p >= s.grid.occupied_thresh or p <= s.grid.free_thresh:
                        wx, wy = s.grid.grid_to_world(c, r)
                        gc, gr = global_grid.world_to_grid(wx, wy)
                        if global_grid.is_in_bounds(gc, gr):
                            # Combine log odds: max magnitude
                            cur_log = global_grid.grid[gr, gc]
                            new_log = s.grid.grid[r, c]
                            if abs(new_log) > abs(cur_log):
                                global_grid.grid[gr, gc] = new_log

        global_grid.compute_distance_field()
        return global_grid

    def should_autosave(self) -> bool:
        """Check if autosave interval has elapsed."""
        return (time.time() - self._last_autosave_time) >= self.autosave_interval

