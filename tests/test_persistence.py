"""Unit tests for Persistence: Atomic I/O, ROS map export, and Save/Load roundtrip."""

import os
import tempfile
import numpy as np
import pytest
from PIL import Image
import yaml

from persistence.atomic_io import atomic_write_bytes, atomic_write_text
from persistence.ros_exporter import RosMapExporter
from persistence.map_manager import MapManager
from slam.occupancy_grid import OccupancyGrid
from slam.submap import Submap
from slam.pose_graph import PoseGraph


def test_atomic_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "subdir", "test_atomic.txt")
        atomic_write_text(test_file, "Antigravity SLAM Atomic Write")

        assert os.path.exists(test_file)
        with open(test_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == "Antigravity SLAM Atomic Write"


def test_ros_map_exporter():
    with tempfile.TemporaryDirectory() as tmpdir:
        grid = OccupancyGrid(resolution_m=0.05, width_m=6.0, height_m=4.0, origin_x_m=-3.0, origin_y_m=-2.0)
        # Mark an obstacle cell at (0, 0)
        c, r = grid.world_to_grid(0.0, 0.0)
        grid.grid[r, c] = grid.log_odds_thresh_occ + 1.0

        yaml_path, png_path = RosMapExporter.export(grid, output_dir=tmpdir, map_name="test_map")

        assert os.path.exists(yaml_path)
        assert os.path.exists(png_path)

        with open(yaml_path, "r") as f:
            meta = yaml.safe_load(f)

        assert meta["image"] == "test_map.png"
        assert meta["resolution"] == 0.05
        assert meta["origin"] == [-3.0, -2.0, 0.0]
        assert meta["occupied_thresh"] == 0.65
        assert meta["free_thresh"] == 0.25

        img = Image.open(png_path)
        assert img.size == (grid.cols, grid.rows)
        # Verify 8-bit grayscale mode
        assert img.mode == "L"


def test_session_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MapManager(maps_root_dir=tmpdir, autosave_interval_sec=5.0)

        # Create submaps
        s0 = Submap(submap_id=0, session_id="session_0", origin_pose=np.array([1.0, 2.0, 0.5]), extent_m=8.0)
        # Put some data in submap grid
        c, r = s0.grid.world_to_grid(1.0, 2.0)
        s0.grid.grid[r, c] = 2.5
        s0.finish()

        # Create pose graph
        graph = PoseGraph()
        graph.add_node(0, np.array([1.0, 2.0, 0.5]), session_id="session_0", submap_id=0)
        graph.add_node(1, np.array([2.5, 2.0, 0.5]), session_id="session_0", submap_id=0)
        graph.add_edge(0, 1, np.array([1.5, 0.0, 0.0]), np.eye(3) * 50.0)

        global_grid = manager.render_composite_global_grid([s0], graph)

        # Save session
        saved_dir = manager.save_session(
            map_name="test_session",
            submaps=[s0],
            pose_graph=graph,
            global_grid=global_grid,
        )
        assert os.path.exists(os.path.join(saved_dir, "session_state.json"))
        assert os.path.exists(os.path.join(saved_dir, "submaps.npz"))
        assert os.path.exists(os.path.join(saved_dir, "map.png"))
        assert os.path.exists(os.path.join(saved_dir, "map.yaml"))

        # Load session back
        submaps_loaded, graph_loaded, grid_loaded, _ = manager.load_session("test_session")

        assert len(submaps_loaded) == 1
        assert submaps_loaded[0].submap_id == 0
        assert submaps_loaded[0].session_id == "session_0"
        np.testing.assert_allclose(submaps_loaded[0].origin_pose, s0.origin_pose)
        np.testing.assert_allclose(submaps_loaded[0].grid.grid, s0.grid.grid)

        assert len(graph_loaded.nodes) == 2
        np.testing.assert_allclose(graph_loaded.nodes[0].pose, graph.nodes[0].pose)
        np.testing.assert_allclose(graph_loaded.nodes[1].pose, graph.nodes[1].pose)

        assert len(graph_loaded.edges) == 1
        np.testing.assert_allclose(graph_loaded.edges[0].relative_pose, graph.edges[0].relative_pose)

