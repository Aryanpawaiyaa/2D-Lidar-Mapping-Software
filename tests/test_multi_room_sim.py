"""Automated Multi-Room Acceptance Test:
Room A -> Save -> Restart -> Relocalize in Room A -> Move to Room B -> Merge on Doorway Overlap.
"""

import os
import tempfile
import numpy as np
import pytest
import yaml

from driver.simulator import SimulatedEnvironment, SimulationLidar
from relocalization.session_merger import MappingStatus
from slam.system import LidarSlamSystem


def load_test_config(tmp_maps_dir: str) -> dict:
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["persistence"]["maps_dir"] = tmp_maps_dir
    # Ensure acceptance test runs quickly
    cfg["relocalization"]["consecutive_matches_needed"] = 2
    return cfg


def test_multi_room_acceptance_scenario():
    with tempfile.TemporaryDirectory() as tmp_maps_dir:
        config = load_test_config(tmp_maps_dir)
        env = SimulatedEnvironment()
        sim = SimulationLidar(env=env, num_beams=720, noise_std_m=0.01, dropout_rate=0.0)

        # -------------------------------------------------------------
        # STEP 1: Map Room A
        # -------------------------------------------------------------
        slam = LidarSlamSystem(config)
        # Start robot at (2.5, 2.5, 0.0) inside Room A
        sim.set_pose(2.5, 2.5, 0.0)
        sim.start()

        # Move robot across Room A
        waypoints_room_a = [
            (2.5, 2.5, 0.0),
            (2.8, 2.5, 0.0),
            (3.2, 2.6, 0.05),
            (3.6, 2.7, 0.05),
            (4.0, 2.5, 0.0),
            (3.5, 2.2, -0.05),
        ]

        for x, y, yaw in waypoints_room_a:
            sim.set_pose(x, y, yaw)
            scan = sim.get_scan()
            assert scan is not None
            res = slam.process_scan(scan)

        assert slam.keyframe_count >= 2
        assert len(slam.submaps) >= 1
        assert slam.submaps[0].num_points_mapped > 500

        # -------------------------------------------------------------
        # STEP 2: Save Room A
        # -------------------------------------------------------------
        map_path = slam.save_map("room_a_session")
        assert os.path.exists(os.path.join(map_path, "session_state.json"))
        assert os.path.exists(os.path.join(map_path, "map.png"))
        assert os.path.exists(os.path.join(map_path, "map.yaml"))

        num_keyframes_room_a = slam.keyframe_count
        room_a_submap_grid = slam.submaps[0].grid.grid.copy()

        # -------------------------------------------------------------
        # STEP 3: Restart program & Relocalize in Room A
        # -------------------------------------------------------------
        slam_restarted = LidarSlamSystem(config)
        slam_restarted.load_map("room_a_session")
        assert slam_restarted.session_manager.status == MappingStatus.STARTUP_LOCALIZING

        # Simulate robot at (3.5, 2.5, 0.0) inside Room A
        sim.set_pose(3.5, 2.5, 0.0)
        scan1 = sim.get_scan()
        scan2 = sim.get_scan()
        slam_restarted.process_scan(scan1)
        update_info = slam_restarted.process_scan(scan2)

        # Confirm Room A relocalized!
        assert slam_restarted.session_manager.status == MappingStatus.TRACKING_EXISTING
        # In SLAM frame (started at 2.5, 2.5), simulated robot at (3.5, 2.5) has expected pose (1.0, 0.0, 0.0)
        reloc_pose = slam_restarted.current_pose
        err_pos = np.hypot(reloc_pose[0] - 1.0, reloc_pose[1] - 0.0)
        assert err_pos < 0.15, f"Relocalized position error: {err_pos:.3f}m"

        # -------------------------------------------------------------
        # STEP 4: Move into unmapped Room B through the doorway
        # -------------------------------------------------------------
        # Waypoints leading through corridor (x=6 to 8) into Room B (x=8 to 14)
        corridor_and_room_b_path = [
            (4.5, 2.5, 0.0),   # approaching doorway
            (5.5, 2.5, 0.0),   # doorway threshold
            (6.5, 2.6, 0.0),   # inside corridor
            (7.5, 2.6, 0.0),   # exiting corridor into Room B
            (9.0, 3.0, 0.0),   # Room B center
            (10.5, 3.5, 0.1),  # exploring Room B
            (11.5, 3.5, 0.0),  # near Room B table
        ]

        for x, y, yaw in corridor_and_room_b_path:
            sim.set_pose(x, y, yaw)
            scan = sim.get_scan()
            assert scan is not None
            slam_restarted.process_scan(scan)

        # Confirm Room A remains intact
        # Keyframes from Room A should be preserved
        assert slam_restarted.keyframe_count > num_keyframes_room_a
        assert len(slam_restarted.submaps) >= 1

        # -------------------------------------------------------------
        # STEP 5: Verify Global Map Consistency vs Ground Truth Walls
        # -------------------------------------------------------------
        # Render composite global map
        global_grid = slam_restarted.map_manager.render_composite_global_grid(
            slam_restarted.submaps, slam_restarted.pose_graph
        )
        assert global_grid is not None
        global_grid.compute_distance_field()

        # Check Ground Truth Walls from SimulatedEnvironment (offset by SLAM origin at 2.5, 2.5):
        # Wall in Room A: West wall at x=0, y=2.5 -> SLAM (-2.5, 0.0)
        # Wall in Room A: North wall at x=3.0, y=5.0 -> SLAM (0.5, 2.5)
        # Wall in Room A: South wall at x=3.0, y=0.0 -> SLAM (0.5, -2.5)
        # Corridor wall at x=7.0, y=2.0 -> SLAM (4.5, -0.5)
        test_wall_pts = np.array([
            [-2.5, 0.0],  # Room A west wall
            [0.5, 2.5],   # Room A north wall
            [0.5, -2.5],  # Room A south wall
            [4.5, -0.5],  # Corridor south wall
        ], dtype=np.float32)

        dists, _, _, valid = global_grid.query_distance_and_gradient(test_wall_pts)
        mean_wall_err = float(np.mean(dists[valid]))
        print(f"\n[ACCEPTANCE TEST REPORT] Mean wall error vs Ground Truth: {mean_wall_err * 100:.2f} cm")
        assert mean_wall_err < 0.15, f"Map error vs ground truth too high: {mean_wall_err}m"
