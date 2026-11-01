"""Unit tests for LikelihoodFieldScanMatcher accuracy and PoseGraph optimization."""

import math
import numpy as np
import pytest

from driver.simulator import SimulatedEnvironment, SimulationLidar
from slam.occupancy_grid import OccupancyGrid
from slam.scan_matcher import LikelihoodFieldScanMatcher
from slam.pose_graph import PoseGraph


def test_scan_matcher_ground_truth_accuracy():
    # Build a simulated environment and populate an initial map
    env = SimulatedEnvironment()
    sim = SimulationLidar(env=env, num_beams=720, noise_std_m=0.0, dropout_rate=0.0)

    # Robot at (3.0, 2.5, 0.0)
    sim.set_pose(3.0, 2.5, 0.0)
    sim.start()
    scan_init = sim.get_scan()
    assert scan_init is not None

    pts_init_cart = scan_init.to_cartesian()

    # Populate occupancy grid at (3.0, 2.5, 0.0)
    grid = OccupancyGrid(resolution_m=0.05, width_m=12.0, height_m=12.0, origin_x_m=-3.0, origin_y_m=-3.5)
    # Convert points to world coordinates using initial pose (3.0, 2.5, 0.0)
    pts_init_world = pts_init_cart + np.array([3.0, 2.5])
    grid.update_scan(sensor_x=3.0, sensor_y=2.5, hit_points_world=pts_init_world)
    grid.compute_distance_field()

    # Move simulated robot by known delta: dx=+0.12m, dy=-0.08m, dyaw=+0.05rad (~2.9 deg)
    true_pose = np.array([3.12, 2.42, 0.05], dtype=np.float64)
    sim.set_pose(true_pose[0], true_pose[1], true_pose[2])
    scan_moved = sim.get_scan()
    assert scan_moved is not None
    pts_moved_cart = scan_moved.to_cartesian()

    matcher = LikelihoodFieldScanMatcher(max_iterations=30, huber_delta_m=0.10)
    # Provide an initial guess with deliberate error: (3.0, 2.5, 0.0)
    guess_pose = np.array([3.0, 2.5, 0.0], dtype=np.float64)

    result = matcher.match(grid, pts_moved_cart, guess_pose, use_coarse_search=True)

    assert result.converged
    assert result.confidence_score > 0.85

    # Check recovered pose error vs ground truth
    err_x = abs(result.pose[0] - true_pose[0])
    err_y = abs(result.pose[1] - true_pose[1])
    err_yaw = abs(result.pose[2] - true_pose[2])

    assert err_x < 0.05, f"X error too high: {err_x}"
    assert err_y < 0.05, f"Y error too high: {err_y}"
    assert err_yaw < 0.02, f"Yaw error too high: {err_yaw}"


def test_pose_graph_optimization_loop_closure():
    # Simulate a 4-node square loop:
    # Node 0: (0, 0, 0)
    # Node 1: (2, 0, 0)
    # Node 2: (2, 2, pi/2)
    # Node 3: (0, 2, pi)
    # And close loop from Node 3 back to Node 0
    graph = PoseGraph()

    # Ground truth poses
    gt_0 = np.array([0.0, 0.0, 0.0])
    gt_1 = np.array([2.0, 0.0, 0.0])
    gt_2 = np.array([2.0, 2.0, np.pi / 2])
    gt_3 = np.array([0.0, 2.0, np.pi])

    # Introduce drifting dead-reckoning initial estimates
    graph.add_node(0, gt_0)
    graph.add_node(1, gt_1 + [0.05, 0.02, 0.01])
    graph.add_node(2, gt_2 + [0.10, 0.06, 0.03])
    graph.add_node(3, gt_3 + [0.15, 0.12, 0.05])

    # Odometry edges with measurements
    # 0 -> 1: dx=2.0, dy=0, dth=0
    graph.add_edge(0, 1, np.array([2.0, 0.0, 0.0]))
    # 1 -> 2: in frame 1, move y=2.0, turn pi/2
    graph.add_edge(1, 2, np.array([0.0, 2.0, np.pi / 2]))
    # 2 -> 3: in frame 2 (facing +y), move x=2.0 (which is world -x), turn pi/2
    # R2 = [[0, -1], [1, 0]], p3 - p2 = [-2, 0], R2^T [-2, 0] = [0, 2]
    graph.add_edge(2, 3, np.array([0.0, 2.0, np.pi / 2]))

    # Loop closure edge from 3 back to 0:
    # R3 = [[-1, 0], [0, -1]], p0 - p3 = [0, -2], R3^T [0, -2] = [0, 2], turn pi
    graph.add_edge(3, 0, np.array([0.0, 2.0, np.pi]), is_loop_closure=True)

    chi2 = graph.optimize(max_iterations=20, fixed_node_id=0)

    # After optimization, nodes should be very close to ground truth
    for nid, gt in [(0, gt_0), (1, gt_1), (2, gt_2), (3, gt_3)]:
        node_pose = graph.nodes[nid].pose
        np.testing.assert_allclose(node_pose[:2], gt[:2], atol=0.03)
        # Proper SO(2) angle wrapping difference
        dth = (node_pose[2] - gt[2] + np.pi) % (2 * np.pi) - np.pi
        assert abs(dth) < 0.03, f"Node {nid} angle difference {dth} exceeds tolerance"
