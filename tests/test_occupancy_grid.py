"""Unit tests for OccupancyGrid log-odds updates and Distance Transform."""

import numpy as np
import pytest

from slam.occupancy_grid import OccupancyGrid


def test_grid_coordinate_transforms():
    grid = OccupancyGrid(resolution_m=0.1, width_m=10.0, height_m=10.0, origin_x_m=-5.0, origin_y_m=-5.0)

    # Origin center (-5.0, -5.0) should be around grid cell (0, 0)
    col, row = grid.world_to_grid(-4.95, -4.95)
    assert col == 0 and row == 0

    # Center (0.0, 0.0) should be around (50, 50)
    col, row = grid.world_to_grid(0.0, 0.0)
    assert col == 50 and row == 50

    # Round trip
    wx, wy = grid.grid_to_world(col, row)
    assert pytest.approx(wx, abs=0.1) == 0.0
    assert pytest.approx(wy, abs=0.1) == 0.0


def test_occupancy_raycast_updates():
    grid = OccupancyGrid(
        resolution_m=0.05,
        width_m=4.0,
        height_m=4.0,
        origin_x_m=-2.0,
        origin_y_m=-2.0,
        p_occ=0.75,
        p_free=0.35,
    )

    # Sensor at (0, 0), obstacle hit at (1.0, 0.0)
    hits = np.array([[1.0, 0.0]], dtype=np.float32)
    grid.update_scan(sensor_x=0.0, sensor_y=0.0, hit_points_world=hits)

    # Endpoint (1.0, 0.0) should have positive log-odds (occupied)
    c_end, r_end = grid.world_to_grid(1.0, 0.0)
    assert grid.grid[r_end, c_end] > 0.0

    # Midpoint (0.5, 0.0) along ray should have negative log-odds (free)
    c_mid, r_mid = grid.world_to_grid(0.5, 0.0)
    assert grid.grid[r_mid, c_mid] < 0.0

    # Probability map check
    prob = grid.get_probability_map()
    assert prob[r_end, c_end] > 0.60
    assert prob[r_mid, c_mid] < 0.40


def test_distance_field_computation():
    grid = OccupancyGrid(resolution_m=0.1, width_m=4.0, height_m=4.0, origin_x_m=-2.0, origin_y_m=-2.0)

    # Mark obstacle at (0.0, 0.0)
    c, r = grid.world_to_grid(0.0, 0.0)
    grid.grid[r, c] = grid.log_odds_thresh_occ + 1.0
    grid.compute_distance_field()

    edt = grid.get_distance_field()
    # Distance at obstacle cell should be 0.0
    assert pytest.approx(edt[r, c], abs=1e-5) == 0.0

    # Query distance at (0.5, 0.0)
    query_pts = np.array([[0.5, 0.0]], dtype=np.float32)
    dists, gx, gy, valid = grid.query_distance_and_gradient(query_pts)
    assert valid[0]
    # Distance should be approx 0.5 meters
    assert pytest.approx(dists[0], abs=0.08) == 0.5
    # Gradient in x should be positive (distance increases towards +x)
    assert gx[0] > 0.0

