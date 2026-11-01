"""Unit tests for Scan data structures, Cartesian conversion, Simulator, and Recorder."""

import os
import tempfile
import numpy as np
import pytest

from driver.interface import Scan
from driver.simulator import SimulationLidar, SimulatedEnvironment
from driver.fake_lidar import FakeLidar
from recorder.scan_recorder import ScanRecorder


def test_scan_cartesian_conversion():
    # Angle 0 should be (+r, 0)
    # Angle pi/2 should be (0, +r)
    # Angle pi should be (-r, 0)
    # Angle -pi/2 should be (0, -r)
    angles = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float32)
    ranges = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    scan = Scan(angles_rad=angles, ranges_m=ranges, timestamp=100.0)

    pts = scan.to_cartesian(min_range=0.1, max_range=10.0)
    assert pts.shape == (4, 2)

    np.testing.assert_allclose(pts[0], [2.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(pts[1], [0.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(pts[2], [-4.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(pts[3], [0.0, -5.0], atol=1e-5)


def test_scan_range_filtering():
    angles = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    # 0.01 is below min_range (0.05), 60.0 is above max_range (50.0)
    ranges = np.array([0.01, 2.5, 60.0], dtype=np.float32)
    scan = Scan(angles_rad=angles, ranges_m=ranges)

    pts = scan.to_cartesian(min_range=0.05, max_range=50.0)
    assert len(pts) == 1
    np.testing.assert_allclose(pts[0], [2.5 * np.cos(0.5), 2.5 * np.sin(0.5)], atol=1e-5)


def test_simulator_raycasting():
    env = SimulatedEnvironment()
    sim = SimulationLidar(env=env, num_beams=360, noise_std_m=0.0, dropout_rate=0.0)
    # Place robot at (3.0, 2.5, 0.0) in Room A
    sim.set_pose(3.0, 2.5, 0.0)
    sim.start()
    scan = sim.get_scan()
    assert scan is not None
    assert len(scan.ranges_m) == 360

    # Beam at 0 deg points along +X: robot at x=3.0, east wall has doorway/wall at x=6.0 -> distance ~ 3.0m
    # Find beam closest to 0 rad
    idx_0 = np.argmin(np.abs(scan.angles_rad - 0.0))
    # Note: east wall at x=6 has doorway between y=2.0 and y=3.2.
    # From (3.0, 2.5), ray at angle 0 goes through the doorway!
    # And reaches Room B east wall at x=14.0 or table at x=10.5
    assert scan.ranges_m[idx_0] > 3.0

    # Beam at pi/2 points north along +Y: robot at y=2.5, north wall at y=5.0 -> distance ~ 2.5m
    idx_north = np.argmin(np.abs(scan.angles_rad - np.pi / 2))
    assert pytest.approx(scan.ranges_m[idx_north], abs=0.05) == 2.5

    # Beam at -pi/2 points south along -Y: robot at y=2.5, south wall at y=0.0 -> distance ~ 2.5m
    idx_south = np.argmin(np.abs(scan.angles_rad - (-np.pi / 2)))
    assert pytest.approx(scan.ranges_m[idx_south], abs=0.05) == 2.5


def test_recorder_and_fake_lidar_npz_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test_scans.npz")
        recorder = ScanRecorder(npz_path, format="npz")
        recorder.start()

        s1 = Scan(
            angles_rad=np.linspace(-np.pi, np.pi, 100, dtype=np.float32),
            ranges_m=np.ones(100, dtype=np.float32) * 2.5,
            intensities=np.ones(100, dtype=np.float32) * 150.0,
            timestamp=1000.0,
        )
        s2 = Scan(
            angles_rad=np.linspace(-np.pi, np.pi, 100, dtype=np.float32),
            ranges_m=np.ones(100, dtype=np.float32) * 3.5,
            intensities=np.ones(100, dtype=np.float32) * 160.0,
            timestamp=1000.1,
        )
        recorder.record_scan(s1)
        recorder.record_scan(s2)
        recorder.close()

        assert os.path.exists(npz_path)

        fake = FakeLidar(npz_path, realtime=False)
        fake.start()
        r1 = fake.get_scan()
        r2 = fake.get_scan()
        r3 = fake.get_scan()

        assert r1 is not None
        assert r2 is not None
        assert r3 is None  # EOF

        np.testing.assert_allclose(r1.ranges_m, s1.ranges_m)
        np.testing.assert_allclose(r1.angles_rad, s1.angles_rad)
        np.testing.assert_allclose(r1.intensities, s1.intensities)
        assert r1.timestamp == 1000.0

        np.testing.assert_allclose(r2.ranges_m, s2.ranges_m)
        assert r2.timestamp == 1000.1

