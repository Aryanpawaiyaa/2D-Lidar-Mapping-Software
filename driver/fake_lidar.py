"""Replay Lidar data source for deterministic offline testing."""

import json
import os
import time
from typing import List, Optional
import numpy as np

from driver.interface import LidarSource, Scan


class FakeLidar(LidarSource):
    """Replays previously recorded scans from .npz or .jsonl logs."""

    def __init__(self, log_path: str, realtime: bool = True, speedup: float = 1.0, loop: bool = False):
        """Initialize FakeLidar.
        
        Args:
            log_path: Path to .npz or .jsonl scan log.
            realtime: If True, sleep between scans to mimic real sensor rate.
            speedup: Playback speed multiplier (e.g. 2.0 = 2x speed).
            loop: If True, restart from first scan when reaching the end.
        """
        self.log_path = log_path
        self.realtime = realtime
        self.speedup = max(0.01, speedup)
        self.loop = loop
        self.scans: List[Scan] = []
        self._current_idx = 0
        self._is_started = False
        self._last_replay_time: Optional[float] = None
        self._last_scan_stamp: Optional[float] = None

        self._load_scans()

    def _load_scans(self) -> None:
        if not os.path.exists(self.log_path):
            raise FileNotFoundError(f"Scan log not found at: {self.log_path}")

        if self.log_path.endswith(".npz"):
            data = np.load(self.log_path)
            count = int(data["count"])
            timestamps = data["timestamps"]
            for i in range(count):
                angles = data[f"angles_{i}"]
                ranges = data[f"ranges_{i}"]
                intensities = data[f"intensities_{i}"] if f"intensities_{i}" in data else None
                self.scans.append(
                    Scan(
                        angles_rad=angles,
                        ranges_m=ranges,
                        intensities=intensities,
                        timestamp=float(timestamps[i]),
                    )
                )
        elif self.log_path.endswith(".jsonl"):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.scans.append(
                        Scan(
                            angles_rad=np.array(obj["angles"], dtype=np.float32),
                            ranges_m=np.array(obj["ranges"], dtype=np.float32),
                            intensities=np.array(obj["intensities"], dtype=np.float32) if obj.get("intensities") else None,
                            timestamp=float(obj.get("timestamp", 0.0)),
                        )
                    )
        else:
            raise ValueError(f"Unsupported log format: {self.log_path}. Must be .npz or .jsonl")

    def start(self) -> None:
        self._is_started = True
        self._current_idx = 0
        self._last_replay_time = time.time()
        self._last_scan_stamp = None

    def get_scan(self, timeout: float = 1.0) -> Optional[Scan]:
        if not self._is_started or len(self.scans) == 0:
            return None

        if self._current_idx >= len(self.scans):
            if self.loop:
                self._current_idx = 0
                self._last_scan_stamp = None
                self._last_replay_time = time.time()
            else:
                return None

        scan = self.scans[self._current_idx]
        self._current_idx += 1

        if self.realtime and self._last_scan_stamp is not None and self._last_replay_time is not None:
            time_delta = (scan.timestamp - self._last_scan_stamp) / self.speedup
            if 0 < time_delta < 2.0:
                elapsed = time.time() - self._last_replay_time
                sleep_needed = time_delta - elapsed
                if sleep_needed > 0:
                    time.sleep(sleep_needed)

        self._last_scan_stamp = scan.timestamp
        self._last_replay_time = time.time()
        return scan

    def stop(self) -> None:
        self._is_started = False

    def is_connected(self) -> bool:
        return self._is_started and (self._current_idx < len(self.scans) or self.loop)

    def describe(self) -> str:
        return f"FakeLidar(log='{os.path.basename(self.log_path)}', scans={len(self.scans)})"

