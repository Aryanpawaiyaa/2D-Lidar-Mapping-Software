"""Scan recording utilities for logging scans deterministically."""

import json
import os
import time
from typing import List, Optional
import numpy as np

from driver.interface import Scan


class ScanRecorder:
    """Records raw 2D lidar scans to disk (.npz or .jsonl)."""

    def __init__(self, output_path: str, format: str = "npz"):
        """Initialize recorder.
        
        Args:
            output_path: Destination file path.
            format: 'npz' (recommended, binary compressed) or 'jsonl'.
        """
        self.output_path = output_path
        self.format = format.lower()
        self.scans: List[Scan] = []
        self._jsonl_file = None
        self._is_recording = False
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if self.format == "jsonl":
            self._jsonl_file = open(output_path, "w", encoding="utf-8")

    def start(self) -> None:
        self._is_recording = True

    def record_scan(self, scan: Scan) -> None:
        """Add a scan to the record."""
        if not self._is_recording:
            return

        if self.format == "npz":
            self.scans.append(scan)
        elif self.format == "jsonl" and self._jsonl_file is not None:
            entry = {
                "timestamp": scan.timestamp,
                "angles": scan.angles_rad.tolist(),
                "ranges": scan.ranges_m.tolist(),
                "intensities": scan.intensities.tolist() if scan.intensities is not None else None,
            }
            self._jsonl_file.write(json.dumps(entry) + "\n")
            self._jsonl_file.flush()

    def close(self) -> None:
        """Finalize and flush recorded data."""
        self._is_recording = False
        if self.format == "npz" and len(self.scans) > 0:
            timestamps = np.array([s.timestamp for s in self.scans], dtype=np.float64)
            # Find max point length to pad arrays cleanly or store ragged arrays
            angles_list = [s.angles_rad for s in self.scans]
            ranges_list = [s.ranges_m for s in self.scans]
            has_intensity = any(s.intensities is not None for s in self.scans)
            
            data_dict = {
                "timestamps": timestamps,
                "count": len(self.scans),
            }
            for i, (a, r) in enumerate(zip(angles_list, ranges_list)):
                data_dict[f"angles_{i}"] = a
                data_dict[f"ranges_{i}"] = r
                if has_intensity and self.scans[i].intensities is not None:
                    data_dict[f"intensities_{i}"] = self.scans[i].intensities

            np.savez_compressed(self.output_path, **data_dict)

        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None

    def __len__(self) -> int:
        return len(self.scans)

