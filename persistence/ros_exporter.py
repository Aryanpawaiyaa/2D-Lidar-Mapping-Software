"""ROS map_server compatible map exporter (YAML + PNG)."""

import io
import os
from typing import Dict, Tuple
import numpy as np
from PIL import Image
import yaml

from persistence.atomic_io import atomic_write_bytes, atomic_write_text
from slam.occupancy_grid import OccupancyGrid


class RosMapExporter:
    """Exports global occupancy grid to standard ROS map_server PNG + YAML."""

    @staticmethod
    def export(
        grid: OccupancyGrid,
        output_dir: str,
        map_name: str = "map",
        occupied_thresh: float = 0.65,
        free_thresh: float = 0.25,
    ) -> Tuple[str, str]:
        """Export grid as ROS map_server compatible <map_name>.png and <map_name>.yaml.
        
        Args:
            grid: OccupancyGrid to export.
            output_dir: Directory where files should be written.
            map_name: Base name for exported files.
            occupied_thresh: Probability threshold for occupied cells.
            free_thresh: Probability threshold for free cells.
            
        Returns:
            Tuple of (yaml_filepath, png_filepath).
        """
        os.makedirs(output_dir, exist_ok=True)
        yaml_path = os.path.join(output_dir, f"{map_name}.yaml")
        png_path = os.path.join(output_dir, f"{map_name}.png")

        # Probability map: shape (rows, cols)
        prob = grid.get_probability_map()

        # In ROS map convention:
        # 0 = occupied, 205 = unknown, 254 = free
        # Row 0 in ROS image corresponds to the bottom (Y_min),
        # whereas PIL row 0 corresponds to top. So flip upside down vertically:
        prob_flipped = np.flipud(prob)
        grid_flipped = np.flipud(grid.grid)

        img_arr = np.full(prob_flipped.shape, 205, dtype=np.uint8)  # default unknown
        # Known free cells (log-odds < 0 and prob < free_thresh)
        free_mask = (grid_flipped < -0.1) & (prob_flipped <= free_thresh)
        img_arr[free_mask] = 254

        # Known occupied cells (log-odds > 0 and prob >= occupied_thresh)
        occ_mask = (grid_flipped > 0.1) & (prob_flipped >= occupied_thresh)
        img_arr[occ_mask] = 0

        # Save PNG to memory bytes then atomic write
        img = Image.fromarray(img_arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        atomic_write_bytes(png_path, buf.getvalue())

        # Generate YAML metadata
        yaml_data: Dict = {
            "image": f"{map_name}.png",
            "resolution": float(grid.resolution),
            "origin": [float(grid.origin_x), float(grid.origin_y), 0.0],
            "negate": 0,
            "occupied_thresh": float(occupied_thresh),
            "free_thresh": float(free_thresh),
        }
        yaml_text = yaml.dump(yaml_data, sort_keys=False, default_flow_style=None)
        atomic_write_text(yaml_path, yaml_text)

        return yaml_path, png_path

