#!/usr/bin/env python3
"""
Cyberpunk Desktop Radar HUD for YDLIDAR X4
Renders real-time 2D polar/Cartesian point clouds, radar sweep line,
target tracking brackets, and telemetry on a translucent desktop widget.
"""

import sys
import math
import time
import argparse
from threading import Thread

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QRadialGradient, QPolygonF
from PyQt6.QtCore import Qt, QTimer, QPointF, pyqtSignal, QObject

# Try importing ydlidar; fallback to simulation mode if not available or requested
try:
    import ydlidar
    HAS_YDLIDAR = True
except ImportError:
    HAS_YDLIDAR = False


class LidarWorker(QObject):
    """Background worker to continuously poll the LiDAR without blocking the GUI."""
    new_scan = pyqtSignal(list, float)  # points list [(x, y, range, angle_deg)], scan_hz
    status_msg = pyqtSignal(str)

    def __init__(self, port="/dev/ttyUSB0", baudrate=128000, simulate=False):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.simulate = simulate
        self.running = True
        self.laser = None

    def start_polling(self):
        thread = Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        if self.simulate or not HAS_YDLIDAR:
            self.status_msg.emit("SIMULATION MODE ACTIVE")
            self._run_simulation()
            return

        self.status_msg.emit(f"Connecting to {self.port} @ {self.baudrate}...")
        ydlidar.os_init()
        self.laser = ydlidar.CYdLidar()
        self.laser.setlidaropt(ydlidar.LidarPropSerialPort, self.port)
        self.laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, self.baudrate)
        self.laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        self.laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        self.laser.setlidaropt(ydlidar.LidarPropScanFrequency, 7.0)
        self.laser.setlidaropt(ydlidar.LidarPropSampleRate, 5)
        self.laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
        self.laser.setlidaropt(ydlidar.LidarPropIntenstiy, False)
        self.laser.setlidaropt(ydlidar.LidarPropSupportMotorDtrCtrl, True)
        self.laser.setlidaropt(ydlidar.LidarPropFixedResolution, False)
        self.laser.setlidaropt(ydlidar.LidarPropAutoReconnect, True)

        if not self.laser.initialize() or not self.laser.turnOn():
            self.status_msg.emit("Failed to initialize LiDAR. Switching to simulation.")
            self._run_simulation()
            return

        self.status_msg.emit("HARDWARE CONNECTED - STREAMING")
        scan = ydlidar.LaserScan()

        while self.running and ydlidar.os_isOk():
            if self.laser.doProcessSimple(scan):
                points = []
                for p in scan.points:
                    r = p.range
                    if 0.15 <= r <= 8.0:
                        deg = math.degrees(p.angle) % 360
                        rad = p.angle
                        x = r * math.sin(rad)
                        y = r * math.cos(rad)
                        points.append((x, y, r, deg))

                hz = 1.0 / scan.config.scan_time if scan.config.scan_time > 0 else 7.0
                self.new_scan.emit(points, hz)
            else:
                time.sleep(0.001)

        if self.laser:
            self.laser.turnOff()
            self.laser.disconnecting()

    def _run_simulation(self):
        """Generates synthetic room reflections and a moving target for testing UI."""
        sim_angle = 0.0
        while self.running:
            sim_angle += 0.05
            points = []
            # Synthesize room walls (box)
            for deg in range(0, 360, 2):
                rad = math.radians(deg)
                # Static outer room walls (~2.5m)
                base_r = 2.5 + 0.3 * math.sin(4 * rad)
                # Moving intruder target
                intruder_r = 1.2 + 0.4 * math.cos(sim_angle)
                intruder_deg = (math.degrees(sim_angle) * 1.5) % 360
                if abs((deg - intruder_deg + 180) % 360 - 180) < 12:
                    r = intruder_r + (math.sin(deg * 5) * 0.02)
                else:
                    r = base_r + (math.cos(deg * 7) * 0.04)

                x = r * math.sin(rad)
                y = r * math.cos(rad)
                points.append((x, y, r, deg))

            self.new_scan.emit(points, 7.2)
            time.sleep(0.07)  # ~14 FPS update rate for simulation

    def stop(self):
        self.running = False


class RadarHudWidget(QWidget):
    """Translucent Cyberpunk Radar HUD Widget with QPainter rendering."""

    def __init__(self, max_range=4.0, theme="cyan", borderless=True, stay_bottom=False):
        super().__init__()
        self.max_range = max_range  # Max range in meters displayed on radar
        self.theme = theme
        self.borderless = borderless
        self.stay_bottom = stay_bottom

        self.points = []
        self.scan_hz = 0.0
        self.status = "INITIALIZING..."
        self.closest_target = None  # (dist, angle, x, y)
        self.target_clusters = []
        self.sweep_angle = 0.0
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

        self._setup_window()
        self._init_palette()

        # Animation timer for radar sweep line (60 FPS)
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._update_sweep)
        self.anim_timer.start(16)

    def _setup_window(self):
        self.setWindowTitle("YDLIDAR Cyberpunk Radar HUD")
        self.resize(720, 720)
        self.setObjectName("lidar-radar-hud")

        flags = Qt.WindowType.Widget
        if self.borderless:
            flags |= Qt.WindowType.FramelessWindowHint
        if self.stay_bottom:
            flags |= Qt.WindowType.WindowStaysOnBottomHint
        self.setWindowFlags(flags)

        # Translucent background
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def _init_palette(self):
        """Configure Cyberpunk Neon Color Scheme."""
        if self.theme == "green":
            self.primary = QColor(0, 255, 128)
            self.dim_primary = QColor(0, 255, 128, 40)
            self.point_color = QColor(64, 255, 170, 220)
            self.alert_color = QColor(255, 50, 80)
        else:  # Default Cyan / Cyberpunk Blue
            self.primary = QColor(0, 220, 255)
            self.dim_primary = QColor(0, 220, 255, 45)
            self.point_color = QColor(80, 240, 255, 230)
            self.alert_color = QColor(255, 80, 80)

        self.bg_color = QColor(10, 15, 24, 215)
        self.grid_color = QColor(20, 45, 65, 120)
        self.text_color = QColor(180, 225, 250)

    def update_scan_data(self, points, hz):
        self.points = points
        self.scan_hz = hz
        self._process_clusters()

    def update_status(self, msg):
        self.status = msg
        self.update()

    def _update_sweep(self):
        # Rotate sweep beam ~1 rev per second
        self.sweep_angle = (self.sweep_angle + 6.0) % 360.0

        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = now

        self.update()

    def _process_clusters(self):
        """Identify closest targets and cluster nearby points."""
        if not self.points:
            self.closest_target = None
            self.target_clusters = []
            return

        sorted_pts = sorted(self.points, key=lambda p: p[2])
        closest = sorted_pts[0]
        self.closest_target = (closest[2], closest[3], closest[0], closest[1])

        # Cluster nearby points within 15 cm of closest point
        cluster = [p for p in sorted_pts if abs(p[2] - closest[2]) < 0.15]
        if len(cluster) >= 4:
            avg_x = sum(p[0] for p in cluster) / len(cluster)
            avg_y = sum(p[1] for p in cluster) / len(cluster)
            self.target_clusters = [(avg_x, avg_y, len(cluster))]
        else:
            self.target_clusters = []

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        center = QPointF(w / 2.0, h / 2.0)
        radius = min(w, h) / 2.0 - 30.0

        # 1. Background disc
        bg_brush = QBrush(self.bg_color)
        painter.setBrush(bg_brush)
        painter.setPen(QPen(self.dim_primary, 1.5))
        painter.drawEllipse(center, radius, radius)

        # 2. Concentric Range Rings
        rings = 4
        painter.setFont(QFont("Monospace", 8))
        for i in range(1, rings + 1):
            r_ring = radius * (i / rings)
            ring_dist = self.max_range * (i / rings)

            # Draw circle
            painter.setPen(QPen(self.dim_primary, 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, r_ring, r_ring)

            # Distance label
            painter.setPen(QPen(self.text_color))
            painter.drawText(int(center.x() + 5), int(center.y() - r_ring + 12), f"{ring_dist:.1f}m")

        # 3. Radial Azimuth Lines & Degrees
        spokes = 12
        for i in range(spokes):
            deg = i * (360 / spokes)
            rad = math.radians(deg)
            x_spoke = center.x() + radius * math.sin(rad)
            y_spoke = center.y() - radius * math.cos(rad)

            painter.setPen(QPen(self.dim_primary, 0.8))
            painter.drawLine(center, QPointF(x_spoke, y_spoke))

            # Angle text around periphery
            label_x = center.x() + (radius + 15) * math.sin(rad) - 10
            label_y = center.y() - (radius + 15) * math.cos(rad) + 4
            painter.setPen(QPen(self.dim_primary))
            painter.drawText(int(label_x), int(label_y), f"{int(deg)}°")

        # 4. Sweep Beam with Fading Phosphor Trail
        sweep_rad = math.radians(self.sweep_angle)
        sweep_x = center.x() + radius * math.sin(sweep_rad)
        sweep_y = center.y() - radius * math.cos(sweep_rad)

        # Glowing Sweep Sector
        sweep_poly = QPolygonF()
        sweep_poly.append(center)
        trail_segments = 16
        for step in range(trail_segments + 1):
            angle = self.sweep_angle - step * 1.5
            rad = math.radians(angle)
            sweep_poly.append(QPointF(center.x() + radius * math.sin(rad),
                                      center.y() - radius * math.cos(rad)))

        sweep_grad = QRadialGradient(center, radius)
        sweep_grad.setColorAt(0.0, QColor(self.primary.red(), self.primary.green(), self.primary.blue(), 10))
        sweep_grad.setColorAt(1.0, QColor(self.primary.red(), self.primary.green(), self.primary.blue(), 60))
        painter.setBrush(QBrush(sweep_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(sweep_poly)

        # Leading sweep line
        painter.setPen(QPen(self.primary, 2.0))
        painter.drawLine(center, QPointF(sweep_x, sweep_y))

        # 5. Point Cloud Rendering
        scale = radius / self.max_range
        painter.setPen(QPen(self.point_color, 2.8))
        for p in self.points:
            px = center.x() + p[0] * scale
            py = center.y() - p[1] * scale

            # Clip points outside the radius circle
            dx = px - center.x()
            dy = py - center.y()
            if (dx * dx + dy * dy) <= (radius * radius):
                painter.drawPoint(QPointF(px, py))

        # 6. Target Tracking Lock Box / Centroid Marker
        for cluster in self.target_clusters:
            cx = center.x() + cluster[0] * scale
            cy = center.y() - cluster[1] * scale

            # Target Lock Brackets
            box_size = 14
            painter.setPen(QPen(self.alert_color, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Corner brackets
            painter.drawLine(QPointF(cx - box_size, cy - box_size), QPointF(cx - box_size + 6, cy - box_size))
            painter.drawLine(QPointF(cx - box_size, cy - box_size), QPointF(cx - box_size, cy - box_size + 6))

            painter.drawLine(QPointF(cx + box_size, cy - box_size), QPointF(cx + box_size - 6, cy - box_size))
            painter.drawLine(QPointF(cx + box_size, cy - box_size), QPointF(cx + box_size, cy - box_size + 6))

            painter.drawLine(QPointF(cx - box_size, cy + box_size), QPointF(cx - box_size + 6, cy + box_size))
            painter.drawLine(QPointF(cx - box_size, cy + box_size), QPointF(cx - box_size, cy + box_size - 6))

            painter.drawLine(QPointF(cx + box_size, cy + box_size), QPointF(cx + box_size - 6, cy + box_size))
            painter.drawLine(QPointF(cx + box_size, cy + box_size), QPointF(cx + box_size, cy + box_size - 6))

            # Target Telemetry Tag
            painter.setFont(QFont("Monospace", 8, QFont.Weight.Bold))
            painter.setPen(QPen(self.alert_color))
            painter.drawText(int(cx + box_size + 4), int(cy - 2), "TRACKING LOCK")
            painter.drawText(int(cx + box_size + 4), int(cy + 10), f"PTS: {cluster[2]}")

        # 7. Telemetry & Status HUD Overlay
        self._draw_hud_telemetry(painter)

    def _draw_hud_telemetry(self, painter):
        painter.setFont(QFont("Monospace", 9, QFont.Weight.Bold))
        painter.setPen(QPen(self.primary))

        # Top-Left Telemetry Block
        lines = [
            f">> SYSTEM: {self.status}",
            f">> SCAN FREQ: {self.scan_hz:4.1f} Hz",
            f">> TOTAL PTS: {len(self.points):4d}",
            f">> DISPLAY FPS: {self.fps:2d}",
        ]

        if self.closest_target:
            dist, deg, _, _ = self.closest_target
            lines.append(f">> CLOSEST OBJ: {dist:4.2f}m @ {deg:05.1f}°")
        else:
            lines.append(f">> CLOSEST OBJ: NONE")

        x_start = 25
        y_start = 35
        for i, text in enumerate(lines):
            painter.drawText(x_start, y_start + (i * 16), text)

        # Crosshair center dot
        painter.setBrush(QBrush(self.primary))
        painter.drawEllipse(QPointF(self.width() / 2, self.height() / 2), 3, 3)

    def keyPressEvent(self, event):
        # Escape or Q quits the HUD
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()


def main():
    parser = argparse.ArgumentParser(description="Cyberpunk Desktop Radar HUD for YDLIDAR")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port (/dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=128000, help="Baud rate (default: 128000)")
    parser.add_argument("--range", type=float, default=4.5, help="Max radar display range in meters (default: 4.5)")
    parser.add_argument("--theme", choices=["cyan", "green"], default="cyan", help="Cyberpunk color theme")
    parser.add_argument("--simulate", action="store_true", help="Run with synthetic simulated room point cloud")
    parser.add_argument("--desktop", action="store_true", help="Pin window to desktop background layer")
    parser.add_argument("--windowed", action="store_true", help="Include window title bar/borders")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    hud = RadarHudWidget(
        max_range=args.range,
        theme=args.theme,
        borderless=not args.windowed,
        stay_bottom=args.desktop
    )

    worker = LidarWorker(port=args.port, baudrate=args.baud, simulate=args.simulate)
    worker.new_scan.connect(hud.update_scan_data)
    worker.status_msg.connect(hud.update_status)
    worker.start_polling()

    hud.show()

    ret = app.exec()
    worker.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()
