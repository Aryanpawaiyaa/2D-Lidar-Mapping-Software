"""Pure-Python direct serial protocol driver for YDLIDAR TG50 (ToF 2D Lidar).

Protocol verified against official YDLIDAR TG Series Development Manual:
- Baud: 512000, 8N1
- Header: 0x55AA (0xAA, 0x55 little-endian)
- CT: Package Type (0x01 = Zero packet / Start of revolution, 0x00 = Normal packet)
- LSN: Sample count in package
- FSA: Starting angle (degrees = (FSA >> 1) / 64.0)
- LSA: Ending angle (degrees = (LSA >> 1) / 64.0)
- CS: Check Code (XOR checksum)
- Si: 3 bytes per sample:
    Byte 0: Intensity (0-255)
    Byte 1-2: Distance in mm (little-endian uint16, direct ToF mm)
"""

import glob
import logging
import struct
import sys
import time
from typing import List, Optional, Tuple
import numpy as np
import serial

# Linux termios2 constants for non-standard baud rates (e.g. 512000)
TCGETS2 = 0x802C542A
TCSETS2 = 0x402C542B
BOTHER = 0o010000
CBAUD = 0o010017

from driver.interface import LidarSource, Scan

logger = logging.getLogger(__name__)


class TG50SerialDriver(LidarSource):
    """Direct serial communication driver for YDLIDAR TG50."""

    START_SIGN = b"\xAA\x55"  # 0x55AA in byte order: AA then 55
    CMD_STOP = b"\xA5\x65"
    CMD_SCAN = b"\xA5\x60"
    CMD_DEVICE_INFO = b"\xA5\x90"
    CMD_HEALTH = b"\xA5\x91"

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 512000,
        auto_detect: bool = True,
        range_min_m: float = 0.05,
        range_max_m: float = 50.0,
        reconnect_timeout_sec: float = 2.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.auto_detect = auto_detect
        self.range_min_m = range_min_m
        self.range_max_m = range_max_m
        self.reconnect_timeout_sec = reconnect_timeout_sec

        self._ser: Optional[serial.Serial] = None
        self._is_running = False
        self._points_accum_angles: List[float] = []
        self._points_accum_ranges: List[float] = []
        self._points_accum_intensities: List[float] = []
        self._last_fsa: float = 0.0

    @staticmethod
    def find_serial_ports() -> List[str]:
        """Detect potential serial ports for YDLIDAR on Linux/Windows/macOS."""
        if sys.platform.startswith("linux"):
            ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ydlidar*")
            return sorted(ports)
        elif sys.platform.startswith("darwin"):
            return sorted(glob.glob("/dev/cu.usbserial*"))
        elif sys.platform.startswith("win"):
            return [f"COM{i + 1}" for i in range(256)]
        return []

    def start(self) -> None:
        """Open port, initialize hardware, enable motor, and start streaming."""
        self._is_running = True
        self._connect()

    def _connect(self) -> bool:
        if self._ser is not None and self._ser.is_open:
            return True

        target_ports = [self.port]
        if self.auto_detect:
            detected = self.find_serial_ports()
            if self.port in detected:
                detected.remove(self.port)
            target_ports = [self.port] + detected

        for p in target_ports:
            try:
                logger.info(f"Attempting to open TG50 serial port {p} at {self.baudrate} baud...")
                try:
                    self._ser = serial.Serial(
                        port=p,
                        baudrate=self.baudrate,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=1.0,
                    )
                except Exception as ex_direct:
                    # On Linux, standard termios raises EINVAL for non-standard 512000 baud
                    if sys.platform.startswith("linux"):
                        import array, fcntl
                        logger.debug(f"Direct baudrate failed ({ex_direct}), applying termios2 BOTHER custom baudrate...")
                        self._ser = serial.Serial(
                            port=p,
                            baudrate=115200,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            timeout=1.0,
                        )
                        fd = self._ser.fileno()
                        buf = array.array("i", [0] * 11)
                        fcntl.ioctl(fd, TCGETS2, buf)
                        buf[2] &= ~CBAUD
                        buf[2] |= BOTHER
                        buf[9] = self.baudrate
                        buf[10] = self.baudrate
                        fcntl.ioctl(fd, TCSETS2, buf)
                    else:
                        raise ex_direct

                self.port = p
                logger.info(f"Successfully opened port: {self.port} at {self.baudrate} baud")

                # Motor control via DTR/RTS (DTR=True enables TG50 motor)
                self._ser.dtr = True
                self._ser.rts = False
                time.sleep(0.05)

                # Send stop command, flush input buffer, then send start scan command
                self._ser.write(self.CMD_STOP)
                time.sleep(0.05)
                self._ser.reset_input_buffer()

                self._ser.write(self.CMD_SCAN)
                time.sleep(0.05)

                # Read response header: 0xA5 0x5A ... (7 bytes device response)
                resp = self._ser.read(7)
                if len(resp) >= 2 and resp[0] == 0xA5 and resp[1] == 0x5A:
                    logger.info("TG50 scan command acknowledged by device.")
                else:
                    logger.warning(f"Device response: {resp.hex() if resp else 'empty'}, continuing...")

                return True
            except Exception as e:
                logger.debug(f"Failed opening port {p}: {e}")
                if self._ser is not None:
                    try:
                        self._ser.close()
                    except Exception:
                        pass
                    self._ser = None

        logger.error(f"Could not connect to TG50 on any port: {target_ports}")
        return False

    def get_scan(self, timeout: float = 1.0) -> Optional[Scan]:
        """Read packets until a complete 360-degree revolution is assembled."""
        if not self._is_running:
            return None

        start_time = time.time()
        while self._is_running and (time.time() - start_time) < timeout:
            if self._ser is None or not self._ser.is_open:
                if not self._connect():
                    time.sleep(self.reconnect_timeout_sec)
                    continue

            try:
                packet = self._read_packet()
                if packet is None:
                    continue

                is_zero_packet, angles, ranges, intensities = packet

                if is_zero_packet and len(self._points_accum_ranges) > 50:
                    # Assemble full completed scan
                    scan = self._finalize_scan()
                    # Add current packet points to next revolution accumulator
                    self._points_accum_angles.extend(angles)
                    self._points_accum_ranges.extend(ranges)
                    self._points_accum_intensities.extend(intensities)
                    return scan

                self._points_accum_angles.extend(angles)
                self._points_accum_ranges.extend(ranges)
                self._points_accum_intensities.extend(intensities)

            except (serial.SerialException, OSError) as e:
                logger.warning(f"Serial communication error: {e}. Reconnecting...")
                if self._ser is not None:
                    try:
                        self._ser.close()
                    except Exception:
                        pass
                    self._ser = None
                time.sleep(0.5)

        return None

    def _read_packet(self) -> Optional[Tuple[bool, List[float], List[float], List[float]]]:
        """Read and validate a single 0x55AA protocol packet."""
        assert self._ser is not None

        # Synchronize with 0x55AA header (bytes b'\xAA\x55')
        header_buf = self._ser.read(2)
        while len(header_buf) == 2 and header_buf != self.START_SIGN:
            header_buf = header_buf[1:] + self._ser.read(1)

        if header_buf != self.START_SIGN:
            return None

        # Read CT (1 byte) and LSN (1 byte)
        ct_lsn = self._ser.read(2)
        if len(ct_lsn) < 2:
            return None
        ct, lsn = ct_lsn[0], ct_lsn[1]
        is_zero_packet = (ct & 0x01) == 0x01

        # Read FSA (2 bytes), LSA (2 bytes), CS (2 bytes)
        angles_cs = self._ser.read(6)
        if len(angles_cs) < 6:
            return None
        fsa_raw, lsa_raw, cs_expected = struct.unpack("<HHH", angles_cs)

        start_angle_deg = (fsa_raw >> 1) / 64.0
        end_angle_deg = (lsa_raw >> 1) / 64.0

        is_zero_packet = ((ct & 0x01) == 1) and (lsn <= 2)
        self._last_fsa = start_angle_deg

        # Read sample points: For TG series ToF, each sample point is 3 bytes (intensity + 2 bytes dist)
        bytes_to_read = lsn * 3
        sample_bytes = self._ser.read(bytes_to_read)
        if len(sample_bytes) < bytes_to_read:
            return None

        # Verify XOR Checksum (Official YDLidar SDK calcCheckSum formula for 3-byte packets)
        cs_calc = 0x55AA ^ (ct | (lsn << 8)) ^ fsa_raw ^ lsa_raw
        for i in range(lsn):
            cs_calc ^= sample_bytes[i * 3]
            dist_u16 = sample_bytes[i * 3 + 1] | (sample_bytes[i * 3 + 2] << 8)
            cs_calc ^= dist_u16

        # If checksum mismatch, discard corrupted packet
        if cs_calc != cs_expected:
            logger.debug(f"Checksum mismatch: calc=0x{cs_calc:04X}, expected=0x{cs_expected:04X}")
            return None

        # Parse sample points
        angles: List[float] = []
        ranges: List[float] = []
        intensities: List[float] = []

        diff_deg = (end_angle_deg - start_angle_deg) % 360.0
        step_deg = diff_deg / max(1, lsn - 1) if lsn > 1 else 0.0

        for i in range(lsn):
            offset = i * 3
            intensity = float(sample_bytes[offset])
            dist_mm = sample_bytes[offset + 1] | (sample_bytes[offset + 2] << 8)
            dist_m = dist_mm / 1000.0

            angle_deg = (start_angle_deg + step_deg * i) % 360.0
            # Convert to radians in [-pi, pi) CCW convention
            angle_rad = np.radians(angle_deg)
            # Wrap to [-pi, pi)
            angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi

            angles.append(float(angle_rad))
            ranges.append(float(dist_m))
            intensities.append(float(intensity))

        return is_zero_packet, angles, ranges, intensities

    def _finalize_scan(self) -> Scan:
        """Pack accumulated points into a Scan object sorted by angle."""
        angles = np.array(self._points_accum_angles, dtype=np.float32)
        ranges = np.array(self._points_accum_ranges, dtype=np.float32)
        intensities = np.array(self._points_accum_intensities, dtype=np.float32)

        # Sort by angle for consistent ordering
        sort_idx = np.argsort(angles)
        angles = angles[sort_idx]
        ranges = ranges[sort_idx]
        intensities = intensities[sort_idx]

        self._points_accum_angles.clear()
        self._points_accum_ranges.clear()
        self._points_accum_intensities.clear()

        return Scan(
            angles_rad=angles,
            ranges_m=ranges,
            intensities=intensities,
            timestamp=time.time(),
        )

    def stop(self) -> None:
        """Stop scanning and shut down serial port."""
        self._is_running = False
        if self._ser is not None:
            try:
                self._ser.write(self.CMD_STOP)
                time.sleep(0.05)
                self._ser.dtr = False
                self._ser.rts = False
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        logger.info("TG50SerialDriver stopped cleanly.")

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def describe(self) -> str:
        status = "connected" if self.is_connected() else "disconnected"
        return f"TG50SerialDriver(port='{self.port}', baud={self.baudrate}, status={status})"

