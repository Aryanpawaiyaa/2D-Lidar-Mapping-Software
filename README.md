# YDLIDAR X4 Linux Driver & Spatial Touch Interface

A complete, field-tested guide and codebase for setting up, compiling, and running the **YDLIDAR X4** (and compatible triangulation LiDARs) on **Ubuntu / Xubuntu** (including modern releases such as Ubuntu 24.04 and 26.04 LTS).

This project includes driver setup instructions, native C++ tools, Python bindings, and a **Spatial Touch Interface** (`spatial_mouse.py`) that transforms any flat desk or wall into a multi-axis digital trackpad using Linux `uinput`.

---

## Table of Contents

- [Overview & Capabilities](#overview--capabilities)
- [System Architecture](#system-architecture)
  - [Hardware Architecture](#hardware-architecture)
  - [Software Stack](#software-stack)
- [Data Pipeline & Mathematical Logic](#data-pipeline--mathematical-logic)
- [Prerequisites & Dependencies](#prerequisites--dependencies)
- [Step 1: Install System Compilers & Tools](#step-1-install-system-compilers--tools)
- [Step 2: Clone & Prepare YDLidar-SDK](#step-2-clone--prepare-ydlidar-sdk)
- [Step 3: CMake 4.x / Modern Toolchain Compatibility](#step-3-cmake-4x--modern-toolchain-compatibility)
- [Step 4: Configure udev Rules (USB Permissions)](#step-4-configure-udev-rules-usb-permissions)
- [Step 5: Build & Install Core C++ SDK](#step-5-build--install-core-c-sdk)
- [Step 6: Build & Install Python Wrapper](#step-6-build--install-python-wrapper)
- [Step 7: Configure Virtual Mouse (`uinput`)](#step-7-configure-virtual-mouse-uinput)
- [Hardware Verification (C++ Test Tool)](#hardware-verification-c-test-tool)
- [Python Applications](#python-applications)
  - [1. The Transparent Desktop Radar HUD (`radar_hud.py`)](#1-the-transparent-desktop-radar-hud-radar_hudpy)
  - [2. The Invisible Geofence & Auto-Lock Security (`geofence_security.py`)](#2-the-invisible-geofence--auto-lock-security-geofence_securitypy)
  - [3. Spatial Touch Interface (`spatial_mouse.py`)](#3-spatial-touch-interface-spatial_mousepy)
  - [4. Diagnostic Scanners (`lidar_touch.py` & `test_lidar.py`)](#4-diagnostic-scanners-lidar_touchpy--test_lidarpy)
- [Window Manager Integration (Hyprland & Xfce)](#window-manager-integration-hyprland--xfce)
- [Physical Calibration Guide](#physical-calibration-guide)
- [Troubleshooting & Critical Pitfalls](#troubleshooting--critical-pitfalls)
- [License](#license)

---

## Overview & Capabilities

The **YDLIDAR X4** is a 360-degree 2D laser range finder (LiDAR) based on triangulation. It operates using standard UART communication via a Silicon Labs CP210x USB bridge.

- **Baud Rate**: `128000`
- **Scanning Frequency**: `5 Hz – 12 Hz` (default `7 Hz`)
- **Sampling Frequency**: `5000 Hz` (5K)
- **Communication Type**: **One-Way (Single-Channel)** telemetry
- **Motor Control**: DTR-triggered via CP2102
- **Effective Detection Range**: `0.23m – 10.0m` (Blindspot: `< 0.23m`)

The included **Spatial Touch Interface** translates hand movements in mid-air directly into absolute X/Y mouse coordinates across Xubuntu / Ubuntu, enabling system-wide touch-free cursor control.

---

## System Architecture

```
+-------------------------------------------------------------+
|                     Physical Environment                    |
|       [Hand Movement in Frontal Interaction Cone]           |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   YDLIDAR X4 Hardware                       |
|   - Rotating Optical Head (360° Triangulation @ 7 Hz)        |
|   - Silicon Labs CP210x USB-to-UART Serial Bridge           |
|   - Dual-rail Power: Logic USB + Auxiliary 5V Motor PWR      |
+-------------------------------------------------------------+
                              | Serial Data (/dev/ttyUSB0 @ 128k)
                              v
+-------------------------------------------------------------+
|                     Linux Kernel Layer                      |
|   - cp210x driver -> /dev/ttyUSB0                           |
|   - udev rule -> persistent permissions (0666)              |
|   - uinput module -> /dev/uinput virtual input device       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   Core C++ YDLIDAR SDK                      |
|   - libydlidar_sdk (Serial framing, packet decoding)         |
|   - SWIG Python Wrapper (_ydlidar.so / ydlidar.py)          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|         Spatial Mouse Application (spatial_mouse.py)         |
|   1. Spatial Filtering: Angle & range bounding box          |
|   2. Centroid Clustering: Hand point cluster averaging      |
|   3. Polar to Cartesian Coordinate Conversion               |
|   4. Screen Dimension Normalization (1920x1080)             |
|   5. Exponential Moving Average (EMA) Smoothing             |
|   6. Linux evdev / uinput Virtual Absolute Mouse Injection   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   X11 / Wayland Desktop                     |
|           (Smooth, Real-time System Cursor Motion)          |
+-------------------------------------------------------------+
```

### Hardware Architecture
- **Sensor**: The YDLIDAR X4 continuously rotates its optical head, firing a laser to measure the triangulation of reflections. It captures ~5,000 distance samples per second (~700 points per 360° rotation at 7 Hz).
- **Data Transport**: The sensor connects to the PC via a USB-to-TTL UART adapter. Data streams serially at high speed (`128,000 bps`) over TX/RX pins.
- **Power Delivery**: The LiDAR's physical motor draws significant surge current during spin-up. The motor enable (`M_EN`) and `VCC` lines should be powered by a stable 5V external source to prevent USB brownouts.

### Software Stack
- **Core Driver (YDLIDAR SDK)**: A compiled C++ library handles low-level serial handshakes, decodes raw hexadecimal packets into distance/angle floats, and manages motor state.
- **Python Interface (SWIG)**: The C++ SDK is wrapped into Python via SWIG, allowing rapid scripting without sacrificing execution speed.
- **OS Integration (`evdev` & `uinput`)**: The Linux `uinput` kernel module allows user-space applications to instantiate virtual input peripherals. The Python `evdev` library writes absolute positioning coordinates (`ABS_X`, `ABS_Y`) treated by the OS identically to a physical touch digitizer.

---

## Data Pipeline & Mathematical Logic

The translation from raw laser scans to cursor motion follows a deterministic mathematical pipeline:

1. **Spatial Filtering (Bounding Box)**:
   Any point outside the active physical zone is discarded:
   - Range limits: $0.24\text{ m} \le r \le 0.75\text{ m}$ (avoiding the optical blindspot below 23 cm)
   - Angle limits: $-50^\circ \le \theta \le +50^\circ$ (frontal interaction cone)

2. **Centroid Clustering**:
   A human hand reflects dozens of discrete laser points. Tracking only the single closest point causes severe jitter as reflections bounce off varying knuckles.
   The algorithm clusters all points within $8\text{ cm}$ of the closest object:
   $$\text{Cluster} = \{ p_i \mid r_i \le \min(r) + 0.08\text{ m} \}$$

3. **Coordinate Conversion**:
   Polar coordinates $(r_i, \theta_i)$ are transformed to Cartesian coordinates $(x, y)$:
   $$x = \frac{1}{N} \sum_{i=1}^N r_i \sin(\theta_i)$$
   $$y = \frac{1}{N} \sum_{i=1}^N r_i \cos(\theta_i)$$

4. **Resolution Mapping**:
   The physical bounding box ($x \in [-X_{\text{bounds}}, +X_{\text{bounds}}]$, $y \in [Y_{\min}, Y_{\max}]$) is normalized to $[0.0, 1.0]$ and scaled to display pixels:
   $$\text{target}_x = \left[\frac{x + X_{\text{bounds}}}{2 \cdot X_{\text{bounds}}}\right] \cdot W_{\text{screen}}$$
   $$\text{target}_y = \left[\frac{y - Y_{\min}}{Y_{\max} - Y_{\min}}\right] \cdot H_{\text{screen}}$$

5. **EMA Smoothing Filter**:
   To eliminate high-frequency hand tremors while maintaining low latency, an Exponential Moving Average (EMA) filter is applied:
   $$x_t^{\text{smooth}} = \alpha \cdot \text{target}_x + (1 - \alpha) \cdot x_{t-1}^{\text{smooth}}$$
   $$y_t^{\text{smooth}} = \alpha \cdot \text{target}_y + (1 - \alpha) \cdot y_{t-1}^{\text{smooth}}$$
   *(Default $\alpha = 0.35$ provides ideal responsiveness and stability).*

6. **Virtual Injection**:
   The smoothed coordinates are emitted as `EV_ABS` events via `uinput.syn()`.

---

## Prerequisites & Dependencies

- **OS**: Ubuntu / Xubuntu (20.04, 22.04, 24.04, 26.04 LTS)
- **Compilers**: GCC / G++ (v9+)
- **Build System**: CMake (v3.5+) & GNU Make
- **Python**: Python 3.8+ with `python3-dev`, `python3-setuptools`, `python3-evdev`, and `swig` (v4.x)

---

## Step 1: Install System Compilers & Tools

```bash
sudo apt update
sudo apt install -y cmake pkg-config build-essential git swig python3-dev python3-pip python3-setuptools python3-evdev
```

Verify installed versions:

```bash
cmake --version
swig -version
python3 --version
```

---

## Step 2: Clone & Prepare YDLidar-SDK

```bash
cd ~
git clone https://github.com/YDLIDAR/YDLidar-SDK.git
cd YDLidar-SDK
```

---

## Step 3: CMake 4.x / Modern Toolchain Compatibility

> [!NOTE]
> On modern Linux distributions with CMake $\ge$ 3.28 or 4.x, deprecated CMake 2.8 policies (`CMP0026`, `CMP0037`, `CMP0043`, `CMP0053`) are rejected.

If building on CMake 4+, apply these fixes:
1. In `CMakeLists.txt`, set `cmake_minimum_required(VERSION 3.5)`.
2. Remove obsolete `cmake_policy(SET CMP0053/CMP0037/CMP0043 OLD)` calls.
3. In `cmake/install_package.cmake`, replace target location reading with:
   ```cmake
   install(TARGETS ${PACKAGE_LIB_NAME} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
   ```

---

## Step 4: Configure udev Rules (USB Permissions)

Install persistent USB permissions so your user account can communicate with `/dev/ttyUSB0` without requiring `sudo`:

```bash
cd ~/YDLidar-SDK
chmod 777 startup/*
sudo sh startup/initenv.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Ensure your user is in the `dialout` group:
```bash
sudo usermod -aG dialout $USER
```

---

## Step 5: Build & Install Core C++ SDK

```bash
cd ~/YDLidar-SDK
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

---

## Step 6: Build & Install Python Wrapper

Compile SWIG bindings and install into your system's Python packages:

```bash
cd ~/YDLidar-SDK
python3 setup.py build
sudo python3 setup.py install
```

Verify import:
```bash
python3 -c "import ydlidar; print('YDLIDAR SDK imported successfully!')"
```

---

## Step 7: Configure Virtual Mouse (`uinput`)

Enable the Linux `uinput` kernel module and set read/write permissions:

```bash
sudo modprobe uinput
sudo chmod 666 /dev/uinput
```

To persist across reboots, add `uinput` to modules and create a udev rule:
```bash
echo "uinput" | sudo tee /etc/modules-load.d/uinput.conf
echo 'KERNEL=="uinput", MODE="0666"' | sudo tee /etc/udev/rules.d/99-uinput.rules
```

---

## Hardware Verification (C++ Test Tool)

Verify raw sensor readings using the compiled C++ test binary:

```bash
cd ~/YDLidar-SDK/build
./tri_test
```
*(Aliased to `./ydlidar_test` if symlinked).*

**Prompt Settings:**
- **Port**: Select the index for `/dev/ttyUSB0` (or `ydlidar4`).
- **Baudrate**: `128000` (Option `1`).
- **Single-Channel / One-Way**: `y`.
- **Scan Frequency**: `7`.

---

## Python Applications

### 1. The Transparent Desktop Radar HUD (`radar_hud.py`)

A cyberpunk-themed, real-time 60 FPS radar widget built with PyQt6. It plots 2D point clouds, azimuth spokes, concentric range rings, dynamic rotating sweep lines with phosphor fade trails, and automated centroid tracking brackets.

```bash
cd ~/Desktop/LIDAR_WS
# Run on live LiDAR hardware
python3 radar_hud.py

# Run with Emerald Green palette pinned to desktop background
python3 radar_hud.py --theme green --desktop

# Run with synthetic point clouds for GUI testing without hardware
python3 radar_hud.py --simulate
```

**Features:**
- **Frameless & Translucent**: Renders with alpha-blended transparency directly above or behind your desktop wallpaper.
- **Real-Time Phosphor Sweep**: 60 FPS animated radar sweep beam highlighting reflection points as the beam passes.
- **Target Lock-On**: Automatically clusters points from the nearest physical object (hand, intruder, obstacles) and draws red tactical lock-on brackets with range/angle telemetry.
- **Telemetry HUD**: Displays current scan frequency (Hz), total points, GUI rendering FPS, and closest object distance/bearing.

---

### 2. The Invisible Geofence & Auto-Lock Security (`geofence_security.py`)

Transforms the LiDAR's 4.7m range into an invisible spatial tripwire and perimeter security system for your workspace or office.

```bash
cd ~/Desktop/LIDAR_WS
# Arm system with 5-second countdown to exit the room
python3 geofence_security.py --arm

# Monitor a doorway corridor between 1.0m and 3.0m in front of sensor
python3 geofence_security.py --min-dist 1.0 --max-dist 3.0 --min-angle -45 --max-angle 45

# Dry-run test (captures photo and sends alerts without locking session)
python3 geofence_security.py --dry-run

# Hardware-free simulation mode (demonstrates tripwire trigger & webcam capture)
python3 geofence_security.py --simulate --dry-run
```

**How It Works:**
1. **Static Baseline Calibration**: On startup, it captures 3 seconds of static room geometry (desks, walls, open doors) across all 360 azimuth bins.
2. **Dynamic Intrusion Filtering**: Stationary furniture is ignored. When a new physical cluster of points ($\ge 6$ points) breaches the defined geofence zone, the system triggers.
3. **Multi-Vector Defense Actions**:
   - **Webcam Snapshot**: Immediately activates the webcam (`cv2`), grabs an image of the intruder, watermarks the timestamp, and saves to `security_captures/breach_YYYYMMDD_HHMMSS.jpg`.
   - **Display Blanking**: Turns off active monitors (`hyprctl dispatch dpms off` or `xset dpms force off`).
   - **Session Lock**: Instantly launches `hyprlock` (Hyprland), `xflock4` (Xfce), or `loginctl lock-session`.
   - **Desktop Notification & Audit Log**: Emits critical `notify-send` alerts and logs events to `security_captures/security_audit.log`.

---

### 3. Spatial Touch Interface (`spatial_mouse.py`)

Converts a flat desk, whiteboard, or mid-air plane into an absolute-positioning digital trackpad using Linux `uinput`.

```bash
cd ~/Desktop/LIDAR_WS
python3 spatial_mouse.py
```

**Key Parameters:**
- `MIN_DIST = 0.24` / `MAX_DIST = 0.75`: Active touch zone depth (m).
- `ANGLE_LIMIT = 50.0`: $\pm 50^\circ$ frontal interaction cone.
- `SCREEN_W = 1920`, `SCREEN_H = 1080`: Target display resolution.
- `SMOOTHING = 0.35`: Exponential Moving Average (EMA) coefficient.
- `X_BOUNDS = 0.35`: $\pm 35\text{ cm}$ hand travel mapped to full screen width.
- `Y_MIN = 0.25`, `Y_MAX = 0.65`: Physical depth mapped to top and bottom of screen.

---

### 4. Diagnostic Scanners (`lidar_touch.py` & `test_lidar.py`)

Lightweight terminal utilities for direct telemetry diagnostics and health monitoring:

```bash
python3 lidar_touch.py
```

---

## Window Manager Integration (Hyprland & Xfce)

### Hyprland Integration
To make the Cyberpunk Radar HUD (`radar_hud.py`) float seamlessly on your desktop background without taking focus or showing borders, add this rule to your `~/.config/hypr/hyprland.conf`:

```ini
# YDLIDAR Cyberpunk Radar HUD Window Rules
windowrulev2 = float, class:^(lidar-radar-hud)$
windowrulev2 = pin, class:^(lidar-radar-hud)$
windowrulev2 = noborder, class:^(lidar-radar-hud)$
windowrulev2 = nofocus, class:^(lidar-radar-hud)$
windowrulev2 = noinitialfocus, class:^(lidar-radar-hud)$
windowrulev2 = move 20 20, class:^(lidar-radar-hud)$
windowrulev2 = size 540 540, class:^(lidar-radar-hud)$
windowrulev2 = opacity 0.88 0.88, class:^(lidar-radar-hud)$
```

### Xfce Desktop Integration
To pin the HUD below active windows on Xfce/Xubuntu:
```bash
python3 radar_hud.py --desktop &
# Or set via wmctrl after launch:
wmctrl -r "YDLIDAR Cyberpunk Radar HUD" -b add,below,sticky
```

---

## Physical Calibration Guide

1. Place the LiDAR horizontally at the edge of your desk or wall.
2. Run `python3 spatial_mouse.py`.
3. Hold your hand ~40 cm (1.5 ft) from the spinning turret.
4. Slowly orbit your hand around the LiDAR until you observe tracking coordinates appear in the terminal:
   ```
   [TRACKING] Hand at Dist: 0.40m | Cursor: ( 960,  540) | Cluster pts:  18
   ```
5. That side corresponds to the sensor's **$0^\circ$ forward axis**. Align this direction towards your interaction area.

---

## Troubleshooting & Critical Pitfalls

### 1. Buffer Overflow (`Current points count > max 5000`) & Timeout (`[YDLIDAR]: -1 No error`)
- **Symptoms**:
  - `[warn] Current points count 7743 > max 5000`
  - `[error] [YDLIDAR]: -1 No error` repeated every second.
- **Root Cause**:
  - The default YDLIDAR SDK driver relies strictly on a hardware zero-degree sync bit in Byte 2 of telemetry packets.
  - If a single sync bit is missed or has noise, the driver accumulated unparsed packets in its internal buffer for up to 2.0 seconds (`TIMEOUT_2S = 2000 ms`), ballooning the scan size beyond the 5,000-point hard limit.
  - When the point count exceeded `MAX_SCAN_NODES`, the unpatched driver dropped all points and failed to signal its synchronization event, causing the consumer thread to time out with `-1 No error`.
- **The Permanent Fix (Applied in this repository)**:
  1. **Angle Wraparound Detection**: In `YDlidarDriver.cpp`, we added automatic detection of the $360^\circ \to 0^\circ$ transition (`m_lastStartAngle > 270°` and `sa < 90°`), ensuring scan frames complete every single revolution even if the zero-sync bit is skipped.
  2. **Safe Scan Truncation**: If an oversized packet cluster ever arrives, the driver truncates to the latest revolution rather than discarding the data, and immediately signals `_dataEvent.set()` to prevent thread timeouts.
  3. **Zero-Spam Error Filter**: In `CYdLidar.cpp`, the driver now filters out non-error timeout pulses, eliminating spurious `[YDLIDAR]: -1 No error` messages.

### 2. Port Collision (`/dev/ttyS9` vs `/dev/ttyUSB0`)
- **Cause**: SDK's `lidarPortList()` enumerates internal motherboard buses first.
- **Fix**: Never iterate `lidarPortList()`. Bind directly to `/dev/ttyUSB0`:
  ```python
  laser.setlidaropt(ydlidar.LidarPropSerialPort, "/dev/ttyUSB0")
  ```

### 3. Handshake Failure / Health Code -2
- **Cause**: YDLIDAR X4 uses one-way telemetry and does not reply to baseplate health queries.
- **Fix**: Set single-channel mode:
  ```python
  laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
  ```

### 4. Motor Fails to Spin
- **Cause**: Missing DTR signal.
- **Fix**: Ensure DTR motor control is enabled:
  ```python
  laser.setlidaropt(ydlidar.LidarPropSupportMotorDtrCtrl, True)
  ```

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
YDLIDAR SDK is licensed under the [BSD 2-Clause License](https://opensource.org/licenses/BSD-2-Clause).
