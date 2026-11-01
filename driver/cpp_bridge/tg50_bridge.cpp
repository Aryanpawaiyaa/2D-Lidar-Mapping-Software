/**
 * tg50_bridge.cpp: Thin C++ capture program using official YDLidar-SDK.
 * Streams binary scans to stdout for IPC consumption by Python.
 */

#include "CYdLidar.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cstring>

using namespace ydlidar;

#pragma pack(push, 1)
struct FrameHeader {
    uint32_t magic;       // 0x59444C44 ('YDLD')
    uint32_t point_count;
    double timestamp;
};

struct PointData {
    float angle_rad;
    float range_m;
    float intensity;
};
#pragma pack(pop)

int main(int argc, char* argv[]) {
    std::string port = "/dev/ttyUSB0";
    int baudrate = 512000;
    float scan_freq = 7.0f;
    int sample_rate = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = argv[++i];
        if (arg == "--baud" && i + 1 < argc) baudrate = std::stoi(argv[++i]);
        if (arg == "--freq" && i + 1 < argc) scan_freq = std::stof(argv[++i]);
        if (arg == "--sample_rate" && i + 1 < argc) sample_rate = std::stoi(argv[++i]);
    }

    ydlidar::os_init();
    CYdLidar laser;

    laser.setlidaropt(LidarPropSerialPort, port.c_str(), port.size());
    laser.setlidaropt(LidarPropSerialBaudrate, &baudrate, sizeof(int));
    
    // Explicit SDK named macros for TG50
    int lidar_type = TYPE_TOF;
    laser.setlidaropt(LidarPropLidarType, &lidar_type, sizeof(int));

    int device_type = YDLIDAR_TYPE_SERIAL;
    laser.setlidaropt(LidarPropDeviceType, &device_type, sizeof(int));

    laser.setlidaropt(LidarPropSampleRate, &sample_rate, sizeof(int));
    laser.setlidaropt(LidarPropScanFrequency, &scan_freq, sizeof(float));

    float min_range = 0.05f;
    float max_range = 50.0f;
    laser.setlidaropt(LidarPropMinRange, &min_range, sizeof(float));
    laser.setlidaropt(LidarPropMaxRange, &max_range, sizeof(float));

    float min_angle = -180.0f;
    float max_angle = 180.0f;
    laser.setlidaropt(LidarPropMinAngle, &min_angle, sizeof(float));
    laser.setlidaropt(LidarPropMaxAngle, &max_angle, sizeof(float));

    bool intensities = true;
    laser.setlidaropt(LidarPropIntensities, &intensities, sizeof(bool));

    bool single_channel = false;
    laser.setlidaropt(LidarPropSingleChannel, &single_channel, sizeof(bool));

    int abnormal_count = 4;
    laser.setlidaropt(LidarPropAbnormalCheckCount, &abnormal_count, sizeof(int));

    if (!laser.initialize()) {
        std::cerr << "CYdLidar initialization failed: " << laser.DescribeError() << std::endl;
        return 1;
    }

    if (!laser.turnOn()) {
        std::cerr << "CYdLidar failed to turnOn." << std::endl;
        return 2;
    }

    LaserScan scan;
    while (ydlidar::os_isOk()) {
        if (laser.doProcessSimple(scan)) {
            uint32_t count = static_cast<uint32_t>(scan.points.size());
            if (count == 0) continue;

            auto now = std::chrono::system_clock::now();
            double ts = std::chrono::duration<double>(now.time_since_epoch()).count();

            FrameHeader header;
            header.magic = 0x59444C44;
            header.point_count = count;
            header.timestamp = ts;

            std::vector<PointData> out_pts(count);
            for (size_t i = 0; i < count; ++i) {
                out_pts[i].angle_rad = scan.points[i].angle;
                out_pts[i].range_m = scan.points[i].range;
                out_pts[i].intensity = static_cast<float>(scan.points[i].intensity);
            }

            // Write frame to stdout
            std::cout.write(reinterpret_cast<const char*>(&header), sizeof(FrameHeader));
            std::cout.write(reinterpret_cast<const char*>(out_pts.data()), count * sizeof(PointData));
            std::cout.flush();
        }
    }

    laser.turnOff();
    laser.disconnecting();
    return 0;
}

