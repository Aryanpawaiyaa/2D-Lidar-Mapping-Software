cmake_minimum_required(VERSION 3.10)
project(tg50_bridge CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ydlidar_sdk QUIET)

if(NOT ydlidar_sdk_FOUND)
    # Check if YDLidar-SDK is cloned alongside or in /usr/local
    find_path(YDLIDAR_INCLUDE_DIR CYdLidar.h PATHS /usr/local/include /usr/include ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/YDLidar-SDK/include)
    find_library(YDLIDAR_LIBRARY ydlidar_sdk PATHS /usr/local/lib /usr/lib ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/YDLidar-SDK/build)
endif()

add_executable(tg50_bridge tg50_bridge.cpp)

if(TARGET ydlidar_sdk)
    target_link_libraries(tg50_bridge PRIVATE ydlidar_sdk)
elseif(YDLIDAR_LIBRARY)
    target_include_directories(tg50_bridge PRIVATE ${YDLIDAR_INCLUDE_DIR})
    target_link_libraries(tg50_bridge PRIVATE ${YDLIDAR_LIBRARY} pthread)
endif()

