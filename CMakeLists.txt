cmake_minimum_required(VERSION 2.6)
project(dag)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/config)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Fix for Qt Creator to list the hxx files.
file(GLOB HEADERS "include/vigra/*.hxx")
add_custom_target(dag SOURCES ${HEADERS})

# C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# VIGRA
find_package(VIGRA REQUIRED)
include_directories(${VIGRA_INCLUDE_DIR})

# HDF5
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})

# Add tests
add_subdirectory(test)
