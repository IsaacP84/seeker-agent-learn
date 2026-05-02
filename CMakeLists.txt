cmake_minimum_required(VERSION 3.20)
project(seeker-agent-learn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(Magic_DIR "${PROJECT_SOURCE_DIR}/engine/lib/cmake/Magic")
if(NOT DEFINED Magic_DIR)
    message(FATAL_ERROR "Magic_DIR must be defined and point to the Magic-Engine app source directory")
endif()
find_package(Magic CONFIG REQUIRED)
add_subdirectory(src)



message(STATUS "seeker-agent-learn configured with Magic_DIR=${Magic_DIR}")
