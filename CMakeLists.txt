cmake_minimum_required(VERSION 3.20)
project(seeker-agent-learn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



set(Magic_DIR "${PROJECT_SOURCE_DIR}/engine/lib/cmake/Magic")
if(NOT DEFINED Magic_DIR)
    message(FATAL_ERROR "Magic_DIR must be defined and point to the Magic-Engine app source directory")
endif()

option(ENABLE_TRACY "Enable Tracy profiler" OFF)
set(TRACY_REQUIRED_VERSION "0.13.1")
if(ENABLE_TRACY)
    set(TRACY_VERSION_FILE "${PROJECT_SOURCE_DIR}/extern/tracy/public/common/TracyVersion.hpp")
    if(NOT EXISTS "${TRACY_VERSION_FILE}")
        message(FATAL_ERROR "Cannot locate Tracy version file: ${TRACY_VERSION_FILE}")
    endif()
    file(READ "${TRACY_VERSION_FILE}" TRACY_VERSION_CONTENTS)
    string(REGEX MATCH "Major = ([0-9]+)" _ ${TRACY_VERSION_CONTENTS})
    set(TRACY_VERSION_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "Minor = ([0-9]+)" _ ${TRACY_VERSION_CONTENTS})
    set(TRACY_VERSION_MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "Patch = ([0-9]+)" _ ${TRACY_VERSION_CONTENTS})
    set(TRACY_VERSION_PATCH ${CMAKE_MATCH_1})
    set(TRACY_VERSION_STRING "${TRACY_VERSION_MAJOR}.${TRACY_VERSION_MINOR}.${TRACY_VERSION_PATCH}")
    if(NOT TRACY_VERSION_STRING STREQUAL TRACY_REQUIRED_VERSION)
        message(FATAL_ERROR "Tracy version mismatch: expected ${TRACY_REQUIRED_VERSION}, but extern/tracy reports ${TRACY_VERSION_STRING}. Please checkout v${TRACY_REQUIRED_VERSION} in extern/tracy.")
    endif()
    # set(TRACY_ENABLE ON CACHE BOOL "Enable profiling in Tracy" FORCE)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/tracy)
endif()

find_package(Magic CONFIG REQUIRED)
add_subdirectory(src)

message(STATUS "seeker-agent-learn configured with Magic_DIR=${Magic_DIR}")
