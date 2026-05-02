cmake_minimum_required(VERSION 3.30.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(app
    entities/seeker.cpp

    scenes/StartScene.cpp
    scenes/LearnScene.cpp
    scenes/TestScene.cpp
    scenes/LoadingScene.cpp
    scenes/factory.cpp

    navigation_env.cpp
    python/register.cpp
    python/navigation_env.cpp

    app.cpp
)

# No C++ modules used — disable per-TU module scanning (avoids slow -fmodules-ts dyndep pass)
set_target_properties(app PROPERTIES CXX_SCAN_FOR_MODULES OFF)


target_link_libraries(app PRIVATE Magic::engine)

if(MINGW)
    target_link_options(app PRIVATE -static-libgcc -static-libstdc++)
    find_file(_winpthread_static "libwinpthread.a"
        HINTS "${CMAKE_C_COMPILER_BINDIR}"
        PATHS "C:/msys64/ucrt64/lib" "C:/msys64/mingw64/lib"
        NO_DEFAULT_PATH
    )
    if(_winpthread_static)
        target_link_libraries(app PRIVATE "${_winpthread_static}")
    else()
        target_link_libraries(app PRIVATE -Wl,-Bstatic,-lwinpthread,-Bdynamic)
    endif()

    # Ship MinGW runtime DLLs alongside the app (vendored DLLs like SDL3 pull them in transitively)
    foreach(_rt_dll IN ITEMS libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll)
        find_file(_rt_path "${_rt_dll}"
            HINTS "${CMAKE_C_COMPILER_BINDIR}"
            PATHS "C:/msys64/ucrt64/bin" "C:/msys64/mingw64/bin"
            NO_DEFAULT_PATH
        )
        if(_rt_path)
            add_custom_command(TARGET app POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_rt_path}" $<TARGET_FILE_DIR:app>
                VERBATIM
            )
            install(FILES "${_rt_path}" DESTINATION bin)
        endif()
        unset(_rt_path CACHE)
    endforeach()
endif()

# Post-build: print DLL dependencies of app.exe
find_program(_objdump objdump HINTS "C:/msys64/ucrt64/bin" "C:/msys64/mingw64/bin")
if(_objdump)
    add_custom_command(TARGET app POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DOBJDUMP=${_objdump}
            -DTARGET_FILE=$<TARGET_FILE:app>
            -P "${CMAKE_CURRENT_SOURCE_DIR}/print_dll_deps.cmake"
        COMMENT "Listing runtime DLL dependencies of app.exe"
        VERBATIM
    )
endif()

set(MAGIC_USE_EMBEDDED_PYTHON ON)
magic_enable_embedded_python(app)
magic_install_embedded_python_packages(app PACKAGE_LIST torch numpy typing-extensions pyyaml matplotlib)

set(ASSETS_USE_PYTHON_SYNC ON)

set(APP_ASSETS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/assets")
if(DEFINED Magic_Engine_SOURCE_DIR)
    set(ENGINE_ASSETS_SOURCE_DIR "${Magic_Engine_SOURCE_DIR}/assets")
else()
    set(ENGINE_ASSETS_SOURCE_DIR "${Magic_DIR}/../../../assets")
endif()
set(ASSETS_BUILD_DIR "${CMAKE_BINARY_DIR}/assets")
set(APP_ASSETS_BUILD_DIR "${ASSETS_BUILD_DIR}")
set(ENGINE_ASSETS_BUILD_DIR "${ASSETS_BUILD_DIR}")
file(GLOB_RECURSE APP_ASSETS_SOURCE_FILES
    "${APP_ASSETS_SOURCE_DIR}/*"
)
file(GLOB_RECURSE ENGINE_ASSETS_SOURCE_FILES
    "${ENGINE_ASSETS_SOURCE_DIR}/*"
)

# Generate lightweight manifests at configure time so the build can depend on a
# single file instead of expanding thousands of file dependencies during the
# Ninja "re-check globbed directories" step. The manifest contains each file's
# path and timestamp. Re-running CMake (configure) updates the manifests.
set(APP_ASSETS_MANIFEST "${CMAKE_BINARY_DIR}/app_assets_manifest.txt")
set(ENGINE_ASSETS_MANIFEST "${CMAKE_BINARY_DIR}/engine_assets_manifest.txt")

file(REMOVE "${APP_ASSETS_MANIFEST}" "${ENGINE_ASSETS_MANIFEST}")
file(WRITE "${APP_ASSETS_MANIFEST}" "")
file(WRITE "${ENGINE_ASSETS_MANIFEST}" "")

foreach(_f IN LISTS APP_ASSETS_SOURCE_FILES)
    file(TO_CMAKE_PATH "${_f}" _p)
    file(TIMESTAMP "${_f}" _t)
    file(APPEND "${APP_ASSETS_MANIFEST}" "${_p}|${_t}\n")
endforeach()

foreach(_f IN LISTS ENGINE_ASSETS_SOURCE_FILES)
    file(TO_CMAKE_PATH "${_f}" _p)
    file(TIMESTAMP "${_f}" _t)
    file(APPEND "${ENGINE_ASSETS_MANIFEST}" "${_p}|${_t}\n")
endforeach()

set(APP_ASSETS_STAMP "${APP_ASSETS_BUILD_DIR}/.assets_stamp")

# Toggle: set to ON to force the Python incremental sync even on Windows
# (useful for debugging or when robocopy is unavailable).
option(ASSETS_USE_PYTHON_SYNC "Use sync_assets.py instead of robocopy on Windows" OFF)

# Build-time checker target: run the CMake script which compares manifests to
# the current source trees and performs the copy only if something changed.
add_custom_target(copy_app_assets
    COMMAND ${CMAKE_COMMAND}
    -DAPP_MANIFEST=${APP_ASSETS_MANIFEST}
    -DENGINE_MANIFEST=${ENGINE_ASSETS_MANIFEST}
    -DAPP_SRC=${APP_ASSETS_SOURCE_DIR}
    -DENGINE_SRC=${ENGINE_ASSETS_SOURCE_DIR}
    -DAPP_BUILD_DIR=${ASSETS_BUILD_DIR}
    -DAPP_STAMP=${APP_ASSETS_STAMP}
    -DUSE_PYTHON_SYNC=${ASSETS_USE_PYTHON_SYNC}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_assets_manifest.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Checking and copying app and engine assets to build tree"
    VERBATIM
)

add_dependencies(app copy_app_assets)

target_compile_features(app PRIVATE cxx_std_20)


install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/assets/"
    DESTINATION "assets"
    PATTERN "__pycache__" EXCLUDE
)

install(
    DIRECTORY "${ENGINE_ASSETS_SOURCE_DIR}/"
    DESTINATION "assets"
    PATTERN "__pycache__" EXCLUDE
)

target_compile_options(app PRIVATE
    $<$<CONFIG:Debug>:-DDEBUG -DMAGIC_DEBUG>
    $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic -g>

    # $<$<CONFIG:Release>:-O3>
)

if(WIN32)
    set_target_properties(app PROPERTIES
        WIN32_EXECUTABLE "$<$<CONFIG:Release>:TRUE>$<$<CONFIG:Debug>:FALSE>"
    )

endif()


# install(TARGETS engine
#     DESTINATION bin
#     COMPONENT app
# )

install(TARGETS app
    RUNTIME DESTINATION bin)

# Install engine + Python + GLEW DLLs
install(FILES
    "$<TARGET_FILE_DIR:app>/libengine.dll"
    "$<TARGET_FILE_DIR:app>/libJolt.dll"
    "$<TARGET_FILE_DIR:app>/python314.dll"
    "$<TARGET_FILE_DIR:app>/python3.dll"
    DESTINATION bin
    OPTIONAL
)
install(FILES "${Magic_DIR}/../../../bin/glew32.dll"
    DESTINATION bin
)

# Install embedded Python runtime directory (exclude dev headers and pycache)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
    DESTINATION bin
    PATTERN "__pycache__" EXCLUDE
    PATTERN "include" EXCLUDE
)

# if(MINGW)
#     find_file(_winpthread_dll "libwinpthread-1.dll"
#         HINTS
#             "${CMAKE_C_COMPILER_AR}"
#             "${CMAKE_CXX_COMPILER}"
#         PATH_SUFFIXES "../bin" "../../bin"
#         PATHS
#             "C:/msys64/ucrt64/bin"
#             "C:/msys64/mingw64/bin"
#         NO_DEFAULT_PATH
#     )
#     if(_winpthread_dll)
#         install(FILES "${_winpthread_dll}" DESTINATION bin)
#     endif()
# endif()
# install(FILES "${Magic_Engine_SOURCE_DIR}/apps/CONTROLS.md"
#     DESTINATION .
#     COMPONENT app
# )

# Include test executables for the app (apps/tests/CMakeLists.txt)
# add_subdirectory(tests)

# set_target_properties(app PROPERTIES
#     WIN32_EXECUTABLE "$<$<CONFIG:Release>:TRUE>$<$<CONFIG:Debug>:FALSE>"
# )
