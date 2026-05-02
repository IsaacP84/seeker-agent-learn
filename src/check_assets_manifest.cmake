if(NOT DEFINED APP_MANIFEST)
    message(FATAL_ERROR "APP_MANIFEST not defined")
endif()
if(NOT DEFINED ENGINE_MANIFEST)
    message(FATAL_ERROR "ENGINE_MANIFEST not defined")
endif()
if(NOT DEFINED APP_SRC)
    message(FATAL_ERROR "APP_SRC not defined")
endif()
if(NOT DEFINED ENGINE_SRC)
    message(FATAL_ERROR "ENGINE_SRC not defined")
endif()
if(NOT DEFINED APP_BUILD_DIR)
    message(FATAL_ERROR "APP_BUILD_DIR not defined")
endif()
if(NOT DEFINED APP_STAMP)
    message(FATAL_ERROR "APP_STAMP not defined")
endif()

message(STATUS "Checking assets manifests...")

# Helper to build a manifest string for a source dir
function(build_manifest src_dir out_var)
    file(GLOB_RECURSE _files RELATIVE "${src_dir}" "${src_dir}/*")
    set(_manifest "")
    foreach(_f IN LISTS _files)
        # Convert to absolute path for timestamp
        set(_full "${src_dir}/${_f}")
        file(TO_CMAKE_PATH "${_full}" _p)
        file(TIMESTAMP "${_full}" _t)
        if(NOT _t)
            set(_t "0")
        endif()
        set(_manifest "${_manifest}${_p}|${_t}\n")
    endforeach()
    set(${out_var} "${_manifest}" PARENT_SCOPE)
endfunction()

build_manifest("${APP_SRC}" _app_manifest_new)
build_manifest("${ENGINE_SRC}" _engine_manifest_new)

if(EXISTS "${APP_MANIFEST}")
    file(READ "${APP_MANIFEST}" _app_manifest_old)
else()
    set(_app_manifest_old "")
endif()

if(EXISTS "${ENGINE_MANIFEST}")
    file(READ "${ENGINE_MANIFEST}" _engine_manifest_old)
else()
    set(_engine_manifest_old "")
endif()

set(_need_copy FALSE)
if(NOT EXISTS "${APP_BUILD_DIR}")
    message(STATUS "Build asset directory missing: ${APP_BUILD_DIR}")
    set(_need_copy TRUE)
elseif(NOT EXISTS "${APP_STAMP}")
    message(STATUS "Asset stamp missing: ${APP_STAMP}")
    set(_need_copy TRUE)
endif()

if(NOT _app_manifest_old STREQUAL _app_manifest_new)
    message(STATUS "App assets changed")
    set(_need_copy TRUE)
endif()
if(NOT _engine_manifest_old STREQUAL _engine_manifest_new)
    message(STATUS "Engine assets changed")
    set(_need_copy TRUE)
endif()

if(NOT _need_copy)
    if(EXISTS "${ENGINE_SRC}")
        file(GLOB_RECURSE _engine_src_files RELATIVE "${ENGINE_SRC}" "${ENGINE_SRC}/*")
        foreach(_f IN LISTS _engine_src_files)
            if(NOT EXISTS "${APP_BUILD_DIR}/${_f}")
                message(STATUS "Missing engine build asset: ${APP_BUILD_DIR}/${_f}")
                set(_need_copy TRUE)
                break()
            endif()
        endforeach()
    endif()
endif()

if(NOT _need_copy)
    if(EXISTS "${APP_SRC}")
        file(GLOB_RECURSE _app_src_files RELATIVE "${APP_SRC}" "${APP_SRC}/*")
        foreach(_f IN LISTS _app_src_files)
            if(NOT EXISTS "${APP_BUILD_DIR}/${_f}")
                message(STATUS "Missing app build asset: ${APP_BUILD_DIR}/${_f}")
                set(_need_copy TRUE)
                break()
            endif()
        endforeach()
    endif()
endif()

if(_need_copy)
    message(STATUS "Updating build asset tree: copying assets...")
    # Ensure target directory exists. On Windows prefer robocopy for efficient incremental sync.
    execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory "${APP_BUILD_DIR}")
    if(WIN32 AND NOT USE_PYTHON_SYNC)
        # Use robocopy to mirror engine assets, then overlay app assets.
        execute_process(
            COMMAND robocopy "${ENGINE_SRC}" "${APP_BUILD_DIR}" /MIR /NFL /NDL /NJH /NJS /NP
            RESULT_VARIABLE _rc1
            OUTPUT_QUIET ERROR_QUIET
        )
        # robocopy returns a bitmask; values <= 7 are not fatal.
        if(_rc1 GREATER 7)
            message(FATAL_ERROR "robocopy failed for engine assets with exit code ${_rc1}")
        endif()

        # Use /E /XO to overlay app assets on top of engine assets WITHOUT purging engine files.
        # /E = copy subdirectories including empty, /XO = skip older (only copy new/changed)
        execute_process(
            COMMAND robocopy "${APP_SRC}" "${APP_BUILD_DIR}" /E /XO /NFL /NDL /NJH /NJS /NP
            RESULT_VARIABLE _rc2
            OUTPUT_QUIET ERROR_QUIET
        )
        if(_rc2 GREATER 7)
            message(FATAL_ERROR "robocopy failed for app assets with exit code ${_rc2}")
        endif()
    else()
        # Attempt to find a Python interpreter
        find_program(_PY_EXECUTABLE python3)
        if(NOT _PY_EXECUTABLE)
            find_program(_PY_EXECUTABLE python)
        endif()
        if(NOT _PY_EXECUTABLE)
            message(FATAL_ERROR "No suitable Python interpreter found for incremental sync")
        endif()

        execute_process(
            COMMAND "${_PY_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/sync_assets.py" "${ENGINE_SRC}" "${APP_SRC}" "${APP_BUILD_DIR}"
            RESULT_VARIABLE _sync_rc
            OUTPUT_VARIABLE _sync_out
            ERROR_VARIABLE _sync_err
        )
        if(NOT _sync_rc EQUAL 0)
            message(FATAL_ERROR "sync_assets.py failed: ${_sync_rc} ${_sync_err}")
        else()
            message(STATUS "sync_assets.py: ${_sync_out}")
        endif()
    endif()

    # write manifests
    file(WRITE "${APP_MANIFEST}" "${_app_manifest_new}")
    file(WRITE "${ENGINE_MANIFEST}" "${_engine_manifest_new}")

    # touch stamp
    execute_process(COMMAND "${CMAKE_COMMAND}" -E touch "${APP_STAMP}")
    message(STATUS "Assets updated and stamp touched")
else()
    message(STATUS "No asset changes detected")
endif()
