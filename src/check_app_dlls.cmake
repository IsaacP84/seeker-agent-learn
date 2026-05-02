# Called by CTest via cmake -P to verify required DLLs are present next to app.exe.
# Expects -DDLLS=<semicolon-separated list of absolute DLL paths> on the command line.
foreach(dll IN LISTS DLLS)
    if(NOT EXISTS "${dll}")
        message(FATAL_ERROR "Required DLL not found: ${dll}")
    endif()
endforeach()
message(STATUS "All required DLLs are present next to app.exe")
