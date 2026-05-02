# Called as a cmake -P script post-build.
# Runs objdump -p on the target and prints only the DLL Name lines.
execute_process(
    COMMAND "${OBJDUMP}" -p "${TARGET_FILE}"
    OUTPUT_VARIABLE _output
    ERROR_QUIET
)

get_filename_component(_name "${TARGET_FILE}" NAME)
message("--- Runtime DLL dependencies of ${_name} ---")
string(REPLACE "\n" ";" _lines "${_output}")
foreach(_line IN LISTS _lines)
    if(_line MATCHES "DLL Name: (.+)")
        message("  ${CMAKE_MATCH_1}")
    endif()
endforeach()
message("---")
