@echo off
setlocal

set BUILD_DIR=build-release
set INSTALL_DIR=%~dp0seeker-agent-learn-install
set DO_TRACY=OFF

@REM if not exist "%BUILD_DIR%\CMakeCache.txt" (
    cmake -S . -B "%BUILD_DIR%" -G "Ninja" ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_INSTALL_MESSAGE=LAZY ^
      -DMagic_DIR="%~dp0engine/lib/cmake/Magic" ^
      -DENABLE_TRACY=DO_TRACY ^
      -DPython3_ROOT_DIR=C:/Github/magic-engine/python ^
      -DPython3_EXECUTABLE=C:/Github/magic-engine/python/python.exe
    if errorlevel 1 exit /b 1
@REM )

cmake --build "%BUILD_DIR%" --config Release -j 6
if errorlevel 1 exit /b 1

cmake --install "%BUILD_DIR%" --config Release --prefix "%INSTALL_DIR%"
if errorlevel 1 exit /b 1

echo Installed to: %INSTALL_DIR%
