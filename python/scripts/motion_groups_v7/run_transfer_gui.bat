@echo off
setlocal EnableExtensions
REM ============================================================================
REM  Windows launcher for the motion_groups_v7 expression-transfer GUI.
REM  Same program as on macOS, just bootstrapped for Windows:
REM    - creates a dedicated Windows venv (.venv-win at the repo root) if missing
REM    - installs the dependencies from requirements-windows.txt if missing
REM    - launches transfer_gui.py
REM  Double-click this file, or run it from PowerShell / cmd.
REM ============================================================================

set "SCRIPT_DIR=%~dp0"

REM repo root = three levels up from python\scripts\motion_groups_v7\
pushd "%SCRIPT_DIR%..\..\.."
set "REPO=%CD%"
popd

set "VENV=%REPO%\.venv-win"
set "VPY=%VENV%\Scripts\python.exe"

REM ── 1) ensure the venv exists ────────────────────────────────────────────────
if not exist "%VPY%" (
    echo [setup] Creating Windows virtual env at "%VENV%" ...
    REM Prefer Python 3.9 (best wheel coverage for open3d/mediapipe); fall back.
    py -3.9 -m venv "%VENV%" 2>nul || py -3 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
    if not exist "%VPY%" (
        echo [error] Could not create the virtual env.
        echo         Install Python 3.9-3.12 ^(64-bit^) from python.org and retry.
        pause
        exit /b 1
    )
)

REM ── 2) ensure dependencies are installed (check, don't assume) ───────────────
"%VPY%" -c "import numpy, scipy, sklearn, h5py, PIL, trimesh, open3d, mediapipe" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing dependencies ^(first run only, may take a few minutes^) ...
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip install -r "%SCRIPT_DIR%requirements-windows.txt"
    if errorlevel 1 (
        echo [error] Dependency install failed - see the messages above.
        pause
        exit /b 1
    )
    REM verify again so we never launch a half-installed env
    "%VPY%" -c "import numpy, scipy, sklearn, h5py, PIL, trimesh, open3d, mediapipe" >nul 2>&1
    if errorlevel 1 (
        echo [error] Dependencies still missing after install - see messages above.
        pause
        exit /b 1
    )
)

REM ── 3) launch ────────────────────────────────────────────────────────────────
echo [run] Launching transfer GUI ...
"%VPY%" "%SCRIPT_DIR%transfer_gui.py"
if errorlevel 1 (
    echo [error] The GUI exited with an error ^(see above^).
    pause
)
endlocal
