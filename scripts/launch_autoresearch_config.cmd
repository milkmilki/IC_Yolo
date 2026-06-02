@echo off
setlocal

set "PROJECT_ROOT=E:\Cjn\PCB_Yolo"
set "PYTHON_EXE=D:\anaconda3\envs\pcb_yolo\python.exe"
set "FOREGROUND=0"
if /I "%~1"=="/foreground" (
    set "FOREGROUND=1"
    shift
)
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=AutoResearch\configs\wm811k_autoresearch_topology_ctm.yaml"

if not exist "%PYTHON_EXE%" (
    echo Python executable not found: %PYTHON_EXE% 1>&2
    exit /b 1
)
if not exist "%PROJECT_ROOT%\scripts\run_wm811k_pipeline.py" (
    echo Pipeline script not found: %PROJECT_ROOT%\scripts\run_wm811k_pipeline.py 1>&2
    exit /b 1
)
if not exist "%CONFIG%" (
    if exist "%PROJECT_ROOT%\%CONFIG%" (
        set "CONFIG=%PROJECT_ROOT%\%CONFIG%"
    ) else (
        echo Config not found: %CONFIG% 1>&2
        exit /b 1
    )
)

if not exist "%PROJECT_ROOT%\AutoResearch\launch_logs" mkdir "%PROJECT_ROOT%\AutoResearch\launch_logs"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "STDOUT_LOG=%PROJECT_ROOT%\AutoResearch\launch_logs\autoresearch_cmd_%STAMP%.out.log"
set "STDERR_LOG=%PROJECT_ROOT%\AutoResearch\launch_logs\autoresearch_cmd_%STAMP%.err.log"

pushd "%PROJECT_ROOT%"
if "%FOREGROUND%"=="1" (
    "%PYTHON_EXE%" "%PROJECT_ROOT%\scripts\run_wm811k_pipeline.py" --config "%CONFIG%"
) else (
    start "wm811k-autoresearch" /B "%PYTHON_EXE%" "%PROJECT_ROOT%\scripts\run_wm811k_pipeline.py" --config "%CONFIG%" > "%STDOUT_LOG%" 2> "%STDERR_LOG%"
)
popd

echo stdout=%STDOUT_LOG%
echo stderr=%STDERR_LOG%
endlocal
