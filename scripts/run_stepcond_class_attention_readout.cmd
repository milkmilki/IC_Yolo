@echo off
setlocal

set "PROJECT_ROOT=E:\Cjn\PCB_Yolo"
set "PYTHON_EXE=D:\anaconda3\envs\pcb_yolo\python.exe"
set "CONFIG=%PROJECT_ROOT%\AutoResearch\configs\wm811k_autoresearch_stepcond_class_attention_readout.yaml"

cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" "%PROJECT_ROOT%\scripts\run_wm811k_pipeline.py" --config "%CONFIG%"

endlocal
