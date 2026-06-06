@echo off
setlocal

set "PROJECT_ROOT=E:\Cjn\PCB_Yolo"
set "PYTHON_EXE=D:\anaconda3\envs\pcb_yolo\python.exe"
set "RUN_DIR=%PROJECT_ROOT%\runs\classify\autoresearch_yoloctm_nodistill_stepcond_attention_readout_tau04_e10_20260605_220639"

cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" "%PROJECT_ROOT%\scripts\run_wm811k_pipeline.py" --config "%RUN_DIR%\config.yaml" --train-resume-checkpoint "%RUN_DIR%\last_yoloctm.pt"

endlocal
