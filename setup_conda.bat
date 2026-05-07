@echo off
setlocal
set "ENV_NAME=mask_iteration_sam3"
set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

if "%CONDA_EXE%"=="" (
  set "CONDA_CMD=conda"
) else (
  set "CONDA_CMD=%CONDA_EXE%"
)

set "CHECKPOINT_SRC=%SAM3_CHECKPOINT_SRC%"
if "%CHECKPOINT_SRC%"=="" set "CHECKPOINT_SRC=%USERPROFILE%\Desktop\sam3.pt"
set "CHECKPOINT_DST=%PROJECT_DIR%\third_party\sam3\checkpoints\sam3.pt"

"%CONDA_CMD%" env list | findstr /R /C:"^%ENV_NAME%[ ]" >nul
if errorlevel 1 (
  "%CONDA_CMD%" create -y -n "%ENV_NAME%" python=3.11
)

"%CONDA_CMD%" run -n "%ENV_NAME%" python -m pip install --upgrade pip setuptools wheel
"%CONDA_CMD%" run -n "%ENV_NAME%" python -m pip install torch torchvision
"%CONDA_CMD%" run -n "%ENV_NAME%" python -m pip install -r "%PROJECT_DIR%\requirements.txt"
"%CONDA_CMD%" run -n "%ENV_NAME%" python -m pip install -e "%PROJECT_DIR%\third_party\sam3"

if not exist "%PROJECT_DIR%\third_party\sam3\checkpoints" mkdir "%PROJECT_DIR%\third_party\sam3\checkpoints"
if not exist "%CHECKPOINT_DST%" (
  if not exist "%CHECKPOINT_SRC%" (
    echo Missing SAM3 checkpoint: %CHECKPOINT_SRC% 1>&2
    exit /b 1
  )
  copy /Y "%CHECKPOINT_SRC%" "%CHECKPOINT_DST%" >nul
)

echo Conda env ready: %ENV_NAME%
echo SAM3 checkpoint: %CHECKPOINT_DST%
echo Run: run_conda.bat
endlocal
