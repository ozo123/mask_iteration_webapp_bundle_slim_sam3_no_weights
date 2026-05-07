@echo off
setlocal
set "ENV_NAME=%CONDA_ENV_NAME%"
if "%ENV_NAME%"=="" set "ENV_NAME=mask_iteration_sam3"
set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

if "%CONDA_EXE%"=="" (
  set "CONDA_CMD=conda"
) else (
  set "CONDA_CMD=%CONDA_EXE%"
)

"%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" ^
  python "%PROJECT_DIR%\start_webapp.py" ^
  --sam3-repo-dir "%PROJECT_DIR%\third_party\sam3" ^
  --checkpoint "%PROJECT_DIR%\third_party\sam3\checkpoints\sam3.pt" ^
  --device auto ^
  --validate-tools-dir "%PROJECT_DIR%\Validate_tools" ^
  %*

endlocal
