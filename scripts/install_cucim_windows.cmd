@echo off
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_cucim_windows.ps1" %*
set "VIPP_CUCIM_INSTALL_EXIT=%ERRORLEVEL%"

if not "%VIPP_CUCIM_NO_PAUSE%"=="1" (
    echo.
    if "%VIPP_CUCIM_INSTALL_EXIT%"=="0" (
        echo The VIPP cuCIM installer finished successfully.
    ) else (
        echo The VIPP cuCIM installer stopped with exit code %VIPP_CUCIM_INSTALL_EXIT%.
    )
    pause
)

exit /b %VIPP_CUCIM_INSTALL_EXIT%
