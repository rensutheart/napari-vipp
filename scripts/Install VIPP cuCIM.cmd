@echo off
setlocal

if exist "%~dp0scripts\install_cucim_windows.cmd" (
    call "%~dp0scripts\install_cucim_windows.cmd" %*
) else (
    call "%~dp0install_cucim_windows.cmd" %*
)

exit /b %ERRORLEVEL%
