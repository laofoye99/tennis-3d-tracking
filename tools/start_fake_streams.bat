@echo off
setlocal
set "ROOT=%~dp0.."
set "MTX_DIR=%~dp0mediamtx"
set "LOG_DIR=%ROOT%\reports\fake_stream_logs"
set "CAM66_VIDEO=%ROOT%\uploads\cam66_20260307_173403_2min.mp4"
rem set "CAM68_VIDEO=%ROOT%\uploads\cam68_20260404_075325_2min_from_frames.mp4"
set "CAM68_VIDEO=D:\tennis-dataset\1001\FV9942588_1_cloudrecord_20260525102315_20260525110000_segments\part_016_20260525103915_20260525104015.mp4"

echo Starting RTSP server + dual camera streams...
echo.

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%MTX_DIR%\mediamtx.exe" (
  echo ERROR: Missing "%MTX_DIR%\mediamtx.exe"
  goto :fail
)

if not exist "%MTX_DIR%\mediamtx.yml" (
  if exist "%MTX_DIR%\mediamtx.zip" (
    echo Restoring mediamtx.yml from mediamtx.zip...
    tar -xf "%MTX_DIR%\mediamtx.zip" -C "%MTX_DIR%" mediamtx.yml
  )
)

if not exist "%MTX_DIR%\mediamtx.yml" (
  echo ERROR: Missing "%MTX_DIR%\mediamtx.yml"
  goto :fail
)

if not exist "%CAM66_VIDEO%" (
  echo ERROR: Missing "%CAM66_VIDEO%"
  goto :fail
)

if not exist "%CAM68_VIDEO%" (
  echo ERROR: Missing "%CAM68_VIDEO%"
  goto :fail
)

:: Start mediamtx RTSP server in background
taskkill /IM mediamtx.exe /F >nul 2>&1
start "RTSP Server" /MIN cmd /c "cd /d "%MTX_DIR%" && mediamtx.exe mediamtx.yml > "%LOG_DIR%\mediamtx.out.log" 2> "%LOG_DIR%\mediamtx.err.log""
timeout /t 2 /nobreak >nul

:: Start both ffmpeg streams simultaneously
echo Starting cam66 and cam68 streams...
taskkill /IM ffmpeg.exe /F >nul 2>&1
start "cam66" /MIN cmd /c "ffmpeg -re -stream_loop -1 -i "%CAM66_VIDEO%" -c:v libx264 -preset ultrafast -tune zerolatency -r 25 -f rtsp rtsp://localhost:8554/cam66 > "%LOG_DIR%\cam66.out.log" 2> "%LOG_DIR%\cam66.err.log""
start "cam68" /MIN cmd /c "ffmpeg -re -stream_loop -1 -i "%CAM68_VIDEO%" -c:v libx264 -preset ultrafast -tune zerolatency -r 25 -f rtsp rtsp://localhost:8554/cam68 > "%LOG_DIR%\cam68.out.log" 2> "%LOG_DIR%\cam68.err.log""

echo.
echo Streams running:
echo   cam66: rtsp://localhost:8554/cam66
echo   cam68: rtsp://localhost:8554/cam68
echo   logs:  %LOG_DIR%
echo.
echo Press any key to stop all streams...
pause >nul

:: Kill all
taskkill /FI "WINDOWTITLE eq RTSP Server" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq cam66" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq cam68" /F >nul 2>&1
taskkill /IM mediamtx.exe /F >nul 2>&1
taskkill /IM ffmpeg.exe /F >nul 2>&1
echo Stopped.
exit /b 0

:fail
echo.
echo Startup failed. Check the paths above.
exit /b 1
