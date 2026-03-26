@echo off
echo Starting RTSP server + dual camera streams...
echo.

:: Start mediamtx RTSP server in background
start "RTSP Server" /MIN cmd /c "cd /d %~dp0\mediamtx && mediamtx.exe"
timeout /t 2 /nobreak >nul

:: Start both ffmpeg streams simultaneously
echo Starting cam66 and cam68 streams...
start "cam66" /MIN cmd /c "ffmpeg -re -stream_loop -1 -i %~dp0\..\uploads\cam66_20260323_184338.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -r 25 -f rtsp rtsp://localhost:8554/cam66 2>nul"
start "cam68" /MIN cmd /c "ffmpeg -re -stream_loop -1 -i %~dp0\..\uploads\cam68_20260323_184338.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -r 25 -f rtsp rtsp://localhost:8554/cam68 2>nul"

echo.
echo Streams running:
echo   cam66: rtsp://localhost:8554/cam66
echo   cam68: rtsp://localhost:8554/cam68
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
