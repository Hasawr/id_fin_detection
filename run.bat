@echo off
setlocal

rem Always run from the folder containing this script.
cd /d "%~dp0"

set "PYTHON_COMMAND=python"
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_COMMAND=%~dp0.venv\Scripts\python.exe"
)

%PYTHON_COMMAND% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Install Python 3.10 or newer and try again.
    pause
    exit /b 1
)

if not exist ".env" (
    copy /y ".env.example" ".env" >nul
    %PYTHON_COMMAND% -c "from pathlib import Path; import secrets; p=Path('.env'); p.write_text(p.read_text(encoding='utf-8').replace('replace-with-a-long-random-key', secrets.token_urlsafe(48)), encoding='utf-8')"
    if errorlevel 1 (
        echo [ERROR] Failed to generate a secure API key.
        exit /b 1
    )
    echo [INFO] Created .env with a random API key.
)

%PYTHON_COMMAND% -c "import ctypes, fastapi, uvicorn, streamlit, multipart, dotenv, cv2, paddle; from importlib.metadata import version; from services.id_fin.detector import configure_nvidia_dll_directories; assert version('paddleocr') == '2.10.0'; assert all(version(name) == '4.11.0.86' for name in ('opencv-python', 'opencv-contrib-python', 'opencv-python-headless')); configure_nvidia_dll_directories(); assert paddle.device.is_compiled_with_cuda(); ctypes.WinDLL('cudnn64_8.dll')" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing missing dependencies and CUDA-enabled PaddlePaddle...
    %PYTHON_COMMAND% -m pip uninstall -y paddlepaddle >nul 2>&1
    %PYTHON_COMMAND% -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless >nul 2>&1
    %PYTHON_COMMAND% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
    %PYTHON_COMMAND% -c "import ctypes, paddle; from services.id_fin.detector import configure_nvidia_dll_directories; configure_nvidia_dll_directories(); assert paddle.device.is_compiled_with_cuda(); ctypes.WinDLL('cudnn64_8.dll')" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] CUDA or cuDNN failed validation.
        pause
        exit /b 1
    )
)

if /i "%~1"=="--check" (
    echo [OK] Configuration and dependencies are ready.
    exit /b 0
)

echo Starting OCR backend at http://127.0.0.1:8000
echo Starting Streamlit frontend at http://127.0.0.1:8501
echo [SECURITY] Streamlit is local-only. Do not expose port 8501 publicly.

start "OCR API Backend" cmd /k ""%PYTHON_COMMAND%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
start "OCR Streamlit Frontend" cmd /k ""%PYTHON_COMMAND%" -m streamlit run demos\streamlit_app.py --server.address 127.0.0.1 --server.port 8501"

echo.
echo Both applications were opened in separate terminal windows.
echo Close those windows to stop the applications.
endlocal
