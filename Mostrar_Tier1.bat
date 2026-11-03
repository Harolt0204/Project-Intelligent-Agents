@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No se encuentra el entorno de Python del proyecto.
    echo Consulta COMO_MOSTRAR_TIER1.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m experiments.mostrar_tier1
if errorlevel 1 pause
