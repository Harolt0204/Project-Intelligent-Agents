@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Ejecuta Preparar_Proyecto.bat antes de mostrar los modelos.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m experiments.mostrar_lunarlander --group base
pause
