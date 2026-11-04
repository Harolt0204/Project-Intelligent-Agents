@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Primero abre Preparar_Proyecto.bat para crear el entorno.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-notebook.txt
if errorlevel 1 (
    echo No se pudo preparar el notebook. Revisa el mensaje anterior.
    pause
    exit /b 1
)
echo Listo. Abre Tier1_Explicado.ipynb en VS Code.
echo En Seleccionar kernel, escoge el Python de .venv.
pause
