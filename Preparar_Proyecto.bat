@echo off
cd /d "%~dp0"
echo Preparando Project Intelligent Agents - Tier 1
echo Se necesita Python 3.12 instalado con el lanzador py.
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo No se encontro Python 3.12. Instalalo y vuelve a abrir este archivo.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo No se pudo crear el entorno del proyecto.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo No se pudieron instalar todas las dependencias. Revisa el mensaje anterior.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m experiments.verificar_instalacion
if errorlevel 1 (
    echo La comprobacion no termino correctamente. Revisa el mensaje anterior.
    pause
    exit /b 1
)
echo.
echo Listo. Ahora abre Mostrar_Tier1.bat para ver los agentes.
pause
