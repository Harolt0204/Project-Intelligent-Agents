@echo off
cd /d "%~dp0"
if not exist "results\tier23_lunarlander\base\seed_42\online_final.pt" (
    echo El modelo final todavia no esta disponible. Consulta el estado en Tier2_LunarLander.ipynb.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m experiments.mostrar_lunarlander --group base
pause
