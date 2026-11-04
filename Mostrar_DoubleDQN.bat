@echo off
cd /d "%~dp0"
if not exist "results\tier23_lunarlander\double\seed_42\online_final.pt" (
    echo El modelo final todavia no esta disponible. Consulta el estado en Tier3_DoubleDQN.ipynb.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m experiments.mostrar_lunarlander --group double
pause
