@echo off
echo ===========================================
echo   INICIANDO GALILEU NODE (PYTHON/GPU)
echo ===========================================
cd /d %~dp0
python setup_node.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Ambiente pronto. Iniciando servidor...
    python main.py
) else (
    echo.
    echo [ERRO] Falha no setup. Verifique se o Python esta no PATH.
    pause
)
