@echo off
TITLE Painel de Controle MLOps - Termografia UFF
echo.
echo ============================================================
echo   INICIALIZANDO ECOSSISTEMA MLOPS (MESTRADO UFF)
echo ============================================================
echo.

:: 1. Verifica se o Docker esta rodando
docker stats --no-stream >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] O Docker Desktop nao parece estar aberto.
    echo Por favor, abra o Docker Desktop e tente novamente.
    echo.
    pause
    exit
)

:: 2. Sobe os containers
echo [1/4] Subindo containers (Gradio, API, MLflow, Airflow)...
docker-compose up --build -d --remove-orphans

if %errorlevel% neq 0 (
    echo.
    echo [ERRO] Falha ao iniciar os containers.
    pause
    exit
)

echo.
echo [2/4] Aguardando 15 segundos para estabilizacao dos servicos...
timeout /t 20 /nobreak >nul

:: 3. Abre as paginas no navegador
echo [3/4] Abrindo interfaces de gerenciamento...

:: Interface do Usuario (Gradio)
start http://localhost:7860

:: Rastreamento de Experimentos (MLflow)
timeout /t 2 /nobreak >nul
start http://localhost:5000

:: Orquestracao de Pipelines (Airflow)
timeout /t 2 /nobreak >nul
start http://localhost:8080

echo.
echo [4/4] TUDO PRONTO!
echo.
echo ------------------------------------------------------------
echo  - Portas Ativas:
echo    * Interface Gradio: 7860
echo    * Painel MLflow:    5000
echo    * Painel Airflow:   8080
echo ------------------------------------------------------------
echo.
echo Pode minimizar esta janela. 
echo Para encerrar os servicos, feche o Docker Desktop.
echo.
pause