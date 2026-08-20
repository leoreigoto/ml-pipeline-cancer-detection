@echo off
TITLE Encerrar - MLOps UFF
echo.
echo ============================================================
echo   ENCERRANDO SERVICOS E LIMPANDO MEMORIA
echo ============================================================
echo.

:: O comando down para os containers e remove a rede virtual criada
docker-compose down

if %errorlevel% neq 0 (
    echo.
    echo [AVISO] Houve um problema ao encerrar, mas voce pode 
    echo fechar os containers manualmente pelo Docker Desktop.
) else (
    echo.
    echo ============================================================
    echo   SISTEMA ENCERRADO COM SUCESSO!
    echo ============================================================
)

echo.
pause