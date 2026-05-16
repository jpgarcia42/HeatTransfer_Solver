@echo off
title Servidor Django – TransCal CoolProp API
echo ============================================
echo   INICIANDO SERVIDOR DJANGO – CoolProp API
echo ============================================
echo.
echo Acesse o app em: C:\Users\ASUS\Documents\2026\TransCal\index.html
echo API rodando em:  http://127.0.0.1:8000/props/
echo.
echo Pressione Ctrl+C para parar o servidor.
echo.

cd /d "%~dp0backend"

:: Instala Django se ainda não estiver instalado
pip show django >nul 2>&1 || pip install django

python manage.py runserver 127.0.0.1:8000
pause
