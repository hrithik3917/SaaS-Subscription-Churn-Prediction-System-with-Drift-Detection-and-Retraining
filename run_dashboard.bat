@echo off
REM ============================================================
REM Churn Dashboard Launcher
REM Sets PYTHONPATH and starts the Streamlit dashboard
REM ============================================================

set PYTHONPATH=%~dp0
call venv\Scripts\activate
streamlit run src\dashboard\app.py