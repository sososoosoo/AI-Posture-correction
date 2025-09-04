@echo off
cd /d C:\AI_solution\camera
call .venv\Scripts\activate
start "" cmd /k "cd /d C:\AI_solution\camera && streamlit run webapp\app.py --server.address 0.0.0.0 --server.port 8501"
timeout /t 5 /nobreak >nul
start "" cmd /k "cloudflared tunnel run posture-correction"
