@echo off
rem 터널 종료
taskkill /IM cloudflared.exe /F >nul 2>&1
rem 스트림릿(파이썬) 창도 같이 닫고 싶을 때: (다른 파이썬 작업도 종료될 수 있으니 주의)
taskkill /IM python.exe /F >nul 2>&1