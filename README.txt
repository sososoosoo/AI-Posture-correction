도메인 주소 https://app.boaa-posture.store/

.\.venv\Scripts\activate
터미널1: streamlit run webapp/app.py --server.address 0.0.0.0 --server.port 8501
터미널2: cloudflared tunnel run posture-correction