# [파일] webapp/app.py
# --------------------------------------------------------------------
# Streamlit + streamlit-webrtc 기반 웹앱
# - 탭1: 정면(노트북) -> final_front.py의 UI/로직을 웹스트림에 맞게 이식
# - 탭2: 측면(모바일) -> Mediapipe 기반 간단 각도 판정(데모 안정)
# --------------------------------------------------------------------
import time
from typing import Tuple, Dict

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# [추가: base64 로더 & 에셋 경로]
import base64
from pathlib import Path
import sys, importlib  # (기존에 있다면 중복 import 제거)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSETS = ROOT / "webapp" / "assets"
LOGO_PATH = ASSETS / "logo.png"
BG_PATH = ASSETS / "sidebar_bg.jpg"

def _b64(path: Path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

LOGO_B64 = _b64(LOGO_PATH)
BG_B64 = _b64(BG_PATH)

# ===== 공통 =====
mp_pose = mp.solutions.pose

# 정면용 튜닝(최신 final_front.py 로직과 합치)
DIST_THRESH = {"good": 0.54, "bad": 0.46}  # 어깨-턱 중점 거리(어깨폭 정규화)
ROUND_THRESH = {  # (정면 라운드숄더 심각도 계산에 쓰는 기본값. 웹앱에서는 시각화 중심)
    "depth_good": -0.010, "depth_bad": -0.045,
    "elbow_good": -0.010, "elbow_bad": -0.055
}
ROUND_W = {"depth": 0.7, "elbow": 0.3}

def _to_pixel(xy: Tuple[float, float], shape):
    x, y = xy
    h, w = shape[:2]
    return int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))

def _sev_smaller_is_worse(value, good, bad):
    if value >= good: return 0.0
    if value <= bad:  return 1.0
    return float((good - value) / (good - bad))

def _lerp_color(cg, cr, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(cg[i] + (cr[i] - cg[i]) * t) for i in range(3))



# ===== 측면(모바일) 변환기 (간단/안정) =====
def _angle_3pt(a, b, c):
    ba = a - b; bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

# === (교체) final_front / final_side를 코어로 불러와 쓰는 래퍼들 ===
def make_front_transformer():
    ff = importlib.import_module("final_front")
    ff = importlib.reload(ff)  # 새로고침 시 최신 반영
    FrontCore = getattr(ff, "FrontCore")  # 위 1)에서 추가한 클래스

    class FrontTransformer(VideoTransformerBase):
        def __init__(self):
            self.core = FrontCore()

        # UI에서 버튼으로 호출할 메서드들
        def start_measure(self): self.core.start_timer()
        def stop_measure(self):  self.core.stop_timer()
        def reset_all(self):     self.core.reset_all()
        def request_calibration(self): self.core.request_calibration()
        def get_stats(self):     return self.core.get_stats()

        def recv(self, frame: av.VideoFrame):
            img = frame.to_ndarray(format="bgr24")
            out = self.core.process_frame(img)
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    return FrontTransformer


def make_side_transformer():
    fs = importlib.import_module("final_side")
    fs = importlib.reload(fs)  # 새로고침 시 최신 반영
    SideCore = getattr(fs, "SideCore")  # 위 2)에서 추가한 클래스

    class SideTransformer(VideoTransformerBase):
        def __init__(self):
            self.core = SideCore()

        def start_measure(self): self.core.start_timer()
        def stop_measure(self):  self.core.stop_timer()
        def reset_all(self):     self.core.reset_all()
        def get_stats(self):     return self.core.get_stats()

        def recv(self, frame: av.VideoFrame):
            img = frame.to_ndarray(format="bgr24")
            out = self.core.process_frame(img)
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    return SideTransformer

# ===== Streamlit UI =====
st.set_page_config(
    page_title="PoseGuard Web",
    layout="wide",
    initial_sidebar_state="expanded",
)
rtc_cfg = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.markdown("""
<style>
/* 히어로 높이/배경 편차 */
[data-testid="stSidebar"] .hero {
  text-align:center; padding: 24px 10px 12px;
  min-height: 230px;                       /* 높이 살짝 확보 */
}

/* 타이틀 약간 더 크고, 자간/그림자 */
[data-testid="stSidebar"] h1 {
  font-size: 30px; margin: 4px 0 4px; letter-spacing:.3px;
  text-shadow: 0 2px 8px rgba(0,0,0,.25);
}

/* 서브타이틀 살짝 강조 */
[data-testid="stSidebar"] .sub {
  font-size: 13.5px; opacity:.95; margin-bottom: 12px; letter-spacing:.2px;
}

/* 링크를 ‘버튼’처럼 */
[data-testid="stSidebar"] .links a {
  display:block; width:100%;
  padding:9px 12px; margin:7px 0;
  border-radius:12px; text-decoration:none;
  background: rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.15);
}
[data-testid="stSidebar"] .links a:hover {
  background: rgba(255,255,255,.16);
  border-color: rgba(255,255,255,.32);
}

/* 카드형 라디오 테두리/호버 보정 */
[data-testid="stSidebar"] div[role="radiogroup"] > label {
  display:block; padding:14px 14px;
  border:1px solid rgba(255,255,255,.20);
  border-radius:14px; margin:10px 0; cursor:pointer;
  background: rgba(255,255,255,.07);
  backdrop-filter: blur(3px);
  transition: all .15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
  border-color: rgba(255,255,255,.38);
  background: rgba(255,255,255,.14);
}
</style>
""", unsafe_allow_html=True)

# [추가: 사이드바 렌더 함수]
def render_sidebar() -> str:
    # 히어로 영역(로고/타이틀/서브타이틀/링크)
    hero_html = f"""
    <div class='hero'>
      {f'<img class="logo" src="data:image/png;base64,{LOGO_B64}" />' if LOGO_B64 else '<div class="logo-fallback">PG</div>'}
      <h1>자세 어때?</h1>
      <div class='sub'>AI / Posture Coach</div>
    </div>
    """
    st.sidebar.markdown(hero_html, unsafe_allow_html=True)

    # 모드 라디오(카드처럼 보이도록 CSS로 스타일링)
    mode = st.sidebar.radio(
        "모드 선택",
        ["정면 자세 교정", "측면 자세 교정"],
        index=0,
        label_visibility="collapsed",
        key="mode_radio"
    )

    # 하단 소셜 아이콘 영역(원하면 링크 바꿔도 됨)
    st.sidebar.markdown(
        """
        <div class="sidebar-bottom">
          <a href="https://x.com" target="_blank">𝕏</a>
          <a href="https://github.com" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # CSS 인젝션
    bg_css = (
        f'linear-gradient(180deg, rgba(0,0,0,.40), rgba(0,0,0,.55)), url("data:image/jpeg;base64,{BG_B64}") no-repeat center/cover'
        if BG_B64 else
        'linear-gradient(180deg,#0f172a,#1f2937)'
    )
    st.markdown(
        f"""
        <style>
          /* 사이드바 배경/글자색 */
          [data-testid="stSidebar"] {{
            background: {bg_css};
            color: #f3f4f6;
          }}
          [data-testid="stSidebar"] * {{
            color: #f3f4f6 !important;
          }}

          /* 히어로 */
          [data-testid="stSidebar"] .hero {{
            text-align:center; padding: 18px 8px 6px;
          }}
          [data-testid="stSidebar"] .logo {{
            width:86px; height:86px; border-radius:50%;
            box-shadow: 0 10px 24px rgba(0,0,0,.35);
            margin-bottom: 8px;
            object-fit: cover;
          }}
          [data-testid="stSidebar"] .logo-fallback {{
            width:86px;height:86px;border-radius:50%;
            background:rgba(255,255,255,.16);
            display:flex;align-items:center;justify-content:center;
            font-weight:700;font-size:28px;letter-spacing:.3px;
            margin: 0 auto 8px;
          }}
          [data-testid="stSidebar"] h1 {{
            font-size: 28px; margin: 2px 0 2px; letter-spacing:.2px;
          }}
          [data-testid="stSidebar"] .sub {{
            font-size: 13px; opacity:.92; margin-bottom: 10px;
          }}
          [data-testid="stSidebar"] .links a {{
            display:block; width:100%;
            padding:8px 10px; margin:6px 0;
            border-radius:12px; text-decoration:none;
            background: rgba(255,255,255,.07);
          }}
          [data-testid="stSidebar"] .links a:hover {{
            background: rgba(255,255,255,.14);
          }}

          /* 라디오 → 카드 스타일 */
          [data-testid="stSidebar"] div[role="radiogroup"] > label {{
            display:block; padding:14px 14px;
            border:1px solid rgba(255,255,255,.18);
            border-radius:14px; margin:10px 0; cursor:pointer;
            background: rgba(255,255,255,.07);
            backdrop-filter: blur(2px);
          }}
          [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
            border-color: rgba(255,255,255,.35);
            background: rgba(255,255,255,.12);
          }}
          /* 선택 표시(●/○) */
          [data-testid="stSidebar"] div[role="radiogroup"] > label::before {{
            content: "○"; margin-right:10px; font-size:18px; vertical-align:middle; color: #d1d5db;
          }}
          [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {{
            border-color:#fb7185; background: rgba(251,113,133,.18);
          }}
          [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"]::before {{
            content: "●"; color:#ef4444;
          }}
          /* 기본 라벨 숨김 */
          [data-testid="stSidebar"] > div:has(> div[role="radiogroup"]) > label {{
            display:none;
          }}

          /* 하단 고정 아이콘 바 */
          [data-testid="stSidebar"] .sidebar-bottom {{
            position: fixed; bottom: 12px; left: 12px; right: 12px;
            display:flex; justify-content:center; gap:16px; opacity:.95;
          }}
          [data-testid="stSidebar"] .sidebar-bottom a {{
            text-decoration:none; padding:8px 12px; border-radius:10px;
            background: rgba(255,255,255,.10);
          }}
          [data-testid="stSidebar"] .sidebar-bottom a:hover {{
            background: rgba(255,255,255,.18);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return mode

mode = render_sidebar()   # ← 기존 st.sidebar.radio 블록 전체 대체

# -------------------------------
# 정면 / 측면 페이지 라우팅
# -------------------------------
if mode == "정면 자세 교정":
    st.markdown("***정면 자세***\n\n"
                "**Start**를 누르면 카메라가 실행됩니다.")
    ctx_front = webrtc_streamer(
        key="front",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=make_front_transformer(),
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        async_processing=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    if ctx_front and ctx_front.video_transformer:
        if col1.button("측정 시작"):   ctx_front.video_transformer.start_measure()
        if col2.button("측정 정지"):   ctx_front.video_transformer.stop_measure()
        if col3.button("캘리브레이션"): ctx_front.video_transformer.request_calibration()
        if col4.button("리셋"):       ctx_front.video_transformer.reset_all()

        stats = ctx_front.video_transformer.get_stats()
        st.caption(
            f"**측정 시간**: {stats['total']:.1f}s  |  "
            f"**나쁜 자세 누적**: {stats['forward_neck']:.1f}s  |  "
            f"**Posture Score**: {stats['posture_score']:.2f}  |  "
            f"**Neck Tilt**: {stats['neck_tilt']:.3f}  |  "
            f"**Calibrated**: {'Yes' if stats['calibrated'] else 'No'}"
        )
    else:
        st.info("**Start** 버튼을 먼저 눌러주세요.")

elif mode == "측면 자세 교정":
    st.markdown("***측면 자세***\n\n"
                "**Start**를 누르면 카메라가 실행됩니다.")
    ctx_side = webrtc_streamer(
        key="side",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=make_side_transformer(),
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )

    c1, c2, c3 = st.columns(3)
    if ctx_side and ctx_side.video_transformer:
        if c1.button("측정 시작(측면)"): ctx_side.video_transformer.start_measure()
        if c2.button("측정 정지(측면)"): ctx_side.video_transformer.stop_measure()
        if c3.button("리셋(측면)"):     ctx_side.video_transformer.reset_all()
        st.caption(ctx_side.video_transformer.get_stats())
    else:
        st.info("**Start** 버튼을 먼저 눌러주세요.")