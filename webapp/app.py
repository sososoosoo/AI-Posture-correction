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

# ===== 공통 =====
mp_pose = mp.solutions.pose

# 정면용 튜닝(최신 final_front.py 로직과 합치)
DIST_THRESH = {"good": 0.54, "bad": 0.46}  # 어깨-턱 중점 거리(어깨폭 정규화)
ROUND_THRESH = {  # (정면 라운드숄더 심각도 계산에 쓰는 기본값. 웹앱에서는 시각화 중심)
    "depth_good": -0.010, "depth_bad": -0.045,
    "elbow_good": -0.010, "elbow_bad": -0.055
}
ROUND_W = {"depth": 0.7, "elbow": 0.3}

# === (추가) 프로젝트 루트 경로 및 리로드 도우미 ===
from pathlib import Path
import sys, importlib

ROOT = Path(__file__).resolve().parent.parent  # 프로젝트 루트(= final_*.py 위치)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.sidebar.caption(f"Loaded webapp: {__file__}")
st.sidebar.caption(f"cwd: {Path.cwd()}")

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
st.set_page_config(page_title="PoseGuard Web", layout="wide")
st.title("자세 어때?")

tabs = st.tabs(["정면 자세 교정", "측면 자세 교정"])

rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- 정면 탭 ---
with tabs[0]:
    st.markdown("**Start**를 누르면 카메라가 실행됩니다.")
    ctx_front = webrtc_streamer(
        key="front",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=make_front_transformer(),  # << 변경
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


# --- 측면 탭 ---
with tabs[1]:
    st.markdown("**Start**를 누르면 카메라가 실행됩니다.")
    ctx_side = webrtc_streamer(
        key="side",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=make_side_transformer(),  # << 변경
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