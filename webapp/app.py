# [파일] C:\AI_solution\camera\webapp\app.py
# --------------------------------------------------------------------
# Streamlit + streamlit-webrtc 기반 웹앱
# - 탭1: 정면(노트북) -> Mediapipe 기반 거북목 + 라운드숄더 지표
# - 탭2: 측면(모바일) -> Mediapipe 기반 측면 각도 간단 판단(데모 안정)
# 필요 패키지: streamlit, streamlit-webrtc, av, mediapipe, opencv-python, numpy
# --------------------------------------------------------------------
import sys
from pathlib import Path
import time
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import mediapipe as mp

# PyInstaller 대비 (향후 패키징 시)
BASE_DIR = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent)).parent  # /webapp/ 상위 = 프로젝트 루트

mp_pose = mp.solutions.pose

# ===== 튜닝 파라미터(정면) =====
DIST_THRESH = {"good": 0.54, "bad": 0.46}
ROUND_THRESH = {"depth_good": -0.010, "depth_bad": -0.045,
                "elbow_good": -0.010, "elbow_bad": -0.055}
ROUND_W = {"depth": 0.7, "elbow": 0.3}

def _to_pixel(xy, shape):
    x, y = xy
    h, w = shape[:2]
    return int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))

def _sev_smaller_is_worse(value, good, bad):
    if value >= good: return 0.0
    if value <= bad:  return 1.0
    return float((good - value) / (good - bad))

def _lerp_color(cg, cr, t):
    t = float(np.clip(t, 0, 1)); return tuple(int(cg[i] + (cr[i]-cg[i])*t) for i in range(3))

def _draw_skeleton(frame, lms, t):
    color = _lerp_color((0,220,0), (0,0,255), t)
    for a, b in mp_pose.POSE_CONNECTIONS:
        pa, pb = lms[a], lms[b]
        if pa.visibility < 0.5 or pb.visibility < 0.5: continue
        xa, ya = _to_pixel((pa.x, pa.y), frame.shape)
        xb, yb = _to_pixel((pb.x, pb.y), frame.shape)
        cv2.line(frame, (xa, ya), (xb, yb), color, 3)
    for lm in lms:
        if lm.visibility < 0.5: continue
        x, y = _to_pixel((lm.x, lm.y), frame.shape)
        cv2.circle(frame, (x, y), 4, color, -1)

# ================== 정면(노트북)용 변환기 ==================
class FrontTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                 enable_segmentation=False, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
        self.last_ts = time.time()
        self.total = 0.0
        self.bad_time = 0.0
        self.fh_time = 0.0

    def recv(self, frame: av.VideoFrame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            res = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            sev_head = 0.0
            sev_round = 0.0
            forward_head = False
            rounded = False
            bad_posture = False

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                ML = mp_pose.PoseLandmark.MOUTH_LEFT.value
                MR = mp_pose.PoseLandmark.MOUTH_RIGHT.value
                LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
                RE = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                LH = mp_pose.PoseLandmark.LEFT_HIP.value
                RH = mp_pose.PoseLandmark.RIGHT_HIP.value

                ls, rs = lms[LS], lms[RS]
                ml, mr = lms[ML], lms[MR]
                le, re = lms[LE], lms[RE]
                lh, rh = lms[LH], lms[RH]

                # forward-head: 어깨중점-턱중점 / 어깨폭
                pls = _to_pixel((ls.x, ls.y), img.shape); prs = _to_pixel((rs.x, rs.y), img.shape)
                pml = _to_pixel((ml.x, ml.y), img.shape); pmr = _to_pixel((mr.x, mr.y), img.shape)
                sh_mid = ((pls[0]+prs[0])//2, (pls[1]+prs[1])//2)
                jaw_mid = ((pml[0]+pmr[0])//2, (pml[1]+pmr[1])//2)
                shoulder_w = max(1.0, float(np.hypot(prs[0]-pls[0], prs[1]-pls[1])))
                d_norm = float(np.hypot(jaw_mid[0]-sh_mid[0], jaw_mid[1]-sh_mid[1])) / shoulder_w

                if   d_norm >= DIST_THRESH["good"]: sev_head = 0.0
                elif d_norm <= DIST_THRESH["bad"]:  sev_head = 1.0
                else:
                    sev_head = (DIST_THRESH["good"] - d_norm) / (DIST_THRESH["good"] - DIST_THRESH["bad"])

                cv2.circle(img, sh_mid, 6, (255,255,0), -1)
                cv2.circle(img, jaw_mid, 6, (0,165,255), -1)
                cv2.line(img, sh_mid, jaw_mid, (50,200,255), 2)
                cv2.putText(img, f"norm-dist={d_norm:.3f}", (20,70), cv2.FONT_HERSHEY_PLAIN, 1.4, (255,255,255), 1, cv2.LINE_AA)

                # rounded-shoulder: z-깊이
                avg_sh_z = (ls.z + rs.z)/2.0
                avg_hp_z = (lh.z + rh.z)/2.0
                avg_el_z = (le.z + re.z)/2.0
                depth_delta = avg_sh_z - avg_hp_z
                elbow_fwd   = avg_el_z - avg_sh_z

                sev_rd = _sev_smaller_is_worse(depth_delta, ROUND_THRESH["depth_good"], ROUND_THRESH["depth_bad"])
                sev_re = _sev_smaller_is_worse(elbow_fwd,   ROUND_THRESH["elbow_good"], ROUND_THRESH["elbow_bad"])
                sev_round = float(ROUND_W["depth"]*sev_rd + ROUND_W["elbow"]*sev_re)

                # draw skeleton with max severity
                _draw_skeleton(img, lms, max(sev_head, sev_round))

                forward_head = (sev_head  >= 0.6)
                rounded      = (sev_round >= 0.6)
                bad_posture  = forward_head or rounded

                label = []
                if forward_head: label.append("Forward Head")
                if rounded:      label.append("Rounded Shoulders")
                if not label:    label.append("Good (Front)")
                color = (0,0,255) if bad_posture else (0,200,0)
                cv2.putText(img, " | ".join(label), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # timers (스트리밍 중 항상 카운트)
            now = time.time()
            inc = now - self.last_ts
            self.last_ts = now
            self.total += inc
            if bad_posture: self.bad_time += inc
            if forward_head: self.fh_time += inc

            # HUD timers
            cv2.putText(img, f"Bad: {self.bad_time:5.1f}s  |  Forward: {self.fh_time:5.1f}s  |  Total: {self.total:5.1f}s",
                        (20, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            # 에러가 나도 세션이 죽지 않게 안전 처리
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"ERROR: {type(e).__name__}: {e}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================== 측면(모바일)용 변환기 ==================
def _angle_3pt(a, b, c):
    # 각도: a-b-c (도 단위)
    ba = a - b; bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

class SideTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                 enable_segmentation=False, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    def recv(self, frame: av.VideoFrame):
        try:
            img = frame.to_ndarray(format="bgr24")
            # 모바일 전/후면 카메라 방향 차이 고려: 필요 시 flip 조정
            # img = cv2.flip(img, 1)

            res = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                # 측면 간단 지표: 귀-어깨-엉덩이 각도(목/상체)로 Good/Bad
                # 좌/우 중 보이는 쪽을 선택(visibility 높은 쪽)
                ids = mp_pose.PoseLandmark
                cand = [
                    (ids.LEFT_EAR.value, ids.LEFT_SHOULDER.value, ids.LEFT_HIP.value),
                    (ids.RIGHT_EAR.value, ids.RIGHT_SHOULDER.value, ids.RIGHT_HIP.value)
                ]
                best = None; best_vis = -1
                for a,b,c in cand:
                    vis = lms[a].visibility + lms[b].visibility + lms[c].visibility
                    if vis > best_vis:
                        best_vis = vis; best = (a,b,c)

                if best is not None:
                    a,b,c = best
                    def p(i): return np.array(_to_pixel((lms[i].x, lms[i].y), img.shape), dtype=np.float32)
                    A, B, C = p(a), p(b), p(c)
                    deg = _angle_3pt(A,B,C)
                    # 대충: 160°~180° 좋음 / 140°~160° 주의 / <140° 나쁨
                    if deg >= 160: color = (0,200,0); label = f"Good angle {deg:.1f}°"
                    elif deg >= 145: color = (0,165,255); label = f"Warn {deg:.1f}°"
                    else: color = (0,0,255); label = f"Bad {deg:.1f}°"

                    cv2.line(img, tuple(A.astype(int)), tuple(B.astype(int)), color, 3)
                    cv2.line(img, tuple(B.astype(int)), tuple(C.astype(int)), color, 3)
                    for P in (A,B,C): cv2.circle(img, tuple(P.astype(int)), 5, color, -1)
                    cv2.putText(img, label, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"ERROR: {type(e).__name__}: {e}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================== Streamlit UI ==================
st.set_page_config(page_title="PoseGuard Web", layout="wide")
st.title("PoseGuard — 웹앱(정면/측면)")

tabs = st.tabs(["정면(노트북)", "측면(모바일)"])

rtc_cfg = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

with tabs[0]:
    st.markdown("노트북 브라우저에서 실행하세요. Start를 누르면 노트북 카메라를 사용합니다.")
    webrtc_streamer(
        key="front",
        mode=WebRtcMode.SENDRECV,  # 양방향: 브라우저→서버 전송 + 처리 결과 표시
        video_transformer_factory=FrontTransformer,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        async_processing=True,  # (권장) 서버 측 처리 프레임 드랍 줄이기
    )

with tabs[1]:
    st.markdown("휴대폰 브라우저에서 접속해서 Start를 누르세요. 모바일 카메라가 서버로 전송됩니다.")
    webrtc_streamer(
        key="side",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SideTransformer,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )
