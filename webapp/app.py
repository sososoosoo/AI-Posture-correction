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

# ===== 정면(노트북) 변환기 =====
class FrontTransformer(VideoTransformerBase):
    """
    - final_front.py의 기능을 웹스트림 컨텍스트로 포팅
    - 버튼으로 측정/정지/캘리브레이션/리셋 제어
    - 나쁜 자세(빨간 상태)일 때 화면 우상단에 WARNING!! (화면 약 20% 크기)
    """
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            enable_segmentation=False, min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 타이머/상태
        self.timer_active = False
        self.last_ts = time.time()
        self.total = 0.0
        self.forward_neck_time = 0.0

        # 캘리브레이션 상태
        self.calibrated = False
        self.calib_chin_shoulder_dist = None
        self.calib_shoulder_width_px = None
        self._do_calibrate = False  # 버튼 신호

        # 디버그/HUD
        self.posture_score = 0.0
        self.neck_tilt = 0.0
        self._warn_cache = None  # (font_scale, thickness, text_size) 캐시

    # ---------- 외부(UI)에서 호출하는 메서드 ----------
    def start_measure(self):
        self.timer_active = True
        self.total = 0.0
        self.forward_neck_time = 0.0
        self.last_ts = time.time()

    def stop_measure(self):
        self.timer_active = False

    def request_calibration(self):
        self._do_calibrate = True  # 다음 프레임에서 조건 맞으면 수행

    def reset_all(self):
        self.timer_active = False
        self.total = 0.0
        self.forward_neck_time = 0.0
        self.calibrated = False
        self.calib_chin_shoulder_dist = None
        self.calib_shoulder_width_px = None
        self.posture_score = 0.0
        self.neck_tilt = 0.0
        self._warn_cache = None

    def get_stats(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "forward_neck": self.forward_neck_time,
            "posture_score": self.posture_score,
            "neck_tilt": self.neck_tilt,
            "calibrated": float(self.calibrated)
        }

    # ---------- 내부 유틸 ----------
    def _draw_skeleton(self, img, lms, t):
        color = _lerp_color((0, 220, 0), (0, 0, 255), t)
        for a, b in mp_pose.POSE_CONNECTIONS:
            pa, pb = lms[a], lms[b]
            if pa.visibility < 0.5 or pb.visibility < 0.5:
                continue
            xa, ya = _to_pixel((pa.x, pa.y), img.shape)
            xb, yb = _to_pixel((pb.x, pb.y), img.shape)
            cv2.line(img, (xa, ya), (xb, yb), color, 2)
        for lm in lms:
            if lm.visibility < 0.5:
                continue
            x, y = _to_pixel((lm.x, lm.y), img.shape)
            cv2.circle(img, (x, y), 3, color, -1)

    def _warning_overlay(self, img):
        """
        오른쪽 위에 'WARNING!!'를 화면 면적 약 20% 비중으로 표시.
        텍스트 높이를 프레임 높이의 0.20 근처가 되도록 폰트 스케일을 동적으로 조절.
        """
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 캐시 재사용
        if self._warn_cache is None:
            target_h = int(h * 0.20)
            scale, thickness = 1.0, max(2, h // 200)
            # 간단한 스케일 업 루프 (수십 µs 수준)
            while True:
                (tw, th), _ = cv2.getTextSize("WARNING!!", font, scale, thickness)
                if th >= target_h or scale > 20:
                    break
                scale += 0.5
            self._warn_cache = (scale, thickness, (tw, th))

        scale, thickness, (tw, th) = self._warn_cache

        # 우상단에 약간의 여백 두고 배치
        x = w - tw - int(0.03 * w)
        y = int(0.03 * h) + th

        # 텍스트 뒤에 옅은 반투명 흰 배경(가시성 향상)
        pad = int(0.02 * min(w, h))
        x0, y0 = max(0, x - pad), max(0, y - th - pad)
        x1, y1 = min(w, x + tw + pad), min(h, y + pad)
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # 빨간 텍스트
        cv2.putText(img, "WARNING!!", (x, y), font, scale, (0, 0, 255), thickness + 1, cv2.LINE_AA)

    # ---------- 프레임 처리 ----------
    def recv(self, frame: av.VideoFrame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            res = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

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
                NO = mp_pose.PoseLandmark.NOSE.value

                ls, rs = lms[LS], lms[RS]
                ml, mr = lms[ML], lms[MR]
                le, re = lms[LE], lms[RE]
                lh, rh = lms[LH], lms[RH]
                nose   = lms[NO]

                # 픽셀 좌표
                pls = _to_pixel((ls.x, ls.y), img.shape)
                prs = _to_pixel((rs.x, rs.y), img.shape)
                pml = _to_pixel((ml.x, ml.y), img.shape)
                pmr = _to_pixel((mr.x, mr.y), img.shape)

                sh_mid = ((pls[0] + prs[0]) // 2, (pls[1] + prs[1]) // 2)
                jaw_mid = ((pml[0] + pmr[0]) // 2, (pml[1] + pmr[1]) // 2)

                shoulder_w_px = max(1.0, float(np.hypot(prs[0] - pls[0], prs[1] - pls[1])))
                d_norm = float(np.hypot(jaw_mid[0] - sh_mid[0], jaw_mid[1] - sh_mid[1])) / shoulder_w_px

                # 캘리브레이션 적용: final_front.py와 동일 개념
                min_dist, max_dist = 0.08, 0.15
                if self.calibrated and self.calib_shoulder_width_px and self.calib_shoulder_width_px > 0:
                    # 현재 어깨폭/캘리브 어깨폭 비율로 이상적 거리 보정
                    scale_factor = shoulder_w_px / float(self.calib_shoulder_width_px)
                    adjusted_ideal = self.calib_chin_shoulder_dist * scale_factor
                    max_dist = adjusted_ideal
                    min_dist = adjusted_ideal * 0.7

                # posture_score (1=좋음, 0=나쁨)
                self.posture_score = max(0.0, min((d_norm - min_dist) / (max_dist - min_dist + 1e-6), 1.0))

                # 목기울임 (코-어깨중점 x오프셋)
                shoulder_mid_x = (ls.x + rs.x) / 2.0
                self.neck_tilt = nose.x - shoulder_mid_x
                max_tilt_threshold = 0.08
                tilt_bad = abs(self.neck_tilt) > (max_tilt_threshold / 2.0)

                # 라운드숄더 보조지표(색 혼합용)
                avg_sh_z = (ls.z + rs.z) / 2.0
                avg_hp_z = (lh.z + rh.z) / 2.0
                avg_el_z = (le.z + re.z) / 2.0
                depth_delta = avg_sh_z - avg_hp_z
                elbow_fwd = avg_el_z - avg_sh_z
                sev_rd = _sev_smaller_is_worse(depth_delta, ROUND_THRESH["depth_good"], ROUND_THRESH["depth_bad"])
                sev_re = _sev_smaller_is_worse(elbow_fwd, ROUND_THRESH["elbow_good"], ROUND_THRESH["elbow_bad"])
                sev_round = float(ROUND_W["depth"] * sev_rd + ROUND_W["elbow"] * sev_re)

                # 스켈레톤 색: 두 지표의 최댓값으로 혼합
                self._draw_skeleton(img, lms, max(1.0 - self.posture_score, sev_round))

                # 어깨 라인(녹->적)
                color = _lerp_color((0, 220, 0), (0, 0, 255), 1.0 - self.posture_score)
                cv2.line(img, pls, prs, color, 3)

                # 턱/어깨중점 표시
                cv2.circle(img, sh_mid, 5, (255, 255, 0), -1)
                cv2.circle(img, jaw_mid, 5, (0, 165, 255), -1)
                cv2.line(img, sh_mid, jaw_mid, (50, 200, 255), 2)

                # 라벨
                forward_head = (self.posture_score < 0.9)
                rounded = (sev_round >= 0.6)
                bad_posture = forward_head or tilt_bad or rounded
                label = []
                if forward_head: label.append("Forward Head")
                if tilt_bad:     label.append("Neck Tilt")
                if rounded:      label.append("Rounded Shoulders")
                if not label:    label.append("Good (Front)")
                cv2.putText(img, " | ".join(label), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255) if bad_posture else (0, 180, 0), 2, cv2.LINE_AA)

                # ----- 캘리브레이션 요청 처리 -----
                if self._do_calibrate:
                    # 얼굴/어깨 가시성 충분할 때만 채택
                    if all(x.visibility > 0.7 for x in [ls, rs, ml, mr]):
                        self.calib_chin_shoulder_dist = abs(((ls.y + rs.y) / 2.0) - ((ml.y + mr.y) / 2.0))
                        self.calib_shoulder_width_px = shoulder_w_px
                        self.calibrated = True
                        self._do_calibrate = False
                    else:
                        # 다음 프레임에서 다시 시도
                        pass

                # ----- 타이머 누적 -----
                now = time.time()
                inc = now - self.last_ts
                self.last_ts = now
                if self.timer_active:
                    self.total += inc
                    if bad_posture:
                        self.forward_neck_time += inc

                # WARNING 오버레이 (빨간 상태일 때)
                if bad_posture:
                    self._warning_overlay(img)

                # HUD
                cv2.putText(img,
                            f"Bad: {self.forward_neck_time:5.1f}s | Total: {self.total:5.1f}s | Calib: {'Y' if self.calibrated else 'N'}",
                            (20, img.shape[0]-22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            else:
                # 사람 미검출: 시간 갱신만
                now = time.time()
                inc = now - self.last_ts
                self.last_ts = now
                if self.timer_active:
                    self.total += inc
                cv2.putText(img, "No person detected", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"ERROR: {type(e).__name__}: {e}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===== 측면(모바일) 변환기 (간단/안정) =====
def _angle_3pt(a, b, c):
    ba = a - b; bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

class SideTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                 enable_segmentation=False, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
    def recv(self, frame: av.VideoFrame):
        try:
            img = frame.to_ndarray(format="bgr24")
            res = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                ids = mp_pose.PoseLandmark
                cand = [
                    (ids.LEFT_EAR.value, ids.LEFT_SHOULDER.value, ids.LEFT_HIP.value),
                    (ids.RIGHT_EAR.value, ids.RIGHT_SHOULDER.value, ids.RIGHT_HIP.value)
                ]
                best, best_vis = None, -1
                for a, b, c in cand:
                    vis = lms[a].visibility + lms[b].visibility + lms[c].visibility
                    if vis > best_vis:
                        best_vis = vis; best = (a, b, c)
                if best:
                    a, b, c = best
                    def p(i): return np.array(_to_pixel((lms[i].x, lms[i].y), img.shape), dtype=np.float32)
                    A, B, C = p(a), p(b), p(c)
                    deg = _angle_3pt(A, B, C)
                    if deg >= 160: color = (0, 200, 0); label = f"Good {deg:.1f}°"
                    elif deg >= 145: color = (0, 165, 255); label = f"Warn {deg:.1f}°"
                    else: color = (0, 0, 255); label = f"Bad {deg:.1f}°"
                    cv2.line(img, tuple(A.astype(int)), tuple(B.astype(int)), color, 3)
                    cv2.line(img, tuple(B.astype(int)), tuple(C.astype(int)), color, 3)
                    for P in (A, B, C): cv2.circle(img, tuple(P.astype(int)), 5, color, -1)
                    cv2.putText(img, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"ERROR: {type(e).__name__}: {e}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===== Streamlit UI =====
st.set_page_config(page_title="PoseGuard Web", layout="wide")
st.title("PoseGuard — 웹앱(정면/측면)")

tabs = st.tabs(["정면(노트북)", "측면(모바일)"])

rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- 정면 탭 ---
with tabs[0]:
    st.markdown("노트북 브라우저에서 실행하세요. **Start**를 누르면 노트북 카메라를 사용합니다.")
    ctx_front = webrtc_streamer(
        key="front",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=FrontTransformer,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        async_processing=True,
    )

    # 제어 버튼(스트림 시작 후에만 작동)
    col1, col2, col3, col4 = st.columns(4)
    if ctx_front and ctx_front.video_transformer:
        if col1.button("측정 시작"):
            ctx_front.video_transformer.start_measure()
        if col2.button("측정 정지"):
            ctx_front.video_transformer.stop_measure()
        if col3.button("캘리브레이션"):
            ctx_front.video_transformer.request_calibration()
        if col4.button("리셋"):
            ctx_front.video_transformer.reset_all()

        # 간단한 지표 표시
        stats = ctx_front.video_transformer.get_stats()
        st.caption(
            f"**측정 시간**: {stats['total']:.1f}s  |  "
            f"**나쁜 자세 누적**: {stats['forward_neck']:.1f}s  |  "
            f"**Posture Score**: {stats['posture_score']:.2f}  |  "
            f"**Neck Tilt**: {stats['neck_tilt']:.3f}  |  "
            f"**Calibrated**: {'Yes' if stats['calibrated'] else 'No'}"
        )
    else:
        st.info("스트림을 먼저 **Start** 해주세요.")

# --- 측면 탭 ---
with tabs[1]:
    st.markdown("휴대폰 브라우저에서 접속해서 **Start**를 누르세요. 모바일 카메라가 서버로 전송됩니다.")
    webrtc_streamer(
        key="side",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SideTransformer,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )
