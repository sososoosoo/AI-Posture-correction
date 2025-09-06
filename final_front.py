import cv2
import mediapipe as mp
import time
import collections
import threading
import numpy as np

# MediaPipe Pose 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 캘리브레이션 관련 전역 변수 ---
calibrated = False
calibrated_chin_shoulder_dist = None
calibrated_shoulder_width_px = None
calibrated_neck_tilt_offset = 0.0
calibration_feedback_text = ""
calibration_feedback_time = 0

# ===== (수정) 알람 사운드 유틸 (pygame 사용) =====
import os
import pygame

# 알람 파일 탐색: 프로젝트/웹앱 자주 쓰는 위치 3곳을 순서대로 찾음
_ALARM_CANDIDATES = [
    "alarm.wav", "alarm.mp3",
    os.path.join("assets", "alarm.wav"),
    os.path.join("assets", "alarm.mp3"),
    os.path.join("webapp", "assets", "alarm.wav"),
    os.path.join("webapp", "assets", "alarm.mp3"),
]
def _find_alarm_path():
    for p in _ALARM_CANDIDATES:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

class AlarmPlayer:
    """
    - pygame.mixer를 사용하여 사운드 재생 제어
    - MP3, WAV 등 다양한 포맷 지원
    - 무한 반복 및 중간 정지 기능
    """
    def __init__(self, sound_path: str | None):
        self.sound_path = sound_path
        self._is_initialized = False
        if not sound_path:
            print("알람 파일 경로가 없습니다.")
            return

        try:
            pygame.mixer.init()
            pygame.mixer.music.load(sound_path)
            self._is_initialized = True
            print(f"알람 파일 '{sound_path}' 로드 성공.")
        except Exception as e:
            print(f"pygame mixer 초기화 또는 사운드 파일 로드 실패: {e}")
            self.sound_path = None # 에러 발생 시 재생 시도 방지

    def start(self):
        """알람을 무한 반복으로 재생합니다."""
        if not self._is_initialized or pygame.mixer.music.get_busy():
            return
        try:
            pygame.mixer.music.play(loops=-1)
        except Exception as e:
            print(f"알람 재생 실패: {e}")

    def stop(self):
        """알람 재생을 즉시 중지합니다."""
        if not self._is_initialized:
            return
        try:
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"알람 중지 실패: {e}")

    def quit(self):
        """pygame.mixer를 종료합니다."""
        if self._is_initialized:
            pygame.mixer.quit()

# VideoPlayer 클래스는 기존과 동일합니다.
class VideoPlayer:
    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0, width=960, height=540):
        self.cv2 = cv2
        self.__cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.__cap.isOpened():
            raise RuntimeError(f"Cannot open {'camera' if isinstance(source, int) else ''} {source}")
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
        self.__input_fps = self.__cap.get(cv2.CAP_PROP_FPS)
        if self.__input_fps <= 0:
            self.__input_fps = 60
        self.__output_fps = fps if fps is not None else self.__input_fps
        self.__flip = flip
        self.__size = None
        self.__interpolation = None
        if size is not None:
            self.__size = size
            self.__interpolation = cv2.INTER_AREA if size[0] < self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH) else cv2.INTER_LINEAR
        _, self.__frame = self.__cap.read()
        self.__lock = threading.Lock()
        self.__thread = None
        self.__stop = False

    def start(self):
        self.__stop = False
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def stop(self):
        self.__stop = True
        if self.__thread is not None:
            self.__thread.join()
        self.__cap.release()

    def __run(self):
        prev_time = 0
        while not self.__stop:
            t1 = time.time()
            ret, frame = self.__cap.read()
            if not ret:
                break
            if 1 / self.__output_fps < time.time() - prev_time:
                prev_time = time.time()
                with self.__lock:
                    self.__frame = frame
            t2 = time.time()
            wait_time = 1 / self.__input_fps - (t2 - t1)
            time.sleep(max(0, wait_time))
        self.__frame = None

    def next(self):
        with self.__lock:
            if self.__frame is None:
                return None
            frame = self.__frame.copy()
        if self.__size is not None:
            frame = self.cv2.resize(frame, self.__size, interpolation=self.__interpolation)
        if self.__flip:
            frame = self.cv2.flip(frame, 1)
        return frame

# PostureTimer 클래스는 기존과 동일합니다.
class PostureTimer:
    def __init__(self, duration):
        self.start_time = None
        self.duration = duration
        self.timer_active = False
        self.forward_neck_time = 0.0
        self.total_time = 0.0

    def start_timer(self):
        self.start_time = time.time()
        self.timer_active = True
        self.forward_neck_time = 0.0
        self.total_time = 0.0

    def stop_timer(self):
        self.timer_active = False
        self.total_time = time.time() - self.start_time if self.start_time else 0
        return {
            "forward_neck_time": self.forward_neck_time,
            "total_time": self.total_time
        }

    def add_forward_neck_time(self, duration):
        if self.timer_active:
            self.forward_neck_time += duration

# --- UI 및 타이머 상태 변수 ---
timer_running = False
show_results = False
posture_timer = PostureTimer(duration=1800)
posture_score = 0.0
neck_tilt = 0.0

# --- UI 상수 정의 ---
FRAME_WIDTH = 960
PANEL_WIDTH = 350
BUTTON_X_OFFSET = 20  # UI 패널 내부의 X좌표 오프셋
BUTTON_Y_START = 50
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 60
BUTTON_SPACING = 80
# 마우스 클릭 감지를 위한 실제 창 기준 X좌표
CLICK_BUTTON_X_START = FRAME_WIDTH + BUTTON_X_OFFSET

def on_mouse_click(event, x, y, flags, param):
    global timer_running, show_results
    if event == cv2.EVENT_LBUTTONDOWN:
        # Wrap conditions in parentheses to avoid using backslash
        start_condition = (
            CLICK_BUTTON_X_START <= x <= CLICK_BUTTON_X_START + BUTTON_WIDTH and
            BUTTON_Y_START <= y <= BUTTON_Y_START + BUTTON_HEIGHT and
            not timer_running
        )
        stop_condition = (
            CLICK_BUTTON_X_START <= x <= CLICK_BUTTON_X_START + BUTTON_WIDTH and
            (BUTTON_Y_START + BUTTON_SPACING) <= y <= (BUTTON_Y_START + BUTTON_SPACING + BUTTON_HEIGHT) and
            timer_running
        )

        if start_condition:
            posture_timer.start_timer()
            timer_running = True
            show_results = False
            print("타이머 시작")
        elif stop_condition:
            posture_timer.stop_timer()
            timer_running = False
            show_results = True
            print("타이머 종료. 분석 결과: ", posture_timer.stop_timer())


def format_time(seconds):
    """초를 '00h 00m 00s' 형식의 문자열로 변환합니다."""
    seconds = int(seconds)  # 소수점 버리기
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"

# --- 새롭게 정의된 UI 드로잉 함수 ---
# --- 기존의 draw_ui 함수를 이 코드로 완전히 교체하세요 ---

# --- 기존의 draw_ui 함수를 이 코드로 완전히 교체하세요 ---

def draw_ui(frame, fps):
    global posture_score, neck_tilt, calibrated, calibration_feedback_text, calibration_feedback_time
    
    ui_panel = np.zeros((frame.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)

    # 버튼 그리기 (패널 내부 좌표 사용)
    start_button_color = (0, 150, 0) if not timer_running else (50, 50, 50)
    cv2.rectangle(ui_panel, (BUTTON_X_OFFSET, BUTTON_Y_START), (BUTTON_X_OFFSET + BUTTON_WIDTH, BUTTON_Y_START + BUTTON_HEIGHT), start_button_color, -1)
    cv2.putText(ui_panel, 'Start Timer', (BUTTON_X_OFFSET + 45, BUTTON_Y_START + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    stop_button_color = (0, 0, 150) if timer_running else (50, 50, 50)
    cv2.rectangle(ui_panel, (BUTTON_X_OFFSET, BUTTON_Y_START + BUTTON_SPACING), (BUTTON_X_OFFSET + BUTTON_WIDTH, BUTTON_Y_START + BUTTON_SPACING + BUTTON_HEIGHT), stop_button_color, -1)
    cv2.putText(ui_panel, 'Stop Timer', (BUTTON_X_OFFSET + 55, BUTTON_Y_START + BUTTON_SPACING + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 텍스트 y 좌표 시작 위치 조정 (버튼 아래로 충분한 간격 확보)
    y_pos = BUTTON_Y_START + BUTTON_HEIGHT + BUTTON_SPACING + 40

    # 실시간 분석 정보
    cv2.putText(ui_panel, "Real-time Analysis", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    y_pos += 30
    cv2.putText(ui_panel, f"Posture Score: {posture_score:.2f}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    y_pos += 25
    cv2.putText(ui_panel, "(maintain over 0.9 score)", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    y_pos += 35
    cv2.putText(ui_panel, f"Neck Tilt: {neck_tilt:.2f}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 타이머 결과
    y_pos += 45
    cv2.putText(ui_panel, "Analysis Results", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    y_pos += 30
    current_total_time = (time.time() - posture_timer.start_time) if timer_running else posture_timer.total_time
    
    if timer_running or show_results:
        # format_time 함수를 사용하여 시간 표시 형식을 변경
        forward_neck_str = format_time(posture_timer.forward_neck_time)
        total_time_str = format_time(current_total_time)
        
        cv2.putText(ui_panel, f"Forward Neck: {forward_neck_str}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
        y_pos += 25
        cv2.putText(ui_panel, f"Total Time:   {total_time_str}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    # 캘리브레이션 정보
    y_pos = frame.shape[0] - 120
    cv2.putText(ui_panel, "--- Calibration ---", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    y_pos += 30
    if not calibrated:
        cv2.putText(ui_panel, "Press 'c' to calibrate", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        cv2.putText(ui_panel, "Calibrated! (Press 'c' again)", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if time.time() - calibration_feedback_time < 2.5:
        y_pos += 25
        cv2.putText(ui_panel, calibration_feedback_text, (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
    # 성능 정보
    y_pos = frame.shape[0] - 40
    cv2.putText(ui_panel, f"FPS: {fps:.1f}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return cv2.hconcat([frame, ui_panel])

# ===== (추가) Streamlit용 코어 클래스: 프레임 단위 처리 =====
class FrontCore:
    """
    - Streamlit WebRTC에서 매 프레임 호출되는 process_frame()을 제공
    - 기존 final_front.py의 계산식을 그대로 사용하되, OpenCV 윈도우/마우스 콜백/VideoPlayer는 제거
    - 외부에서 start/stop/reset/calibrate 제어 가능
    """
    def __init__(self):
        # MediaPipe 초기화
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # 타이머/상태
        self.timer_active = False
        self.start_time = None
        self.forward_neck_time = 0.0
        self.total_time = 0.0
        self.last_time = time.time()
        # 캘리브레이션 상태
        self.calibrated = False
        self.calibrated_chin_shoulder_dist = None
        self.calibrated_shoulder_width_px = None
        self._want_calibrate = False
        # 디스플레이 값
        self.posture_score = 0.0
        self.neck_tilt = 0.0

        # === (추가) 알람 상태 ===
        alarm_path = _find_alarm_path()
        self._alarm = AlarmPlayer(alarm_path)
        self._alarm_on = False

        # === (추가) posture_score 기준 알람 토글

    def _update_alarm(self, forward_head: bool):
        """
        forward_head == True 이고 타이머가 켜져 있을 때만 알람 ON
        (요구사항: posture_score < 0.9로 내려가 'forward neck' 시간이 오를 때)
        """
        if not self._alarm:
            return
        want_on = bool(self.timer_active and forward_head)
        if want_on and not self._alarm_on:
            self._alarm.start()
            self._alarm_on = True
        elif (not want_on) and self._alarm_on:
            self._alarm.stop()
            self._alarm_on = False


    # ---- 외부 제어용 ----
    def start_timer(self):
        self.timer_active = True
        self.start_time = time.time()
        self.forward_neck_time = 0.0
        self.total_time = 0.0
        self.last_time = time.time()

    def reset_all(self):
        self.timer_active = False
        self.start_time = None
        self.forward_neck_time = 0.0
        self.total_time = 0.0
        self.last_time = time.time()
        self.calibrated = False
        self.calibrated_chin_shoulder_dist = None
        self.calibrated_shoulder_width_px = None
        self.posture_score = 0.0
        self.neck_tilt = 0.0
        # 알람 즉시 OFF
        if self._alarm_on and self._alarm:
            self._alarm.stop()
            self._alarm_on = False

    def request_calibration(self):
        self._want_calibrate = True

    def get_stats(self):
        return {
            "total": self.total_time,
            "forward_neck": self.forward_neck_time,
            "posture_score": self.posture_score,
            "neck_tilt": self.neck_tilt,
            "calibrated": self.calibrated
        }

    def stop_timer(self):
        self.timer_active = False
        # 알람 즉시 OFF
        if self._alarm_on and self._alarm:
            self._alarm.stop()
            self._alarm_on = False

    # ---- 내부 유틸 ----
    @staticmethod
    def _to_pixel(x, y, shape):
        h, w = shape[:2]
        return int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))

    # ---- 프레임 처리 메인 ----
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        입력: BGR ndarray (WebRTC 프레임)
        출력: 주석/라벨이 그려진 BGR ndarray
        """
        img = frame_bgr.copy()
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # 시간 갱신
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if self.timer_active:
            self.total_time += dt

        # Mediapipe 추론
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            # 주요 포인트
            LS = mp_pose.PoseLandmark.LEFT_SHOULDER
            RS = mp_pose.PoseLandmark.RIGHT_SHOULDER
            ML = mp_pose.PoseLandmark.MOUTH_LEFT
            MR = mp_pose.PoseLandmark.MOUTH_RIGHT
            NO = mp_pose.PoseLandmark.NOSE
            LH = mp_pose.PoseLandmark.LEFT_HIP
            RH = mp_pose.PoseLandmark.RIGHT_HIP
            LE = mp_pose.PoseLandmark.LEFT_ELBOW
            RE = mp_pose.PoseLandmark.RIGHT_ELBOW

            ls, rs = lms[LS.value], lms[RS.value]
            ml, mr = lms[ML.value], lms[MR.value]
            nose   = lms[NO.value]
            lh, rh = lms[LH.value], lms[RH.value]
            le, re = lms[LE.value], lms[RE.value]

            # 픽셀 좌표
            pls = self._to_pixel(ls.x, ls.y, img.shape)
            prs = self._to_pixel(rs.x, rs.y, img.shape)
            pml = self._to_pixel(ml.x, ml.y, img.shape)
            pmr = self._to_pixel(mr.x, mr.y, img.shape)

            shoulder_mid = ((pls[0] + prs[0]) // 2, (pls[1] + prs[1]) // 2)
            jaw_mid = ((pml[0] + pmr[0]) // 2, (pml[1] + pmr[1]) // 2)

            shoulder_w_px = max(1.0, float(np.hypot(prs[0] - pls[0], prs[1] - pls[1])))
            d_norm = float(np.hypot(jaw_mid[0] - shoulder_mid[0], jaw_mid[1] - shoulder_mid[1])) / shoulder_w_px

            # (final_front.py에 있던) 캘리브레이션 적용
            min_dist, max_dist = 0.08, 0.15
            if self.calibrated and self.calibrated_shoulder_width_px and self.calibrated_shoulder_width_px > 0:
                scale_factor = shoulder_w_px / float(self.calibrated_shoulder_width_px)
                adjusted_ideal = self.calibrated_chin_shoulder_dist * scale_factor
                max_dist = adjusted_ideal
                min_dist = adjusted_ideal * 0.7

            # posture_score: 1=좋음, 0=나쁨 (final_front.py 로직 그대로)
            self.posture_score = max(0, min((d_norm - min_dist) / (max_dist - min_dist + 1e-6), 1))

            # 목 기울임
            shoulder_mid_x = (ls.x + rs.x) / 2.0
            self.neck_tilt = nose.x - shoulder_mid_x
            max_tilt_threshold = 0.08
            tilt_bad = abs(self.neck_tilt) > (max_tilt_threshold / 2.0)

            # 색상 및 보조 라벨
            red = int(255 * (1 - self.posture_score)); green = int(255 * self.posture_score)
            shoulder_color = (0, green, red)

            # 시각화
            cv2.line(img, pls, prs, shoulder_color, 3)
            cv2.circle(img, shoulder_mid, 5, (255, 255, 0), -1)
            cv2.circle(img, jaw_mid, 5, (0, 165, 255), -1)
            cv2.line(img, shoulder_mid, self._to_pixel(nose.x, nose.y, img.shape), (50, 200, 255), 2)

            forward_head = (self.posture_score < 0.9)
            self._update_alarm(forward_head)
            # === (추가) 알람 상태 갱신
            self._update_alarm(forward_head)
            label = []
            if forward_head: label.append("Forward Head")
            if tilt_bad:     label.append("Neck Tilt")
            if not label:    label.append("Good (Front)")
            cv2.putText(img, " | ".join(label), (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if (forward_head or tilt_bad) else (0, 180, 0), 2, cv2.LINE_AA)

            # 캘리브레이션 요청 처리
            if self._want_calibrate:
                if all(pt.visibility > 0.7 for pt in [ls, rs, ml, mr]):
                    self.calibrated_chin_shoulder_dist = abs(((ls.y + rs.y) / 2.0) - ((ml.y + mr.y) / 2.0))
                    self.calibrated_shoulder_width_px = shoulder_w_px
                    self.calibrated = True
                    self._want_calibrate = False

            # 타이머 누적
            bad_posture = forward_head or tilt_bad
            if self.timer_active and bad_posture:
                self.forward_neck_time += dt

            # HUD
            cv2.putText(img,
                f"Bad: {self.forward_neck_time:5.1f}s | Total: {self.total_time:5.1f}s | Calib: {'Y' if self.calibrated else 'N'}",
                (20, img.shape[0]-22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if bad_posture else (120, 120, 120), 2, cv2.LINE_AA)

        else:
            # 미검출: 알람 강제 OFF
            self._update_alarm(False)
            # 미검출 HUD
            cv2.putText(img, "No person detected", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        return img

def main():
    global posture_score, neck_tilt
    global calibrated, calibrated_chin_shoulder_dist, calibrated_shoulder_width_px, calibrated_neck_tilt_offset, calibration_feedback_text, calibration_feedback_time

    VIDEO_SOURCE = 0
    player = None

    try:
        # === (추가) 로컬 실행 알람 준비
        _alarm_local = AlarmPlayer(_find_alarm_path())
        _alarm_local_on = False

        player = VideoPlayer(source=VIDEO_SOURCE, flip=True, fps=30)
        player.start()
        
        title = "AI Front Posture Analysis"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, on_mouse_click)

        last_time = time.time()

        while True:
            frame = player.next()
            if frame is None:
                break
            
            # 영상 리사이즈 (가로 1280에 맞게)
            h, w, _ = frame.shape
            scale = 960 / w
            frame = cv2.resize(frame, (960, int(h * scale)), interpolation=cv2.INTER_AREA)
            
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            fps = 1 / delta_time if delta_time > 0 else 0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
                mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
                nose = landmarks[mp_pose.PoseLandmark.NOSE]

                if all(lm.visibility > 0.7 for lm in [left_shoulder, right_shoulder, mouth_left, mouth_right, nose]):
                    frame_height, frame_width, _ = frame.shape
                    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                    chin_y = (mouth_left.y + mouth_right.y) / 2
                    distance = abs(shoulder_y - chin_y)

                    min_dist, max_dist = 0.08, 0.15
                    if calibrated:
                        current_shoulder_width_px = abs((left_shoulder.x - right_shoulder.x) * frame_width)
                        if calibrated_shoulder_width_px > 0:
                            scale_factor = current_shoulder_width_px / calibrated_shoulder_width_px
                            adjusted_ideal_dist = calibrated_chin_shoulder_dist * scale_factor
                            max_dist = adjusted_ideal_dist
                            min_dist = adjusted_ideal_dist * 0.7
                    
                    posture_score = max(0, min((distance - min_dist) / (max_dist - min_dist), 1))
                    red_color, green_color = 255 * (1 - posture_score), 255 * posture_score
                    shoulder_color = (0, green_color, red_color)
                    
                    left_shoulder_px = (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height))
                    right_shoulder_px = (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height))
                    cv2.line(frame, left_shoulder_px, right_shoulder_px, shoulder_color, 3)

                    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
                    neck_tilt = (nose.x - shoulder_mid_x) - calibrated_neck_tilt_offset
                    
                    max_tilt_threshold = 0.08
                    tilt_ratio = max(0, 1 - (abs(neck_tilt) / max_tilt_threshold))
                    tilt_color = (0, 255 * tilt_ratio, 255 * (1 - tilt_ratio))
                    
                    nose_px = (int(nose.x * frame_width), int(nose.y * frame_height))
                    shoulder_mid_px = (int(shoulder_mid_x * frame_width), int(shoulder_y * frame_height))
                    cv2.line(frame, nose_px, shoulder_mid_px, tilt_color, 2)

                    if posture_score < 0.9 or abs(neck_tilt) > max_tilt_threshold / 2:
                        posture_timer.add_forward_neck_time(delta_time)

                    # === (추가) 로컬 알람: posture_score 기준만 체크
                    if timer_running and posture_score < 0.9:
                        if not _alarm_local_on and _alarm_local:
                            _alarm_local.start()
                            _alarm_local_on = True
                    else:
                        if _alarm_local_on and _alarm_local:
                            _alarm_local.stop()
                            _alarm_local_on = False

            display_frame = draw_ui(frame, fps)
            cv2.imshow(title, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            
            if key == ord('c'):
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    if all(landmarks[i].visibility > 0.7 for i in [11, 12, 9, 10]):
                        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
                        mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
                        
                        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                        chin_y = (mouth_left.y + mouth_right.y) / 2
                        calibrated_chin_shoulder_dist = abs(shoulder_y - chin_y)
                        
                        h, w, _ = frame.shape
                        l_s_px = left_shoulder.x * w
                        r_s_px = right_shoulder.x * w
                        calibrated_shoulder_width_px = abs(l_s_px - r_s_px)

                        nose = landmarks[mp_pose.PoseLandmark.NOSE]
                        calibrated_neck_tilt_offset = nose.x - (left_shoulder.x + right_shoulder.x) / 2
                        calibrated = True
                        calibration_feedback_text = "Calibrated!"
                    else:
                        calibration_feedback_text = "Cannot see face/shoulders clearly."
                else:
                    calibration_feedback_text = "No person detected."
                calibration_feedback_time = time.time()

    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            player.stop()
        pose.close()
        cv2.destroyAllWindows()

        # === (추가) 알람 정리
        try:
            if _alarm_local_on and _alarm_local:
                _alarm_local.stop()
        except Exception:
            pass

if __name__ == '__main__':
    main()
