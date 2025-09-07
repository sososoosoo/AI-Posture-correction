import cv2
import mediapipe as mp
import time
import collections
import threading
import numpy as np
import tkinter as tk
import os
import pygame
from datetime import datetime, timedelta

# MediaPipe Pose 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils # MediaPipe 드로잉 유틸리티 추가
mp_drawing_styles = mp.solutions.drawing_styles # MediaPipe 드로잉 스타일 추가

# --- 캘리브레이션 관련 전역 변수 ---
calibrated = False
calibrated_chin_shoulder_dist = None
calibrated_shoulder_width_px = None
calibrated_neck_tilt_offset = 0.0
calibration_feedback_text = ""
calibration_feedback_time = 0

# ===== 알람 사운드 유틸 (pygame 사용) =====
_ALARM_CANDIDATES = [
    "alarm.wav", "alarm.mp3",
    os.path.join("assets", "alarm.wav"),
    os.path.join("assets", "alarm.mp3"),
]
def _find_alarm_path():
    for p in _ALARM_CANDIDATES:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

class AlarmPlayer:
    def __init__(self, sound_path: str | None):
        self.sound_path = sound_path
        self._is_initialized = False
        if not sound_path:
            print("알람 파일 경로가 없습니다. (alarm.wav 또는 alarm.mp3)")
            return
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(sound_path)
            self._is_initialized = True
            print(f"알람 파일 '{sound_path}' 로드 성공.")
        except Exception as e:
            print(f"pygame mixer 초기화 또는 사운드 파일 로드 실패: {e}")
            self.sound_path = None

    def start(self):
        if not self._is_initialized or pygame.mixer.music.get_busy():
            return
        try:
            pygame.mixer.music.play(loops=-1)
        except Exception as e:
            print(f"알람 재생 실패: {e}")

    def stop(self):
        if not self._is_initialized:
            return
        try:
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"알람 중지 실패: {e}")

    def quit(self):
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


# ===== (수정) PostureTimer 클래스: 로깅 기능 추가 =====
class PostureTimer:
    def __init__(self, duration):
        self.start_time = None
        self.duration = duration
        self.timer_active = False
        self.forward_neck_time = 0.0
        self.total_time = 0.0
        
        # 로깅 관련 변수
        self.bad_posture_log = []  # (시작시간, 종료시간) 튜플 저장
        self._bad_posture_start_time = None # 현재 안 좋은 자세 시작 시간

    def start_timer(self):
        now = time.time()
        self.start_time = now
        self.timer_active = True
        self.forward_neck_time = 0.0
        self.total_time = 0.0
        # 로깅 초기화
        self.bad_posture_log = []
        self._bad_posture_start_time = None

    def stop_timer(self):
        self.timer_active = False
        now = time.time()
        self.total_time = now - self.start_time if self.start_time else 0
        
        # 타이머 중지 시, 진행중이던 안 좋은 자세가 있었다면 기록
        if self._bad_posture_start_time is not None:
            self.bad_posture_log.append((self._bad_posture_start_time, now))
            self._bad_posture_start_time = None
        
        return {
            "forward_neck_time": self.forward_neck_time,
            "total_time": self.total_time,
            "bad_posture_log": self.bad_posture_log
        }

    def add_forward_neck_time(self, duration):
        if self.timer_active:
            self.forward_neck_time += duration

    def update_posture_status(self, is_bad: bool, current_time: float):
        """현재 자세 상태를 기반으로 로그를 기록합니다."""
        if not self.timer_active:
            return

        # 1. 좋은 자세 -> 안 좋은 자세로 변경된 순간
        if is_bad and self._bad_posture_start_time is None:
            self._bad_posture_start_time = current_time
        
        # 2. 안 좋은 자세 -> 좋은 자세로 변경된 순간
        elif not is_bad and self._bad_posture_start_time is not None:
            self.bad_posture_log.append((self._bad_posture_start_time, current_time))
            self._bad_posture_start_time = None


# --- UI 및 타이머 상태 변수 ---
timer_running = False
show_results = False
alarm_enabled = True  # 알람 ON/OFF 상태
posture_timer = PostureTimer(duration=1800)
posture_score = 0.0
neck_tilt = 0.0

# --- UI 상수 정의 ---
FRAME_WIDTH = 960
PANEL_WIDTH = 350
BUTTON_X_OFFSET = 20
BUTTON_Y_START = 50
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 60
BUTTON_SPACING = 80
CLICK_BUTTON_X_START = FRAME_WIDTH + BUTTON_X_OFFSET

# ===== (수정) 마우스 클릭 이벤트 핸들러: 알람 버튼 추가 =====
def on_mouse_click(event, x, y, flags, param):
    global timer_running, show_results, alarm_enabled
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start 버튼
        if not timer_running and (CLICK_BUTTON_X_START <= x <= CLICK_BUTTON_X_START + BUTTON_WIDTH and BUTTON_Y_START <= y <= BUTTON_Y_START + BUTTON_HEIGHT):
            posture_timer.start_timer()
            timer_running = True
            show_results = False
            print("타이머 시작")
        # Stop 버튼
        elif timer_running and (CLICK_BUTTON_X_START <= x <= CLICK_BUTTON_X_START + BUTTON_WIDTH and (BUTTON_Y_START + BUTTON_SPACING) <= y <= (BUTTON_Y_START + BUTTON_SPACING + BUTTON_HEIGHT)):
            results = posture_timer.stop_timer()
            timer_running = False
            show_results = True
            print("타이머 종료. 분석 결과:")
            print(f"  - 총 측정 시간: {results['total_time']:.2f}초")
            print(f"  - 안좋은 자세 누적 시간: {results['forward_neck_time']:.2f}초")
            print("  - 안좋은 자세 구간 로그:")
            if not results['bad_posture_log']:
                print("    (기록 없음)")
            else:
                for start, end in results['bad_posture_log']:
                    start_str = datetime.fromtimestamp(start).strftime('%H:%M:%S')
                    end_str = datetime.fromtimestamp(end).strftime('%H:%M:%S')
                    duration = end - start
                    print(f"    - 시작: {start_str}, 종료: {end_str} ({duration:.2f}초)")

        # Alarm 버튼
        alarm_button_y_start = BUTTON_Y_START + 2 * BUTTON_SPACING
        if (CLICK_BUTTON_X_START <= x <= CLICK_BUTTON_X_START + BUTTON_WIDTH and alarm_button_y_start <= y <= alarm_button_y_start + BUTTON_HEIGHT):
            alarm_enabled = not alarm_enabled
            print(f"알람이 {'활성화' if alarm_enabled else '비활성화'}되었습니다.")

def format_time(seconds):
    """초를 '00h 00m 00s' 형식의 문자열로 변환합니다."""
    return str(timedelta(seconds=int(seconds)))

# ===== (수정) UI 드로잉 함수: 알람 버튼 및 결과 로그 표시 =====
def draw_ui(frame, fps):
    global posture_score, neck_tilt, calibrated, calibration_feedback_text, calibration_feedback_time, alarm_enabled, show_results
    
    ui_panel = np.zeros((frame.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)

    # --- 버튼 그리기 ---
    # Start 버튼
    start_button_color = (0, 150, 0) if not timer_running else (50, 50, 50)
    cv2.rectangle(ui_panel, (BUTTON_X_OFFSET, BUTTON_Y_START), (BUTTON_X_OFFSET + BUTTON_WIDTH, BUTTON_Y_START + BUTTON_HEIGHT), start_button_color, -1)
    cv2.putText(ui_panel, 'Start Timer', (BUTTON_X_OFFSET + 70, BUTTON_Y_START + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Stop 버튼
    stop_button_y = BUTTON_Y_START + BUTTON_SPACING
    stop_button_color = (0, 0, 150) if timer_running else (50, 50, 50)
    cv2.rectangle(ui_panel, (BUTTON_X_OFFSET, stop_button_y), (BUTTON_X_OFFSET + BUTTON_WIDTH, stop_button_y + BUTTON_HEIGHT), stop_button_color, -1)
    cv2.putText(ui_panel, 'Stop Timer', (BUTTON_X_OFFSET + 75, stop_button_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Alarm 버튼
    alarm_button_y = BUTTON_Y_START + 2 * BUTTON_SPACING
    alarm_text = f"Alarm: {'ON' if alarm_enabled else 'OFF'}"
    alarm_color = (0, 180, 0) if alarm_enabled else (50, 50, 180)
    cv2.rectangle(ui_panel, (BUTTON_X_OFFSET, alarm_button_y), (BUTTON_X_OFFSET + BUTTON_WIDTH, alarm_button_y + BUTTON_HEIGHT), alarm_color, -1)
    cv2.putText(ui_panel, alarm_text, (BUTTON_X_OFFSET + 75, alarm_button_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_pos = alarm_button_y + BUTTON_HEIGHT + 40

    # --- 실시간 분석 정보 ---
    cv2.putText(ui_panel, "Real-time Analysis", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    y_pos += 25
    cv2.putText(ui_panel, f"Posture Score: {posture_score:.2f}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += 20
    cv2.putText(ui_panel, "(maintain over 0.9)", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # === Neck Tilt 점수 표시 (복원) ===
    y_pos += 25 
    cv2.putText(ui_panel, f"Neck Tilt: {neck_tilt:.2f}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- 타이머 결과 ---
    y_pos += 40
    cv2.putText(ui_panel, "Analysis Results", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    y_pos += 25
    
    current_total_time = (time.time() - posture_timer.start_time) if timer_running else posture_timer.total_time
    
    if timer_running or show_results:
        forward_neck_str = format_time(posture_timer.forward_neck_time)
        total_time_str = format_time(current_total_time)
        
        cv2.putText(ui_panel, f"Bad Posture: {forward_neck_str}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        y_pos += 20
        cv2.putText(ui_panel, f"Total Time:   {total_time_str}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if show_results:
        y_pos += 30
        cv2.putText(ui_panel, "Bad Posture Log (Last 5)", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += 20
        
        # 마지막 5개 로그만 표시
        for start, end in posture_timer.bad_posture_log[-5:]:
            if y_pos > frame.shape[0] - 80: break # 패널 넘어가면 중단
            duration = end - start
            start_str = datetime.fromtimestamp(start).strftime('%H:%M:%S')
            log_text = f"{start_str} ({duration:.1f}s)"
            cv2.putText(ui_panel, log_text, (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_pos += 18

    # --- 캘리브레이션 및 성능 정보 ---
    y_pos = frame.shape[0] - 80
    cv2.putText(ui_panel, "--- Calibration ---", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    y_pos += 25
    if not calibrated:
        cv2.putText(ui_panel, "Press 'c' to calibrate", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(ui_panel, "Calibrated! (Press 'c' again)", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y_pos += 25
    cv2.putText(ui_panel, f"FPS: {fps:.1f}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return cv2.hconcat([frame, ui_panel])


def main():
    global posture_score, neck_tilt
    global calibrated, calibrated_chin_shoulder_dist, calibrated_shoulder_width_px, calibrated_neck_tilt_offset, calibration_feedback_text, calibration_feedback_time
    global timer_running, alarm_enabled

    VIDEO_SOURCE = 0
    player = None
    _alarm_local = None

    try:
        _alarm_local = AlarmPlayer(_find_alarm_path())
        _alarm_local_on = False

        player = VideoPlayer(source=VIDEO_SOURCE, flip=True, fps=30, width=960, height=540)
        player.start()
        
        title = "AI Front Posture Analysis"
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, on_mouse_click)

        last_time = time.time()

        while True:
            frame = player.next()
            if frame is None: break
            
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            fps = 1 / delta_time if delta_time > 0 else 0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            is_bad_posture = False

            if results.pose_landmarks:
                # MediaPipe 랜드마크 그리기 (복원)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                landmarks = results.pose_landmarks.landmark
                
                # 필요한 랜드마크들
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
                mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                
                # 가시성 검사
                required_landmarks = [left_shoulder, right_shoulder, mouth_left, mouth_right, nose]
                if all(lm.visibility > 0.7 for lm in required_landmarks):
                    frame_height, frame_width, _ = frame.shape
                    
                    # 어깨 중앙점 계산
                    shoulder_mid_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame_width)
                    shoulder_mid_y = int((left_shoulder.y + right_shoulder.y) / 2 * frame_height)
                    
                    # 코 끝점 계산
                    nose_x = int(nose.x * frame_width)
                    nose_y = int(nose.y * frame_height)

                    # 목 기울기 계산
                    shoulder_mid_x_norm = (left_shoulder.x + right_shoulder.x) / 2
                    neck_tilt = (nose.x - shoulder_mid_x_norm) - calibrated_neck_tilt_offset
                    
                    # === Neck Tilt 시각화 (복원 및 수정) ===
                    max_tilt_threshold = 0.08
                    tilt_ratio = max(0, 1 - (abs(neck_tilt) / max_tilt_threshold))
                    tilt_color = (0, 255 * tilt_ratio, 255 * (1 - tilt_ratio)) # 초록색(양호) -> 빨간색(나쁨)

                    # 코와 어깨 중앙을 잇는 선 그리기 (색상 적용)
                    cv2.line(frame, (nose_x, nose_y), (shoulder_mid_x, shoulder_mid_y), tilt_color, 2)
                    
                    # 거북목 스코어 계산 (기존 로직)
                    shoulder_y_norm = (left_shoulder.y + right_shoulder.y) / 2
                    chin_y_norm = (mouth_left.y + mouth_right.y) / 2
                    distance_norm = abs(shoulder_y_norm - chin_y_norm) # 정규화된 거리
                    
                    min_dist, max_dist = 0.08, 0.15
                    if calibrated:
                        current_shoulder_width_px = abs((left_shoulder.x - right_shoulder.x) * frame_width)
                        if calibrated_shoulder_width_px > 0:
                            scale_factor = current_shoulder_width_px / calibrated_shoulder_width_px
                            adjusted_ideal_dist = calibrated_chin_shoulder_dist * scale_factor
                            max_dist = adjusted_ideal_dist
                            min_dist = adjusted_ideal_dist * 0.7
                    
                    posture_score = max(0, min((distance_norm - min_dist) / (max_dist - min_dist + 1e-6), 1))
                    
                    # 어깨선 그리기
                    left_shoulder_px = (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height))
                    right_shoulder_px = (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height))
                    red_color, green_color = 255 * (1 - posture_score), 255 * posture_score
                    cv2.line(frame, left_shoulder_px, right_shoulder_px, (0, green_color, red_color), 3)

                    if posture_score < 0.9 or abs(neck_tilt) > 0.04: # 거북목 또는 목 기울어짐
                        is_bad_posture = True
                        if timer_running:
                            posture_timer.add_forward_neck_time(delta_time)

            if timer_running:
                posture_timer.update_posture_status(is_bad_posture, current_time)

            # --- 통합 알람 로직 ---
            should_alarm = timer_running and alarm_enabled and is_bad_posture
            if should_alarm:
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
            if key == ord('q') or key == 27: break
            
            if key == ord('c'):
                # 캘리브레이션 로직 (기존과 동일)
                if results.pose_landmarks and all(landmarks[i].visibility > 0.7 for i in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT, mp_pose.PoseLandmark.NOSE]):
                    calibrated_chin_shoulder_dist = abs(((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2) - ((landmarks[mp_pose.PoseLandmark.MOUTH_LEFT].y + landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT].y) / 2))
                    calibrated_shoulder_width_px = abs((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) * frame.shape[1])
                    calibrated_neck_tilt_offset = landmarks[mp_pose.PoseLandmark.NOSE].x - (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
                    calibrated = True
                    calibration_feedback_text = "Calibrated!"
                else:
                    calibration_feedback_text = "Cannot see landmarks for calibration. Please adjust your posture."
                calibration_feedback_time = time.time()

    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"프로그램 종료: {e}")
    finally:
        if player: player.stop()
        if _alarm_local: _alarm_local.quit()
        pose.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()