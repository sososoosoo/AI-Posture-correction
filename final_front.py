import cv2
import mediapipe as mp
import time
import collections
import threading
import numpy as np
import os
import pygame
from datetime import datetime, timedelta
import openpyxl
from win10toast_persist import ToastNotifier

# MediaPipe Pose 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 전역 변수 ---
calibrated = False
calibrated_chin_shoulder_dist = None
calibrated_shoulder_width_px = None
calibrated_neck_tilt_offset = 0.0
save_feedback_text = ""
save_feedback_time = 0

# --- UI 및 타이머 상태 변수 ---
timer_running = False
show_results = False
alarm_enabled = True
toast_alarm_enabled = True

# ===== (이하 생략된 클래스 정의들은 이전 답변의 전체 코드를 참고해주세요) =====
# AlarmPlayer, VideoPlayer, PostureTimer 클래스 정의...
_ALARM_CANDIDATES = ["alarm.wav", "alarm.mp3", os.path.join("assets", "alarm.wav"), os.path.join("assets", "alarm.mp3")]


def _find_alarm_path():
    for p in _ALARM_CANDIDATES:
        if os.path.exists(p): return os.path.abspath(p)
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
            print(f"pygame mixer 초기화 실패: {e}")
            self.sound_path = None

    def start(self):
        if not self._is_initialized or pygame.mixer.music.get_busy(): return
        try:
            pygame.mixer.music.play(loops=-1)
        except Exception as e:
            print(f"알람 재생 실패: {e}")

    def stop(self):
        if not self._is_initialized: return
        try:
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"알람 중지 실패: {e}")

    def quit(self):
        if self._is_initialized: pygame.mixer.quit()


class VideoPlayer:
    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0, width=960, height=540):
        self.cv2 = cv2
        self.__cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.__cap.isOpened(): raise RuntimeError(f"Cannot open camera {source}")
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
        self.__input_fps = self.__cap.get(cv2.CAP_PROP_FPS) or 60
        self.__output_fps = fps or self.__input_fps
        self.__flip = flip
        self.__size = size
        if size is not None: self.__interpolation = cv2.INTER_AREA if size[0] < self.__cap.get(
            cv2.CAP_PROP_FRAME_WIDTH) else cv2.INTER_LINEAR
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
        if self.__thread is not None: self.__thread.join()
        self.__cap.release()

    def __run(self):
        prev_time = 0
        while not self.__stop:
            t1 = time.time()
            ret, frame = self.__cap.read()
            if not ret: break
            if 1 / self.__output_fps < time.time() - prev_time:
                prev_time = time.time()
                with self.__lock: self.__frame = frame
            t2 = time.time()
            wait_time = 1 / self.__input_fps - (t2 - t1)
            time.sleep(max(0, wait_time))
        self.__frame = None

    def next(self):
        with self.__lock:
            if self.__frame is None: return None
            frame = self.__frame.copy()
        if self.__size is not None: frame = self.cv2.resize(frame, self.__size, interpolation=self.__interpolation)
        if self.__flip: frame = self.cv2.flip(frame, 1)
        return frame


class PostureTimer:
    def __init__(self, duration):
        self.start_time = None;
        self.duration = duration;
        self.timer_active = False
        self.forward_neck_time = 0.0;
        self.total_time = 0.0
        self.bad_posture_log = [];
        self._bad_posture_start_time = None

    def start_timer(self):
        self.start_time = time.time();
        self.timer_active = True
        self.forward_neck_time = 0.0;
        self.total_time = 0.0
        self.bad_posture_log = [];
        self._bad_posture_start_time = None

    def stop_timer(self):
        self.timer_active = False
        now = time.time()
        self.total_time = now - self.start_time if self.start_time else 0
        if self._bad_posture_start_time is not None:
            self.bad_posture_log.append((self._bad_posture_start_time, now))
            self._bad_posture_start_time = None
        return {"forward_neck_time": self.forward_neck_time, "total_time": self.total_time,
                "bad_posture_log": self.bad_posture_log}

    def add_forward_neck_time(self, duration):
        if self.timer_active: self.forward_neck_time += duration

    def update_posture_status(self, is_bad: bool, current_time: float):
        if not self.timer_active: return
        if is_bad and self._bad_posture_start_time is None:
            self._bad_posture_start_time = current_time
        elif not is_bad and self._bad_posture_start_time is not None:
            self.bad_posture_log.append((self._bad_posture_start_time, current_time))
            self._bad_posture_start_time = None


posture_timer = PostureTimer(duration=1800)
posture_score = 0.0
neck_tilt = 0.0

# --- ★★★ UI 상수 정의 (수정된 값) ★★★ ---
FRAME_WIDTH = 960;
PANEL_WIDTH = 350;
BUTTON_X_OFFSET = 20
BUTTON_Y_START = 30  # 40 -> 30
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 40  # 50 -> 40
BUTTON_SPACING = 55  # 65 -> 55
CLICK_BUTTON_X_START = FRAME_WIDTH + BUTTON_X_OFFSET


def save_results_to_excel(timer_data):
    try:
        workbook = openpyxl.Workbook()
        summary_sheet = workbook.active;
        summary_sheet.title = "요약"
        summary_sheet.append(["항목", "결과"])
        summary_sheet.append(["총 측정 시간 (초)", f"{timer_data.total_time:.2f}"])
        summary_sheet.append(["안좋은 자세 누적 시간 (초)", f"{timer_data.forward_neck_time:.2f}"])
        log_sheet = workbook.create_sheet(title="상세 로그")
        log_sheet.append(["시작 시간", "종료 시간", "지속 시간 (초)"])
        for start, end in timer_data.bad_posture_log:
            log_sheet.append([datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S'),
                              datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S'), f"{(end - start):.2f}"])
        filename = f"posture_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
        workbook.save(filename)
        return f"Saved: {filename.split('_')[-1]}"
    except Exception as e:
        return f"Error: {e}"


def on_mouse_click(event, x, y, flags, param):
    global timer_running, show_results, alarm_enabled, toast_alarm_enabled, save_feedback_text, save_feedback_time
    if event == cv2.EVENT_LBUTTONDOWN:
        if not timer_running and (
                CLICK_BUTTON_X_START <= x <= CLICK_BUTTON_X_START + BUTTON_WIDTH and BUTTON_Y_START <= y <= BUTTON_Y_START + BUTTON_HEIGHT):
            posture_timer.start_timer();
            timer_running = True;
            show_results = False;
            print("타이머 시작")
        elif timer_running and (CLICK_BUTTON_X_START <= x <= CLICK_BUTTON_X_START + BUTTON_WIDTH and (
                BUTTON_Y_START + BUTTON_SPACING) <= y <= (BUTTON_Y_START + BUTTON_SPACING + BUTTON_HEIGHT)):
            posture_timer.stop_timer();
            timer_running = False;
            show_results = True;
            print("타이머 종료. 결과:", posture_timer.stop_timer())

        alarm_button_y = BUTTON_Y_START + 2 * BUTTON_SPACING
        if (
                CLICK_BUTTON_X_START <= x <= CLICK_BUTTON_X_START + BUTTON_WIDTH and alarm_button_y <= y <= alarm_button_y + BUTTON_HEIGHT):
            alarm_enabled = not alarm_enabled;
            print(f"소리 알람 {'활성화' if alarm_enabled else '비활성화'}")

        toast_button_y = BUTTON_Y_START + 3 * BUTTON_SPACING
        if (
                CLICK_BUTTON_X_START <= x <= CLICK_BUTTON_X_START + BUTTON_WIDTH and toast_button_y <= y <= toast_button_y + BUTTON_HEIGHT):
            toast_alarm_enabled = not toast_alarm_enabled;
            print(f"윈도우 알람 {'활성화' if toast_alarm_enabled else '비활성화'}")

        save_button_y = BUTTON_Y_START + 4 * BUTTON_SPACING
        if show_results and (
                CLICK_BUTTON_X_START <= x <= CLICK_BUTTON_X_START + BUTTON_WIDTH and save_button_y <= y <= save_button_y + BUTTON_HEIGHT):
            feedback = save_results_to_excel(posture_timer)
            print(feedback)
            save_feedback_text = feedback;
            save_feedback_time = time.time()
            show_results = False


def format_time_str(seconds):
    return str(timedelta(seconds=int(seconds)))


def draw_ui(frame, fps):
    global posture_score, neck_tilt, calibrated, show_results, save_feedback_text, save_feedback_time, toast_alarm_enabled
    ui_panel = np.zeros((frame.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)

    # --- 버튼 그리기 (수정된 텍스트 위치) ---
    text_y_offset = 28  # 32 -> 28
    cv2.rectangle(ui_panel, (BUTTON_X_OFFSET, BUTTON_Y_START),
                  (BUTTON_X_OFFSET + BUTTON_WIDTH, BUTTON_Y_START + BUTTON_HEIGHT),
                  (0, 150, 0) if not timer_running else (50, 50, 50), -1)
    cv2.putText(ui_panel, 'Start Timer', (BUTTON_X_OFFSET + 70, BUTTON_Y_START + text_y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    stop_y = BUTTON_Y_START + BUTTON_SPACING
    cv2.rectangle(ui_panel, (BUTTON_X_OFFSET, stop_y), (BUTTON_X_OFFSET + BUTTON_WIDTH, stop_y + BUTTON_HEIGHT),
                  (0, 0, 150) if timer_running else (50, 50, 50), -1)
    cv2.putText(ui_panel, 'Stop Timer', (BUTTON_X_OFFSET + 75, stop_y + text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    alarm_y = BUTTON_Y_START + 2 * BUTTON_SPACING
    cv2.rectangle(ui_panel, (BUTTON_X_OFFSET, alarm_y), (BUTTON_X_OFFSET + BUTTON_WIDTH, alarm_y + BUTTON_HEIGHT),
                  (0, 180, 0) if alarm_enabled else (50, 50, 180), -1)
    cv2.putText(ui_panel, f"Alarm (Sound): {'ON' if alarm_enabled else 'OFF'}",
                (BUTTON_X_OFFSET + 30, alarm_y + text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    toast_y = BUTTON_Y_START + 3 * BUTTON_SPACING
    cv2.rectangle(ui_panel, (BUTTON_X_OFFSET, toast_y), (BUTTON_X_OFFSET + BUTTON_WIDTH, toast_y + BUTTON_HEIGHT),
                  (0, 180, 100) if toast_alarm_enabled else (80, 50, 50), -1)
    cv2.putText(ui_panel, f"Toast Alarm: {'ON' if toast_alarm_enabled else 'OFF'}",
                (BUTTON_X_OFFSET + 50, toast_y + text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    save_y = BUTTON_Y_START + 4 * BUTTON_SPACING
    save_color = (180, 100, 0) if show_results else (50, 50, 50)
    cv2.rectangle(ui_panel, (BUTTON_X_OFFSET, save_y), (BUTTON_X_OFFSET + BUTTON_WIDTH, save_y + BUTTON_HEIGHT),
                  save_color, -1)
    cv2.putText(ui_panel, 'Save to Excel', (BUTTON_X_OFFSET + 55, save_y + text_y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    y_pos = save_y + BUTTON_HEIGHT + 25

    # (이하 UI 로직은 기존과 동일)
    cv2.putText(ui_panel, "Real-time Analysis", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0),
                1)
    y_pos += 25
    cv2.putText(ui_panel, f"Posture Score: {posture_score:.2f}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    y_pos += 25
    cv2.putText(ui_panel, f"Neck Tilt: {neck_tilt:.2f}", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    if time.time() - save_feedback_time < 3.0:
        y_pos += 25
        cv2.putText(ui_panel, save_feedback_text, (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
    if show_results:
        y_pos += 20
        cv2.putText(ui_panel, "Final Results", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0),
                    1)
        y_pos += 25
        cv2.putText(ui_panel, f"Bad Posture: {format_time_str(posture_timer.forward_neck_time)}",
                    (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        y_pos += 20
        cv2.putText(ui_panel, f"Total Time:  {format_time_str(posture_timer.total_time)}", (BUTTON_X_OFFSET, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 30
        cv2.putText(ui_panel, "Bad Posture Log (Last 5)", (BUTTON_X_OFFSET, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)
        y_pos += 20
        for i, (start, end) in enumerate(posture_timer.bad_posture_log[-5:]):
            duration = end - start
            start_str = datetime.fromtimestamp(start).strftime('%H:%M:%S')
            cv2.putText(ui_panel, f"{start_str} ({duration:.1f}s)", (BUTTON_X_OFFSET, y_pos + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    else:
        y_pos_bottom = frame.shape[0] - 80
        cv2.putText(ui_panel, "--- Calibration ---", (BUTTON_X_OFFSET, y_pos_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 0), 1)
        y_pos_bottom += 25
        if not calibrated:
            cv2.putText(ui_panel, "Press 'c' to calibrate", (BUTTON_X_OFFSET, y_pos_bottom), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
        else:
            cv2.putText(ui_panel, "Calibrated! (Press 'c')", (BUTTON_X_OFFSET, y_pos_bottom), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
        y_pos_bottom += 25
        cv2.putText(ui_panel, f"FPS: {fps:.1f}", (BUTTON_X_OFFSET, y_pos_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    return cv2.hconcat([frame, ui_panel])


def main():
    global posture_score, neck_tilt, calibrated, timer_running, alarm_enabled, toast_alarm_enabled, show_results
    global calibrated_chin_shoulder_dist, calibrated_shoulder_width_px, calibrated_neck_tilt_offset

    toaster = ToastNotifier()
    icon_path = "posture_alert.ico" if os.path.exists("posture_alert.ico") else None
    VIDEO_SOURCE = 0;
    player = None;
    _alarm_local = None

    last_notification_time = 0
    NOTIFICATION_COOLDOWN = 5

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
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                landmarks = results.pose_landmarks.landmark
                ls, rs, ml, mr, nose = (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.MOUTH_LEFT], landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT],
                landmarks[mp_pose.PoseLandmark.NOSE])

                if all(lm.visibility > 0.7 for lm in [ls, rs, ml, mr, nose]):
                    h, w, _ = frame.shape
                    shoulder_mid_x, shoulder_mid_y = int((ls.x + rs.x) / 2 * w), int((ls.y + rs.y) / 2 * h)
                    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                    shoulder_mid_x_norm = (ls.x + rs.x) / 2
                    neck_tilt = (nose.x - shoulder_mid_x_norm) - calibrated_neck_tilt_offset
                    max_tilt = 0.08
                    tilt_ratio = max(0, 1 - (abs(neck_tilt) / max_tilt))
                    tilt_color = (0, 255 * tilt_ratio, 255 * (1 - tilt_ratio))
                    cv2.line(frame, (nose_x, nose_y), (shoulder_mid_x, shoulder_mid_y), tilt_color, 2)
                    shoulder_y_norm, chin_y_norm = (ls.y + rs.y) / 2, (ml.y + mr.y) / 2
                    dist_norm = abs(shoulder_y_norm - chin_y_norm)
                    min_dist, max_dist = 0.08, 0.15
                    if calibrated and calibrated_shoulder_width_px is not None and calibrated_shoulder_width_px > 0:
                        scale = abs((ls.x - rs.x) * w) / calibrated_shoulder_width_px
                        adj_dist = calibrated_chin_shoulder_dist * scale
                        max_dist, min_dist = adj_dist, adj_dist * 0.7
                    posture_score = max(0, min((dist_norm - min_dist) / (max_dist - min_dist + 1e-6), 1))
                    ls_px, rs_px = (int(ls.x * w), int(ls.y * h)), (int(rs.x * w), int(rs.y * h))
                    cv2.line(frame, ls_px, rs_px, (0, 255 * posture_score, 255 * (1 - posture_score)), 3)
                    if posture_score < 0.9 or abs(neck_tilt) > max_tilt / 2:
                        is_bad_posture = True
                        if timer_running: posture_timer.add_forward_neck_time(delta_time)

            if timer_running: posture_timer.update_posture_status(is_bad_posture, current_time)

            time_since_last_notification = current_time - last_notification_time
            if timer_running and toast_alarm_enabled and is_bad_posture and (
                    time_since_last_notification > NOTIFICATION_COOLDOWN):
                print(f"자세 나쁨 감지 ({NOTIFICATION_COOLDOWN}초 경과). Windows 알림을 보냅니다.")
                toaster.show_toast(
                    "자세 경고!!! ",
                    "자세가 좋지 않습니다. 고개를 들고 허리를 곧게 펴세요!",
                    icon_path=icon_path,
                    duration=None,
                    threaded=True
                )
                last_notification_time = current_time

            should_alarm_sound = timer_running and alarm_enabled and is_bad_posture
            if should_alarm_sound and not _alarm_local_on:
                _alarm_local.start(); _alarm_local_on = True
            elif not should_alarm_sound and _alarm_local_on:
                _alarm_local.stop(); _alarm_local_on = False

            display_frame = draw_ui(frame, fps)
            cv2.imshow(title, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]: break
            if key == ord('c'):
                if results.pose_landmarks and all(landmarks[i].visibility > 0.7 for i in [11, 12, 9, 10, 0]):
                    calibrated_chin_shoulder_dist = abs(
                        ((landmarks[11].y + landmarks[12].y) / 2) - ((landmarks[9].y + landmarks[10].y) / 2))
                    calibrated_shoulder_width_px = abs((landmarks[11].x - landmarks[12].x) * frame.shape[1])
                    calibrated_neck_tilt_offset = landmarks[0].x - (landmarks[11].x + landmarks[12].x) / 2
                    calibrated = True;
                    print("캘리브레이션 완료!")
                else:
                    print("캘리브레이션 실패: 얼굴과 어깨를 정면으로 보여주세요.")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if player: player.stop()
        if _alarm_local: _alarm_local.quit()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()