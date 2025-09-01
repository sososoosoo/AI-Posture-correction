
import cv2
import mediapipe as mp
import time
import collections
import threading
import numpy as np

# MediaPipe Pose 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- 캘리브레이션 관련 전역 변수 ---
calibrated = False
calibrated_chin_shoulder_dist = None
calibrated_shoulder_width_px = None
calibration_feedback_text = ""
calibration_feedback_time = 0

class VideoPlayer:
    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0, width=1280, height=720):
        self.cv2 = cv2
        self.__cap = cv2.VideoCapture(source)
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

# 타이머 및 UI 관련 전역 변수
timer_running = False
show_results = False
collecting_data = False
posture_timer = PostureTimer(duration=1800)

def on_mouse_click(event, x, y, flags, param):
    global timer_running, posture_timer, show_results, collecting_data
    if event == cv2.EVENT_LBUTTONDOWN:
        if 50 <= x <= 200 and 50 <= y <= 100 and not timer_running:
            posture_timer.start_timer()
            timer_running = True
            show_results = False
            collecting_data = True
            print("타이머 시작")
        elif 50 <= x <= 200 and 150 <= y <= 200 and timer_running:
            result = posture_timer.stop_timer()
            timer_running = False
            show_results = True
            collecting_data = False
            print("타이머 종료. 분석 결과: ", result)

def draw_results_on_frame(frame):
    frame_width = frame.shape[1]

    if collecting_data:
        cv2.putText(frame, "Collecting data...", (frame_width - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return

    if show_results:
        current_total_time = posture_timer.total_time
        cv2.putText(frame, f"Forward Neck Time: {posture_timer.forward_neck_time:.1f} s", (frame_width - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Time: {current_total_time:.1f} s", (frame_width - 300, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def main():
    global timer_running, show_results, collecting_data
    global calibrated, calibrated_chin_shoulder_dist, calibrated_shoulder_width_px, calibration_feedback_text, calibration_feedback_time

    VIDEO_SOURCE = 0
    flip = True
    use_popup = True
    player = None

    try:
        player = VideoPlayer(source=VIDEO_SOURCE, flip=flip, fps=30)
        player.start()
        
        title = "Posture Monitoring with MediaPipe"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, on_mouse_click)

        processing_times = collections.deque()
        last_time = time.time()

        while True:
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            
            # --- 프레임 간 시간 계산 ---
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = pose.process(rgb_frame)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(220,220,220), thickness=2, circle_radius=2)
            )

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
                mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]

                if all(lm.visibility > 0.7 for lm in [left_shoulder, right_shoulder, mouth_left, mouth_right]):
                    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                    chin_y = (mouth_left.y + mouth_right.y) / 2
                    distance = abs(shoulder_y - chin_y)

                    min_dist, max_dist = 0.08, 0.15 # 기본값

                    if calibrated:
                        current_shoulder_width_px = abs((left_shoulder.x - right_shoulder.x) * frame_width)
                        if calibrated_shoulder_width_px > 0:
                            scale_factor = current_shoulder_width_px / calibrated_shoulder_width_px
                            adjusted_ideal_dist = calibrated_chin_shoulder_dist * scale_factor
                            max_dist = adjusted_ideal_dist
                            min_dist = adjusted_ideal_dist * 0.7

                    ratio = max(0, min((distance - min_dist) / (max_dist - min_dist), 1))
                    red_color = 255 * (1 - ratio)
                    green_color = 255 * ratio
                    shoulder_color = (0, green_color, red_color)

                    left_shoulder_px = (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height))
                    right_shoulder_px = (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height))
                    cv2.line(frame, left_shoulder_px, right_shoulder_px, shoulder_color, 3)

                    # --- 목 기울기(좌우) 계산 및 시각화 ---
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
                    
                    # 코와 어깨 중심의 수평 거리로 기울기 계산
                    tilt = nose.x - shoulder_mid_x
                    
                    # 기울기에 따른 색상 계산
                    max_tilt_threshold = 0.08 # 이 값 이상 기울어지면 완전한 빨간색
                    tilt_ratio = max(0, 1 - (abs(tilt) / max_tilt_threshold))
                    tilt_color = (0, 255 * tilt_ratio, 255 * (1 - tilt_ratio))

                    # 선 그리기
                    nose_px = (int(nose.x * frame_width), int(nose.y * frame_height))
                    shoulder_mid_px = (int(shoulder_mid_x * frame_width), int(shoulder_y * frame_height))
                    cv2.line(frame, nose_px, shoulder_mid_px, tilt_color, 2)

                    # 기울기 값 텍스트로 표시
                    cv2.putText(frame, f"Neck Tilt: {tilt:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"Posture Score: {ratio:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # --- 새로운 타이머 로직 ---
                    if ratio < 0.9:
                        posture_timer.add_forward_neck_time(delta_time)

            # --- UI 텍스트 그리기 ---
            if not calibrated:
                cv2.putText(frame, "Press 'c' to calibrate good posture", (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"Calibrated. Press 'c' to re-calibrate", (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if time.time() - calibration_feedback_time < 2.0:
                cv2.putText(frame, calibration_feedback_text, (20, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.rectangle(frame, (50, 50), (200, 100), (0, 255, 0), -1)
            cv2.putText(frame, 'Start Timer', (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.rectangle(frame, (50, 150), (200, 200), (0, 0, 255), -1)
            cv2.putText(frame, 'Stop Timer', (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            draw_results_on_frame(frame)
            
            # FPS 계산 (delta_time 사용)
            fps = 1 / delta_time if delta_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- 키보드 입력 처리 ---
            if use_popup:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                
                if key == ord('c'):
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
                        mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
                        if all(lm.visibility > 0.7 for lm in [left_shoulder, right_shoulder, mouth_left, mouth_right]):
                            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                            chin_y = (mouth_left.y + mouth_right.y) / 2
                            calibrated_chin_shoulder_dist = abs(shoulder_y - chin_y)
                            l_s_px = left_shoulder.x * frame_width
                            r_s_px = right_shoulder.x * frame_width
                            calibrated_shoulder_width_px = abs(l_s_px - r_s_px)
                            calibrated = True
                            calibration_feedback_text = "Calibrated!"
                            calibration_feedback_time = time.time()
                        else:
                            calibration_feedback_text = "Cannot see face/shoulders clearly."
                            calibration_feedback_time = time.time()
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
        if use_popup:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
