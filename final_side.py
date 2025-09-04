import torch
import openvino as ov
from utils.plots import Annotator, colors
from typing import List, Tuple
from utils.general import scale_boxes, non_max_suppression
from pathlib import Path
import numpy as np
from PIL import Image
from utils.augmentations import letterbox
from utils.general import yaml_save, yaml_load
import collections
import time
from IPython import display
import cv2
import threading

# VideoPlayer 클래스는 기존과 동일합니다.
class VideoPlayer:
    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0, width=1280, height=720):
        import cv2
        self.cv2 = cv2
        # self.__cap = cv2.VideoCapture(source)
        # http 주석 처리, 로컬 카메라 사용
        cap = cv2.VideoCapture(source) 
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 입력 버퍼 최소화(지연 감소)
        self.__cap = cap
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
        import cv2
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
        self.bad_posture_time = 0.0
        self.forward_head_time = 0.0
        self.posture_data = []
        self.last_detection_time = None
        self.total_time = 0.0

    def start_timer(self):
        self.start_time = time.time()
        self.timer_active = True
        self.bad_posture_time = 0.0
        self.forward_head_time = 0.0
        self.posture_data.clear()
        self.last_detection_time = self.start_time
        self.total_time = 0.0

    def stop_timer(self):
        self.timer_active = False
        self.total_time = time.time() - self.start_time if self.start_time else 0
        return {
            "bad_posture_time": self.bad_posture_time,
            "forward_head_time": self.forward_head_time,
            "total_time": self.total_time
        }

    def log_posture(self, posture_type):
        current_time = time.time()
        time_elapsed = current_time - (self.last_detection_time or current_time)
        self.last_detection_time = current_time

        if posture_type == 'bad_posture':
            self.bad_posture_time += time_elapsed
        elif posture_type == 'forward_head':
            self.forward_head_time += time_elapsed

# --- UI 및 타이머 상태 변수 ---
timer_running = False
show_results = False
posture_timer = PostureTimer(duration=1800)

# --- UI 상수 정의 ---
PANEL_WIDTH = 350
BUTTON_X_START = 1280 + 50
BUTTON_Y_START = 50
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 60
BUTTON_SPACING = 80

# --- 마우스 클릭 이벤트 핸들러 ---
def on_mouse_click(event, x, y, flags, param):
    global timer_running, show_results
    if event == cv2.EVENT_LBUTTONDOWN:
        # 'Start Timer' 버튼 클릭 감지
        if BUTTON_X_START <= x <= BUTTON_X_START + BUTTON_WIDTH and \
           BUTTON_Y_START <= y <= BUTTON_Y_START + BUTTON_HEIGHT and not timer_running:
            posture_timer.start_timer()
            timer_running = True
            show_results = False
            print("타이머 시작")
        # 'Stop Timer' 버튼 클릭 감지
        elif BUTTON_X_START <= x <= BUTTON_X_START + BUTTON_WIDTH and \
             (BUTTON_Y_START + BUTTON_SPACING) <= y <= (BUTTON_Y_START + BUTTON_SPACING + BUTTON_HEIGHT) and timer_running:
            posture_timer.stop_timer()
            timer_running = False
            show_results = True
            print("타이머 종료. 분석 결과: ", posture_timer.stop_timer())

# --- 이미지 전처리 및 모델 추론 함수 (기존과 동일) ---
def preprocess_image(img0: np.ndarray):
    img = letterbox(img0, auto=False)[0]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, img0

def prepare_input_tensor(image: np.ndarray):
    input_tensor = image.astype(np.float32)
    input_tensor /= 255.0
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor

def detect(model: ov.Model, image: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45, classes: List[int] = None, agnostic_nms: bool = False):
    preprocessed_img, orig_img = preprocess_image(image)
    input_tensor = prepare_input_tensor(preprocessed_img)
    predictions = torch.from_numpy(model(input_tensor)[0])
    pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    return pred, orig_img, input_tensor.shape

def draw_boxes(predictions: np.ndarray, input_shape: Tuple[int], image: np.ndarray, names: List[str]):
    if not len(predictions):
        return image
    annotator = Annotator(image, line_width=2, example=str(names))
    predictions[:, :4] = scale_boxes(input_shape[2:], predictions[:, :4], image.shape).round()
    for *xyxy, conf, cls in reversed(predictions):
        if conf >= 0.6:
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(int(cls), True))
    return image

# --- 새롭게 정의된 UI 드로잉 함수 ---
def draw_ui(frame, fps, processing_time):
    # UI 패널 생성 (검은색 배경)
    ui_panel = np.zeros((frame.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)

    # 버튼 그리기
    # Start Timer 버튼
    start_button_color = (0, 150, 0) if not timer_running else (50, 50, 50)
    cv2.rectangle(ui_panel, (50, BUTTON_Y_START), (50 + BUTTON_WIDTH, BUTTON_Y_START + BUTTON_HEIGHT), start_button_color, -1)
    cv2.putText(ui_panel, 'Start Timer', (75, BUTTON_Y_START + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Stop Timer 버튼
    stop_button_color = (0, 0, 150) if timer_running else (50, 50, 50)
    cv2.rectangle(ui_panel, (50, BUTTON_Y_START + BUTTON_SPACING), (50 + BUTTON_WIDTH, BUTTON_Y_START + BUTTON_SPACING + BUTTON_HEIGHT), stop_button_color, -1)
    cv2.putText(ui_panel, 'Stop Timer', (85, BUTTON_Y_START + BUTTON_SPACING + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 상태 및 결과 텍스트 표시
    y_pos = 300
    cv2.putText(ui_panel, "--- Status ---", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y_pos += 40
    timer_status = "Running" if timer_running else "Stopped"
    cv2.putText(ui_panel, f"Timer: {timer_status}", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_pos += 80
    cv2.putText(ui_panel, "--- Analysis Results ---", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y_pos += 40
    
    current_total_time = (time.time() - posture_timer.start_time) if timer_running else posture_timer.total_time
    
    if timer_running or show_results:
        cv2.putText(ui_panel, f"Bad Posture: {posture_timer.bad_posture_time:.1f} s", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        y_pos += 30
        cv2.putText(ui_panel, f"Forward Head: {posture_timer.forward_head_time:.1f} s", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        y_pos += 30
        cv2.putText(ui_panel, f"Total Time: {current_total_time:.1f} s", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 성능 정보 표시
    y_pos = frame.shape[0] - 100
    cv2.putText(ui_panel, "--- Performance ---", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y_pos += 40
    cv2.putText(ui_panel, f"Inference: {processing_time:.1f}ms", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30
    cv2.putText(ui_panel, f"FPS: {fps:.1f}", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 영상 프레임과 UI 패널을 가로로 연결
    combined_frame = cv2.hconcat([frame, ui_panel])
    return combined_frame

def main():
    # 모델 로드
    metadata = yaml_load('model/best_int8.yaml')
    NAMES = metadata["names"]
    core = ov.Core()
    ov_model = core.read_model('model/best_int8.xml')
    compiled_model = core.compile_model(ov_model, 'CPU')

    # 비디오 설정
    VIDEO_SOURCE = 0
    player = None

    try:
        player = VideoPlayer(source=VIDEO_SOURCE, flip=True, fps=30)
        player.start()
        
        title = "AI Side Posture Analysis"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, on_mouse_click)

        processing_times = collections.deque(maxlen=200)

        while True:
            frame = player.next()
            if frame is None:
                break
            
            # 영상 리사이즈 (가로 1280에 맞게)
            h, w, _ = frame.shape
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)

            input_image = np.array(frame)

            # 모델 추론
            start_time = time.time()
            detections, _, input_shape = detect(compiled_model, input_image[:, :, ::-1])
            stop_time = time.time()
            processing_times.append(stop_time - start_time)
            
            # 타이머가 실행 중일 때 자세 로깅
            if timer_running and detections and len(detections[0]) > 0:
                for det in detections[0]:
                    cls = int(det[-1])
                    if cls == 0:  # bad_position
                        posture_timer.log_posture('bad_posture')
                    elif cls == 2:  # forward_head
                        posture_timer.log_posture('forward_head')

            # 바운딩 박스 그리기
            frame = draw_boxes(detections[0], input_shape, input_image, NAMES)

            # 성능 계산
            processing_time_ms = np.mean(processing_times) * 1000
            fps = 1000 / processing_time_ms if processing_time_ms > 0 else 0

            # UI 그리기 및 화면 표시
            display_frame = draw_ui(frame, fps, processing_time_ms)
            cv2.imshow(title, display_frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC 키
                break

    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            player.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()