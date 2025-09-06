import torch
from openvino.runtime import Core, Model, CompiledModel
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
            self.__interpolation = cv2.INTER_AREA if size[0] < self.__cap.get(
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
        self.total_time = 0.0

    def start_timer(self):
        """타이머를 시작하고 모든 값을 초기화합니다."""
        self.start_time = time.time()
        self.timer_active = True
        self.bad_posture_time = 0.0
        self.forward_head_time = 0.0
        self.total_time = 0.0

    def stop_timer(self):
        """타이머를 중지하고 최종 시간을 계산한 후 결과를 반환합니다."""
        self.timer_active = False
        if self.start_time:
            self.total_time = time.time() - self.start_time
        return {
            "bad_posture_time": self.bad_posture_time,
            "forward_head_time": self.forward_head_time,
            "total_time": self.total_time
        }

    def add_posture_time(self, posture_type, duration):
        """
        타이머가 활성화 상태일 때, 특정 자세에 대한 시간을 누적합니다.
        'bad_posture' 또는 'forward_head' 타입과 지속 시간(duration)을 받습니다.
        """
        if not self.timer_active:
            return

        if posture_type == 'bad_posture':
            self.bad_posture_time += duration
        elif posture_type == 'forward_head':
            self.forward_head_time += duration


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
                (BUTTON_Y_START + BUTTON_SPACING) <= y <= (
                BUTTON_Y_START + BUTTON_SPACING + BUTTON_HEIGHT) and timer_running:
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


def detect(model: CompiledModel, image: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45,
           classes: List[int] = None, agnostic_nms: bool = False):
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

def draw_ui(frame, fps, processing_time):
    # UI 패널 생성 (검은색 배경)
    ui_panel = np.zeros((frame.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)

    # --- 좌표 수정 ---
    # 모든 요소의 시작 x좌표를 50 -> 20으로 변경
    # 텍스트 x좌표도 그에 맞게 조절

    # 버튼 그리기
    # Start Timer 버튼
    start_button_color = (0, 150, 0) if not timer_running else (50, 50, 50)
    cv2.rectangle(ui_panel, (20, BUTTON_Y_START), (20 + BUTTON_WIDTH, BUTTON_Y_START + BUTTON_HEIGHT),
                  start_button_color, -1)
    cv2.putText(ui_panel, 'Start Timer', (45, BUTTON_Y_START + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Stop Timer 버튼
    stop_button_color = (0, 0, 150) if timer_running else (50, 50, 50)
    cv2.rectangle(ui_panel, (20, BUTTON_Y_START + BUTTON_SPACING),
                  (20 + BUTTON_WIDTH, BUTTON_Y_START + BUTTON_SPACING + BUTTON_HEIGHT), stop_button_color, -1)
    cv2.putText(ui_panel, 'Stop Timer', (55, BUTTON_Y_START + BUTTON_SPACING + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    # 상태 및 결과 텍스트 표시
    y_pos = 300
    cv2.putText(ui_panel, "--- Status ---", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y_pos += 40
    timer_status = "Running" if timer_running else "Stopped"
    cv2.putText(ui_panel, f"Timer: {timer_status}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y_pos += 80
    cv2.putText(ui_panel, "--- Analysis Results ---", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y_pos += 40

    current_total_time = (time.time() - posture_timer.start_time) if timer_running else posture_timer.total_time

    if timer_running or show_results:
        # format_time 함수를 사용하여 시간 표시 형식을 변경
        bad_posture_str = format_time(posture_timer.bad_posture_time)
        forward_head_str = format_time(posture_timer.forward_head_time)
        total_time_str = format_time(current_total_time)

        cv2.putText(ui_panel, f"Bad Posture:  {bad_posture_str}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 100, 255), 2)
        y_pos += 30
        cv2.putText(ui_panel, f"Forward Head: {forward_head_str}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 100, 255), 2)
        y_pos += 30
        cv2.putText(ui_panel, f"Total Time:   {total_time_str}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    # 성능 정보 표시
    y_pos = frame.shape[0] - 100
    cv2.putText(ui_panel, "--- Performance ---", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y_pos += 40
    cv2.putText(ui_panel, f"Inference: {processing_time:.1f}ms", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    y_pos += 30
    cv2.putText(ui_panel, f"FPS: {fps:.1f}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 영상 프레임과 UI 패널을 가로로 연결
    combined_frame = cv2.hconcat([frame, ui_panel])
    return combined_frame


# ===== (추가) Streamlit용 코어 클래스: 프레임 단위 처리 =====
class SideCore:
    """
    - OpenVINO YOLO 측면 감지를 프레임 단위로 수행
    - final_side.py의 detect()/draw_boxes() 유틸을 그대로 활용
    - 외부에서 start/stop/reset 제어 가능
    """

    def __init__(self, model_dir: str = "model", model_name: str = "best_int8"):
        # 라벨/모델 로드(파일 구조는 final_side.py의 main과 동일 가정)
        metadata = yaml_load(str(Path(model_dir) / f"{model_name}.yaml"))
        self.names = metadata["names"]
        core = Core()
        ov_model = core.read_model(str(Path(model_dir) / f"{model_name}.xml"))
        self.model = core.compile_model(ov_model, 'CPU')
        # 타이머/상태
        self.timer_active = False
        self.start_time = None
        self.total_time = 0.0
        self.bad_posture_time = 0.0
        self.forward_head_time = 0.0
        self.proc_times = collections.deque(maxlen=200)

    # ---- 외부 제어 ----
    def start_timer(self):
        self.timer_active = True
        self.start_time = time.time()
        self.total_time = 0.0
        self.bad_posture_time = 0.0
        self.forward_head_time = 0.0

    def stop_timer(self):
        self.timer_active = False

    def reset_all(self):
        self.timer_active = False
        self.start_time = None
        self.total_time = 0.0
        self.bad_posture_time = 0.0
        self.forward_head_time = 0.0
        self.proc_times.clear()

    def get_stats(self):
        return {
            "total": self.total_time,
            "bad": self.bad_posture_time,
            "forward": self.forward_head_time,
            "fps": (1000.0 / (np.mean(self.proc_times) * 1000) if self.proc_times else 0.0)
        }

    # ---- 프레임 처리 ----
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = frame_bgr.copy()
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        scale = 1280 / max(1, w)
        img = cv2.resize(img, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)

        # 추론
        t0 = time.time()
        detections, _, input_shape = detect(self.model, img[:, :, ::-1])
        t1 = time.time()
        self.proc_times.append(t1 - t0)

        # 타이머 누적
        now = time.time()
        if self.timer_active and self.start_time:
            self.total_time = now - self.start_time
        if self.timer_active and detections and len(detections[0]) > 0:
            for det in detections[0]:
                cls = int(det[-1])
                if cls == 0:  # bad_posture
                    self.bad_posture_time += (t1 - t0)  # 프레임 간격 대용
                elif cls == 2:  # forward_head
                    self.forward_head_time += (t1 - t0)

        # 박스/라벨
        out = draw_boxes(detections[0] if detections else np.array([]), input_shape, img, self.names)

        # HUD
        ms = (np.mean(self.proc_times) * 1000) if self.proc_times else 0.0
        fps = 1000.0 / ms if ms > 0 else 0.0
        cv2.putText(out, f"Infer {ms:5.1f} ms | {fps:4.1f} FPS",
                    (20, out.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(out,
                    f"Bad: {self.bad_posture_time:5.1f}s | F.Head: {self.forward_head_time:5.1f}s | Total: {self.total_time:5.1f}s",
                    (20, out.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        return out


def main():
    # 모델 로드
    metadata = yaml_load('model/best_int8.yaml')
    NAMES = metadata["names"]
    core = Core()
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

            # 1. 모델 추론 및 프레임 처리 시간 계산
            start_time = time.time()
            detections, _, input_shape = detect(compiled_model, input_image[:, :, ::-1])
            stop_time = time.time()

            # 이 프레임이 화면에 표시되는 시간(자세 지속 시간)으로 사용
            processing_duration = stop_time - start_time
            processing_times.append(processing_duration)

            # 2. 타이머가 실행 중일 때만 특정 자세 시간 누적
            if timer_running and detections and len(detections[0]) > 0:
                # 현재 프레임에서 감지된 모든 클래스 종류를 확인 (중복 방지)
                detected_classes = {int(det[-1]) for det in detections[0]}

                # 'caution_position' (bad_posture, 클래스 0)이 감지되면 bad_posture_time 누적
                if 0 in detected_classes:
                    posture_timer.add_posture_time('bad_posture', processing_duration)

                # 'bad_neck' (forward_head, 클래스 2)이 감지되면 forward_head_time 누적
                if 2 in detected_classes:
                    posture_timer.add_posture_time('forward_head', processing_duration)

            # 3. 바운딩 박스 및 UI 그리기
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
    main()