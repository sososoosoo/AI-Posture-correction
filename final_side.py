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

class VideoPlayer:
    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0, width=1280, height=720):
        import cv2
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

class PostureTimer:
    def __init__(self, duration):
        self.start_time = None
        self.duration = duration
        self.timer_active = False
        self.bad_posture_time = 0.0  # 초 단위 누적 시간으로 변경
        self.forward_head_time = 0.0  # 초 단위 누적 시간으로 변경
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
        self.total_time = 0.0  # 초기화

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


# 타이머 시작 및 종료 버튼 상태
timer_running = False
show_results = False  # 결과 표시 여부를 관리하는 플래그
collecting_data = False

def on_mouse_click(event, x, y, flags, param):
    global timer_running, posture_timer, show_results, collecting_data
    if event == cv2.EVENT_LBUTTONDOWN:
        # 'Start Timer' 버튼 클릭 감지
        if 50 <= x <= 200 and 50 <= y <= 100 and not timer_running:
            posture_timer.start_timer()
            timer_running = True
            show_results = False  # 화면에서 결과를 지우기
            collecting_data = True
            print("타이머 시작")
        # 'Stop Timer' 버튼 클릭 감지
        elif 50 <= x <= 200 and 150 <= y <= 200 and timer_running:
            result = posture_timer.stop_timer()
            timer_running = False
            show_results = True  # 화면에 결과 표시
            collecting_data = False
            print("타이머 종료. 분석 결과: ", result)

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

def detect(
    model: ov.Model,
    image_path: Path,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: List[int] = None,
    agnostic_nms: bool = False,
):
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = np.array(Image.open(image_path))
    preprocessed_img, orig_img = preprocess_image(img)
    input_tensor = prepare_input_tensor(preprocessed_img)
    predictions = torch.from_numpy(model(input_tensor)[0])
    pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    return pred, orig_img, input_tensor.shape

def draw_boxes(
    predictions: np.ndarray,
    input_shape: Tuple[int],
    image: np.ndarray,
    names: List[str],
):
    if not len(predictions):
        return image

    annotator = Annotator(image, line_width=1, example=str(names))
    predictions[:, :4] = scale_boxes(input_shape[2:], predictions[:, :4], image.shape).round()

    for *xyxy, conf, cls in reversed(predictions):
        # 신뢰도가 0.6 이상인 경우에만 표시
        if conf >= 0.6:
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(int(cls), True))
    return image

def draw_results_on_frame(frame):
    """
    결과를 화면에 표시하는 함수.
    Stop Timer 버튼이 눌렸을 때만 호출됨.
    """

    frame_width = frame.shape[1]

    # 데이터 수집 중일 때 표시
    if collecting_data:
        cv2.putText(
            img=frame,
            text="Collecting data...",
            org=(frame_width - 300, 80),  # 오른쪽 끝에 표시
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 0, 255),  # 녹색 텍스트
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return  # 데이터 수집 중일 때는 나머지 결과는 표시하지 않음

    # 타이머가 활성화된 상태에서는 실시간으로 total_time을 갱신, 비활성화된 상태에서는 고정된 값을 사용
    if posture_timer.timer_active:
        current_total_time = time.time() - posture_timer.start_time if posture_timer.start_time else 0
    else:
        current_total_time = posture_timer.total_time

    cv2.putText(
        img=frame,
        text=f"Bad Posture Time: {posture_timer.bad_posture_time:.1f} s",
        org=(frame_width - 300, 80),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 0, 255),  # 빨간색 텍스트
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        img=frame,
        text=f"Forward Head Time: {posture_timer.forward_head_time:.1f} s",
        org=(frame_width - 300, 110),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 0, 255),  # 빨간색 텍스트
        thickness=2,
        lineType=cv2.LINE_AA,
    )


    cv2.putText(
        img=frame,
        text=f"Total Time: {current_total_time:.1f} s",
        org=(frame_width - 300, 140),  # 오른쪽 끝으로 이동
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 0, 255),  # 빨간색 텍스트
        thickness=2,
        lineType=cv2.LINE_AA,
    )



metadata = yaml_load('model/best_int8.yaml')
NAMES = metadata["names"]
ov_model_path = 'model/best_int8.xml'

core = ov.Core()
ov_model = core.read_model(ov_model_path)
compiled_model = core.compile_model(ov_model, 'CPU')

VIDEO_SOURCE = 0  
flip = True
use_popup = True
skip_first_frames = 0
player = None

posture_timer = PostureTimer(duration=1800)  # 예: 30분 타이머

try:
    player = VideoPlayer(source=VIDEO_SOURCE, flip=flip, fps=30, skip_first_frames=skip_first_frames)
    player.start()
    
    if use_popup:
        title = "AI Solution Hackaton"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, on_mouse_click)

    processing_times = collections.deque()

    while True:
        frame = player.next()
        if frame is None:
            print("Source ended")
            break
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(
                src=frame,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        input_image = np.array(frame)

        start_time = time.time()
        detections, _, input_shape = detect(compiled_model, input_image[:, :, ::-1])
        stop_time = time.time()

        if posture_timer.timer_active:
             for det in detections[0]:
                 cls = int(det[-1])
             if cls == 0:  # bad_position
                posture_timer.log_posture('bad_posture')
             elif cls == 2:  # bad_neck
                posture_timer.log_posture('forward_head')



        image_with_boxes = draw_boxes(detections[0], input_shape, input_image, NAMES)
        frame = image_with_boxes

        # 버튼 UI 그리기
        cv2.rectangle(frame, (50, 50), (200, 100), (0, 255, 0), -1)
        cv2.putText(frame, 'Start Timer', (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.rectangle(frame, (50, 150), (200, 200), (0, 0, 255), -1)
        cv2.putText(frame, 'Stop Timer', (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 결과 표시가 활성화되었을 때만 화면에 결과 표시
       
        draw_results_on_frame(frame)

        processing_times.append(stop_time - start_time)
        if len(processing_times) > 200:
            processing_times.popleft()

        _, f_width = frame.shape[:2]
        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time
        cv2.putText(
            img=frame,
            text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=f_width / 1000,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        if use_popup:
            cv2.imshow(title, frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            _, encoded_img = cv2.imencode(ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
            i = display.Image(data=encoded_img)
            display.clear_output(wait=True)
            display.display(i)

except KeyboardInterrupt:
    print("Interrupted")
except RuntimeError as e:
    print(e)
finally:
    if player is not None:
        player.stop()
    if use_popup:
        cv2.destroyAllWindows()
