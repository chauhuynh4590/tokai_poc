from pathlib import Path
from typing import Tuple, Dict
import cv2
import numpy as np
from PIL import Image
from openvino.runtime import Core, Model
import collections
import time

# Import custom utilities
# from utils.notebook_utils import download_file, VideoPlayer
from yolo_ultils import VideoPlayer, draw_results, preprocess_image, image_to_tensor, postprocess

# Define constants
DET_MODEL_NAME = "v1"
label_map = {0: "barcode", 1: "tagname"}
models_dir = Path('./model')
models_dir.mkdir(exist_ok=True)

# Define paths and initialize OpenVINO
det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
core = Core()

# Load detection model
device_value = "CPU"
det_ov_model = core.read_model(det_model_path)
if device_value != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})


def detect(image: np.ndarray, model: Model) -> np.ndarray:
    """
    Perform object detection using OpenVINO YOLOv8 model.
    
    Parameters:
        image (np.ndarray): Input image.
        model (Model): OpenVINO compiled model.
    
    Returns:
        detections (np.ndarray): Detected boxes in format [x1, y1, x2, y2, score, label]
    """
    num_outputs = len(model.outputs)
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    result = model(input_tensor)
    boxes = result[model.output(0)]
    masks = None
    if num_outputs > 1:
        masks = result[model.output(1)]
    input_hw = input_tensor.shape[2:]
    detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
    return detections


def run_object_detection(source=0, flip=False, skip_first_frames=0, model=det_ov_model, device="AUTO"):
    if device != "CPU":
        model.reshape({0: [1, 3, 640, 640]})

    compiled_model = core.compile_model(model, device)

    try:
        player = VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        player.start()

        title = "Press ESC to Exit"
        cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()

        while True:
            frame = player.next()

            if frame is None:
                print("Source ended")
                break

            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(src=frame, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            input_image = np.array(frame)

            start_time = time.time()
            detections = detect(input_image[:, :, ::-1], compiled_model)[0]
            stop_time = time.time()

            image_with_boxes = draw_results(detections, input_image, label_map)
            frame = image_with_boxes

            processing_times.append(stop_time - start_time)

            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]

            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time

            cv2.putText(img=frame, text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                        org=(20, 40), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 1000,
                        color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

            cv2.imshow(winname=title, mat=frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break

    except KeyboardInterrupt:
        print("Interrupted")

    except RuntimeError as e:
        print(e)

    finally:
        if player is not None:
            player.stop()


        cv2.destroyAllWindows()


WEBCAM_INFERENCE = False

if WEBCAM_INFERENCE:
    VIDEO_SOURCE = 0  # Webcam
else:
    VIDEO_SOURCE = "demo.mp4"

run_object_detection(source=VIDEO_SOURCE, flip=True, model=det_ov_model, device=device_value)
