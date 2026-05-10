# Ensure eventlet monkey patching happens first
# import eventlet

# eventlet.monkey_patch()

# Now import other modules
import argparse
import os
import sys
import platform
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO, emit
from shapely.geometry import Polygon
import json
import time

# Determine if we're on Windows or Linux
is_windows = platform.system() == "Windows"

# Flask app initialization
app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--Input", default=None, help="Path to input image, video, or None for camera")
parser.add_argument("--model-version", choices=["v8", "v11"], default="v8", help="YOLO model version (v8 or v11)")
parser.add_argument("--ip", default="127.0.0.1", help="IP address to run the server on")
parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
args = parser.parse_args()

# Settings
draw_helmet = 1
draw_vest = 1
text_name_format = "{:s}_{:s}.{:s}"
screenshot_name_format = "{:s}-{:%Y%m%d-%H%M%S}.{:s}"
camera_index = 0
input_type = ""
input_name = ""
img_rh = 720
img_rw = 1200
video_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video')
current_video_path = None
img_oh = 0
img_ow = 0
cameras = []

# Model settings
class ModelSettings:
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5

custom_model_img_size = 512
stream_jpeg_quality = 80
preprocessed_video_names = {
    "important_tb_detected_cam1_step5.mp4",
    "260510 1.mp4",
}

def should_stream_without_analysis(source_path):
    if not source_path:
        return False
    return os.path.basename(source_path).lower() in preprocessed_video_names

torch = None
YOLO = None
device = 'cpu'

def ensure_torch_loaded():
    global torch, device
    if torch is None:
        import torch as torch_module
        torch = torch_module
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch

def ensure_ultralytics_loaded():
    global YOLO
    ensure_torch_loaded()
    if YOLO is None:
        from ultralytics import YOLO as YOLOClass
        YOLO = YOLOClass
    return YOLO

# Camera device detection (platform-specific)
def get_available_cameras():
    cameras = []
    # Try to detect cameras using OpenCV's approach
    if is_windows:
        # Import pygrabber only on Windows
        try:
            from pygrabber.dshow_graph import FilterGraph
            devices = FilterGraph().get_input_devices()
            for i, device in enumerate(devices):
                cameras.append({"index": i, "name": device})
        except ImportError:
            print("Warning: pygrabber not installed, falling back to OpenCV camera detection")
            index = 0
            while True:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    break
                cameras.append({"index": index, "name": f"Camera {index}"})
                cap.release()
                index += 1
    else:
        # Linux approach (works for Jetson Nano)
        import glob
        video_devices = glob.glob('/dev/video*')
        for i, device in enumerate(sorted(video_devices)):
            index = int(device.split('video')[1])
            # Try to open to verify it works
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cameras.append({"index": index, "name": f"Camera {index}"})
                cap.release()
    
    return cameras

# Determine input type
if args.Input is not None:
    if os.path.exists(args.Input):
        input_ext = os.path.splitext(args.Input)[1].lower()
        if input_ext in (".png", ".jpg", ".jpeg"):
            input_type = "image"
            input_name = os.path.basename(args.Input)
        elif input_ext in (".mp4", ".avi", ".mov"):
            input_type = "video"
            input_name = os.path.basename(args.Input)
        else:
            print("Wrong input format, need to be in [*.png, *.jpg, *.jpeg, *.mp4, *.avi, *.mov]")
            exit()
    else:
        print("Input not exist, exiting...")
        exit()
else:
    cameras = get_available_cameras()
    if cameras:
        input_type = "camera"
        camera_index = cameras[0]["index"]
        input_name = cameras[0]["name"]
    else:
        print("No camera detected, exiting...")
        exit()

# Ensure all directories exist with platform-independent paths
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Directories
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')
detected_dir = os.path.join(base_dir, 'detected')
zone_dir = os.path.join(base_dir, 'zone')
screenshots_dir = os.path.join(base_dir, 'static', 'screenshots')
stats_dir = os.path.join(base_dir, 'static', 'stats')

for d in [models_dir, detected_dir, zone_dir, screenshots_dir, stats_dir]:
    ensure_dir(d)
    
# Check if video folder exists, create if not
ensure_dir(video_folder)

# Model paths
person_model_path = os.path.join(models_dir, 'yolov11s.pt')
ppe_model_path = os.path.join(models_dir, 'ppe_v11s.pt')
custom_model_path = os.path.join(models_dir, 'epoch_079.pt')

# Check model files
if not os.path.exists(person_model_path):
    raise FileNotFoundError(f"The person model file '{person_model_path}' does not exist.")
if not os.path.exists(ppe_model_path):
    raise FileNotFoundError(f"The PPE model file '{ppe_model_path}' does not exist.")

# Zone file
zone_text_path = os.path.join(zone_dir, text_name_format.format("zone", input_name, "txt"))
if not os.path.exists(zone_text_path):
    with open(zone_text_path, "w"):
        pass

# Global zone variables
zone_list = []
zone_current_count = []
zone_last_count = []
haveZone = False

# Create path for stats storage
stats_file = os.path.join(stats_dir, 'ppe_stats_history.json')
stats_history = []
stats_last_save_time = None

# Define the load_stats_history function BEFORE it gets called
def load_stats_history():
    global stats_history
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                stats_history = json.load(f)
                
            # Keep only last 24 hours of data
            cutoff_time = datetime.now() - timedelta(hours=24)
            stats_history = [entry for entry in stats_history 
                            if datetime.fromisoformat(entry['timestamp']) > cutoff_time]
        except Exception as e:
            print(f"Error loading stats history: {e}")
            stats_history = []
    else:
        stats_history = []

# Keep the load_zones function unchanged
def load_zones():
    global zone_list, zone_current_count, zone_last_count, haveZone
    zone_list = []
    if os.path.exists(zone_text_path):
        with open(zone_text_path, "r") as f:
            zone_list = [eval(line.strip()) for line in f.readlines() if line.strip()]
    haveZone = len(zone_list) > 0
    zone_current_count = [0] * len(zone_list)
    zone_last_count = [0] * len(zone_list)

# Call both functions separately
load_zones()
load_stats_history()

# Initialize YOLO models only when the selected source actually needs analysis.
analysis_required = not (input_type == "video" and should_stream_without_analysis(args.Input))
use_custom_detection_model = False
person_model = None
ppe_model = None
custom_detection_model = None
custom_detection_names = {}
custom_letterbox = None
custom_non_max_suppression = None
custom_scale_coords = None

if use_custom_detection_model:
    print(f"Custom detection model is available: {custom_model_path}")
    print("It will be loaded lazily only when analysis is required.")
elif analysis_required:
    YOLO = ensure_ultralytics_loaded()
    person_model = YOLO(person_model_path).to(device)
    ppe_model = YOLO(ppe_model_path).to(device)
else:
    print("Preprocessed video selected; skipping YOLO/PyTorch model loading.")

def ensure_custom_detection_model_loaded():
    global custom_detection_model, custom_detection_names
    global custom_letterbox, custom_non_max_suppression, custom_scale_coords

    if custom_detection_model is not None:
        return

    ensure_torch_loaded()

    yolov7_dir = os.path.join(base_dir, 'vendor', 'yolov7')
    if not os.path.exists(yolov7_dir):
        raise FileNotFoundError(
            f"YOLOv7 compatibility code was not found at '{yolov7_dir}'."
        )
    if yolov7_dir not in sys.path:
        sys.path.insert(0, yolov7_dir)

    from utils.datasets import letterbox as custom_letterbox
    from utils.general import non_max_suppression as custom_non_max_suppression
    from utils.general import scale_coords as custom_scale_coords

    try:
        checkpoint = torch.load(custom_model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(custom_model_path, map_location=device)

    if isinstance(checkpoint, dict):
        custom_detection_model = checkpoint.get('ema') or checkpoint.get('model')
    else:
        custom_detection_model = checkpoint
    custom_detection_model = custom_detection_model.float().to(device).eval()
    custom_detection_names = getattr(custom_detection_model, 'names', {})
    print(f"Loaded custom detection model: {custom_model_path}")
    print(f"Custom classes: {custom_detection_names}")

# Global variables
notification_active = False
notification_timestamp = None
notification_count = 0
ppe_stats = {"with_ppe": 0, "without_ppe": 0, "total_persons": 0, "date": ""}
last_alert_time = None

# Overlap functions
def PPE_overlap(object_box, area_box):
    ox1, oy1, ox2, oy2 = object_box
    ax1, ay1, ax2, ay2 = area_box
    ix1 = max(ox1, ax1)
    iy1 = max(oy1, ay1)
    ix2 = min(ox2, ax2)
    iy2 = min(oy2, ay2)
    inter_width = max(0, ix2 - ix1)
    inter_height = max(0, iy2 - iy1)
    intersection_area = inter_width * inter_height
    object_area = (ox2 - ox1) * (oy2 - oy1)
    return intersection_area / object_area if object_area > 0 else 0

def ZONE_overlap(person_box, zone_coord):
    px1, py1, px2, py2 = person_box
    person = Polygon([(px1, py1), (px2, py1), (px2, py2), (px1, py2)])
    zone = Polygon(zone_coord)
    if person.intersects(zone):
        return person.intersection(zone).area / person.area
    return 0

def save_screenshot(frame, zone_idx=None):
    prefix = f"zone{zone_idx}" if zone_idx is not None else "alert"
    screenshot_name = screenshot_name_format.format(prefix, datetime.now(), "jpg")
    screenshot_path = os.path.join(screenshots_dir, screenshot_name)
    cv2.imwrite(screenshot_path, frame)
    return screenshot_name

def class_color(class_id):
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (231, 76, 60),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
    ]
    return palette[int(class_id) % len(palette)]

def get_custom_class_name(class_id):
    if isinstance(custom_detection_names, dict):
        return custom_detection_names.get(int(class_id), str(int(class_id)))
    if isinstance(custom_detection_names, (list, tuple)) and int(class_id) < len(custom_detection_names):
        return custom_detection_names[int(class_id)]
    return str(int(class_id))

def draw_detection_label(frame, box, label, color):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_w, text_h = text_size
    label_y1 = max(0, y1 - text_h - 8)
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

def process_custom_frame(frame, mode="ppe"):
    global ppe_stats
    ensure_custom_detection_model_loaded()

    zone_frame_arr = [frame.copy() for _ in zone_list]
    zone_current_count[:] = [0] * len(zone_list)

    img = custom_letterbox(frame, custom_model_img_size, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        predictions = custom_detection_model(img)[0]
        detections = custom_non_max_suppression(
            predictions,
            ModelSettings.conf_threshold,
            ModelSettings.iou_threshold
        )[0]

    total_detections = 0
    person_count = 0
    if detections is not None and len(detections):
        detections[:, :4] = custom_scale_coords(img.shape[2:], detections[:, :4], frame.shape).round()
        total_detections = len(detections)

        for *xyxy, conf, cls in detections:
            class_id = int(cls)
            class_name = get_custom_class_name(class_id)
            color = class_color(class_id)
            label = f"{class_name} {float(conf):.2f}"
            draw_detection_label(frame, xyxy, label, color)

            if class_name.lower() == "person":
                person_count += 1
                person_bbox = [int(v) for v in xyxy]
                for i, zone_coord in enumerate(zone_list):
                    if ZONE_overlap(person_bbox, zone_coord) > 0.5:
                        zone_current_count[i] += 1

    for i, zone_coord in enumerate(zone_list):
        zone_color = (255, 0, 255) if zone_current_count[i] > 0 else (255, 255, 255)
        cv2.polylines(frame, [np.array(zone_coord, dtype=np.int32)], True, zone_color, 2)
        cv2.polylines(zone_frame_arr[i], [np.array(zone_coord, dtype=np.int32)], True, zone_color, 2)
        cv2.putText(frame, f"Zone {i} ({zone_current_count[i]})", (zone_coord[0][0], zone_coord[0][1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, zone_color, 2)

    ppe_stats["total_persons"] = person_count if person_count else total_detections
    ppe_stats["with_ppe"] = 0
    ppe_stats["without_ppe"] = 0
    ppe_stats["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_stats_history()
    return frame, zone_frame_arr, False

def process_frame(frame, mode="ppe"):
    global ppe_stats, notification_active, notification_timestamp, notification_count, last_alert_time
    if use_custom_detection_model:
        return process_custom_frame(frame, mode)

    PPE_overlap_threshold = 0.4
    ZONE_overlap_threshold = 0.5
    zone_frame_arr = [frame.copy() for _ in zone_list]
    roi_has_unsafe = False

    # Reset zone_current_count for this frame
    zone_current_count[:] = [0] * len(zone_list)

    # Person detection
    person_results = person_model(frame, device=device)
    person_result = person_results[0]
    person_bboxes = np.array(person_result.boxes.xyxy.cpu(), dtype="int")
    person_classes = np.array(person_result.boxes.cls.cpu(), dtype="int")
    person_scores = np.array(person_result.boxes.conf.cpu(), dtype="float")
    person_indices = np.where(person_classes == 0)[0]
    person_bboxes = person_bboxes[person_indices]
    person_scores = person_scores[person_indices]

    ppe_stats["total_persons"] = len(person_bboxes)
    ppe_stats["with_ppe"] = 0
    ppe_stats["without_ppe"] = 0

    # PPE detection
    ppe_results = ppe_model(
        frame, 
        device=device, 
        imgsz=640, 
        conf=ModelSettings.conf_threshold, 
        iou=ModelSettings.iou_threshold
    )
    ppe_result = ppe_results[0]
    ppe_bboxes = np.array(ppe_result.boxes.xyxy.cpu(), dtype="int")
    ppe_classes = np.array(ppe_result.boxes.cls.cpu(), dtype="int")
    ppe_scores = np.array(ppe_result.boxes.conf.cpu(), dtype="float")
    helmet_bboxes = ppe_bboxes[np.where(ppe_classes == 0)[0]]
    vest_bboxes = ppe_bboxes[np.where(ppe_classes == 1)[0]]

    for person_bbox, person_score in zip(person_bboxes, person_scores):
        wearing_helmet = False
        wearing_vest = False

        for helmet_bbox in helmet_bboxes:
            if PPE_overlap(helmet_bbox, person_bbox) > PPE_overlap_threshold:
                wearing_helmet = True
                break
        for vest_bbox in vest_bboxes:
            if PPE_overlap(vest_bbox, person_bbox) > PPE_overlap_threshold:
                wearing_vest = True
                break

        if wearing_helmet and wearing_vest:
            ppe_stats["with_ppe"] += 1
        else:
            ppe_stats["without_ppe"] += 1
            
        person_in_danger = False
        for i, zone_coord in enumerate(zone_list):
            overlap_ratio = ZONE_overlap(person_bbox, zone_coord)
            if overlap_ratio > ZONE_overlap_threshold:
                # Increment count for this zone in the current frame
                zone_current_count[i] += 1
                if not (wearing_helmet and wearing_vest) and mode == "ppe":
                    person_in_danger = True
                    roi_has_unsafe = True
                    cv2.rectangle(zone_frame_arr[i], (person_bbox[0], person_bbox[1]),
                                  (person_bbox[2], person_bbox[3]), (0, 0, 255), 2)
                    cv2.putText(zone_frame_arr[i], "No PPE", (person_bbox[0], person_bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                elif mode == "danger":
                    person_in_danger = True
                    cv2.rectangle(zone_frame_arr[i], (person_bbox[0], person_bbox[1]),
                                  (person_bbox[2], person_bbox[3]), (0, 0, 255), 2)
                    cv2.putText(zone_frame_arr[i], "IN DANGER ZONE", (person_bbox[0], person_bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        color = (0, 255, 0) if wearing_helmet and wearing_vest else (0, 0, 255)
        cv2.rectangle(frame, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), color, 2)
        label = "PPE" if wearing_helmet and wearing_vest else "No PPE"
        cv2.putText(frame, label, (person_bbox[0], person_bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    if draw_helmet:
        for h_bbox in helmet_bboxes:
            cv2.rectangle(frame, (h_bbox[0], h_bbox[1]), (h_bbox[2], h_bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, "Helmet", (h_bbox[0], h_bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    if draw_vest:
        for v_bbox in vest_bboxes:
            cv2.rectangle(frame, (v_bbox[0], v_bbox[1]), (v_bbox[2], v_bbox[3]), (255, 255, 0), 2)
            cv2.putText(frame, "Vest", (v_bbox[0], v_bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

    for i, zone_coord in enumerate(zone_list):
        zone_color = (255, 0, 255) if zone_current_count[i] > 0 else (255, 255, 255)
        cv2.polylines(frame, [np.array(zone_coord, dtype=np.int32)], True, zone_color, 2)
        cv2.polylines(zone_frame_arr[i], [np.array(zone_coord, dtype=np.int32)], True, zone_color, 2)
        cv2.putText(frame, f"Zone {i} ({zone_current_count[i]})", (zone_coord[0][0], zone_coord[0][1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, zone_color, 2)

    # Trigger alerts based on changes in counts
    if roi_has_unsafe or any(c > 0 for c in zone_current_count):
        current_time = datetime.now()
        if last_alert_time is None or (current_time - last_alert_time).total_seconds() > 3:
            last_alert_time = current_time
            notification_active = True
            notification_timestamp = current_time
            notification_count += 1
            message = "Person without PPE in zone" if mode == "ppe" else "Person in danger zone"
            for i, count in enumerate(zone_current_count):
                if count > 0:  # Only emit alerts for zones with people
                    screenshot_name = save_screenshot(zone_frame_arr[i], i)
                    socketio.emit('safety_alert', {
                        'active': True,
                        'timestamp': notification_timestamp.isoformat(),
                        'count': notification_count,
                        'message': f"{message} (Zone {i})",
                        'screenshot': f"screenshots/{screenshot_name}"
                    })
            zone_last_count[:] = zone_current_count[:]

    ppe_stats["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_stats_history()
    return frame, zone_frame_arr, roi_has_unsafe

# Special camera source creator for Jetson Nano
def create_camera_source(camera_index):
    if "jetson" in platform.machine().lower():
        # Use gstreamer pipeline for Jetson Nano
        return cv2.VideoCapture(
            f"nvarguscamerasrc sensor-id={camera_index} ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1", 
            cv2.CAP_GSTREAMER
        )
    else:
        # Normal camera for other platforms
        return cv2.VideoCapture(camera_index)

def get_video_playback_metadata(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None or fps <= 0:
        fps = 30.0
    return fps, total_frames

def read_realtime_video_frame(cap, playback_state):
    fps = playback_state["fps"]
    total_frames = playback_state["total_frames"]

    while True:
        elapsed = time.time() - playback_state["started_at"]
        target_frame = int(elapsed * fps)

        if total_frames > 0 and target_frame >= total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            playback_state["started_at"] = time.time()
            playback_state["last_frame"] = -1
            target_frame = 0

        if target_frame > playback_state["last_frame"]:
            break

        next_frame_time = (playback_state["last_frame"] + 1) / fps
        sleep_time = max(0.001, min(0.01, next_frame_time - elapsed))
        time.sleep(sleep_time)

    if target_frame != playback_state["last_frame"] + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    success, frame = cap.read()
    if success:
        playback_state["last_frame"] = target_frame
        return frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    playback_state["started_at"] = time.time()
    playback_state["last_frame"] = -1
    return None

def encode_stream_frame(frame):
    display = cv2.resize(frame, (img_rw, img_rh))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), stream_jpeg_quality]
    ret, buffer = cv2.imencode('.jpg', display, encode_params)
    if not ret:
        return None
    return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'

def generate_frames(mode="ppe"):
    if input_type == "image":
        frame = cv2.imread(args.Input)
        processed_frame, _, _ = process_frame(frame, mode)
        display = cv2.resize(processed_frame, (img_rw, img_rh))
        ret, buffer = cv2.imencode('.jpg', display)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    # Use current_video_path if available, otherwise use args.Input or camera
    if input_type == "video":
        video_source = current_video_path if current_video_path else args.Input
        cap = cv2.VideoCapture(video_source)
        stream_without_analysis = should_stream_without_analysis(video_source)
    else:  # camera
        cap = create_camera_source(camera_index)
        stream_without_analysis = False
    
    if not cap.isOpened():
        raise RuntimeError('Could not start camera or video.')

    playback_state = None
    if input_type == "video":
        fps, total_frames = get_video_playback_metadata(cap)
        playback_state = {
            "fps": fps,
            "total_frames": total_frames,
            "started_at": time.time(),
            "last_frame": -1
        }
    
    frame_count = 0
    while True:
        if input_type == "video":
            frame = read_realtime_video_frame(cap, playback_state)
            if frame is None:
                continue
            
            elapsed = time.time() - playback_state["started_at"]
            
            # Determine phase based on timeline
            phase = 1
            if 0 <= elapsed < 7:
                phase = 1
            elif 7 <= elapsed < 14:
                phase = 2
            elif 14 <= elapsed < 22:
                phase = 3
            elif 22 <= elapsed < 29:
                phase = 4
            elif 29 <= elapsed <= 35: # Allow some buffer
                phase = 5
                
            socketio.emit('timeline_state', {"elapsed": elapsed, "phase": phase})
        else:
            success, frame = cap.read()
            if not success:
                break

        if stream_without_analysis:
            processed_frame = frame
        else:
            processed_frame, _, _ = process_frame(frame, mode)
        stream_frame = encode_stream_frame(processed_frame)
        if stream_frame is None:
            continue

        frame_count += 1
        if frame_count % 5 == 0:
            time.sleep(0.01)
        yield stream_frame

def generate_raw_frames():
    if input_type == "image":
        frame = cv2.imread(args.Input)
        stream_frame = encode_stream_frame(frame)
        if stream_frame:
            yield stream_frame
        return

    # Use current_video_path if available, otherwise use args.Input or camera
    if input_type == "video":
        video_source = current_video_path if current_video_path else args.Input
        cap = cv2.VideoCapture(video_source)
    else:  # camera
        cap = create_camera_source(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError('Could not start camera or video.')

    playback_state = None
    if input_type == "video":
        fps, total_frames = get_video_playback_metadata(cap)
        playback_state = {
            "fps": fps,
            "total_frames": total_frames,
            "started_at": time.time(),
            "last_frame": -1
        }
    
    frame_count = 0
    while True:
        if input_type == "video":
            frame = read_realtime_video_frame(cap, playback_state)
            if frame is None:
                continue
        else:
            success, frame = cap.read()
            if not success:
                break

        stream_frame = encode_stream_frame(frame)
        if stream_frame is None:
            continue

        frame_count += 1
        if frame_count % 5 == 0:
            eventlet.sleep(0.01)
        yield stream_frame

# Routes
@app.route('/')
def index():
    return render_template('index_v2.html')

@app.route('/get_available_videos')
def get_available_videos():
    videos = []
    if os.path.exists(video_folder):
        for file in os.listdir(video_folder):
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                videos.append({
                    'name': file,
                    'path': os.path.join(video_folder, file)
                })
    return jsonify({'videos': videos})

@app.route('/set_video_source', methods=['POST'])
def set_video_source():
    global input_type, input_name, current_video_path, zone_text_path, camera_index, cameras
    data = request.json
    source_type = data.get('source_type')
    
    if source_type == 'camera':
        input_type = 'camera'
        # Get first available camera
        if cameras:
            camera_index = cameras[0]["index"]
            input_name = cameras[0]["name"]
        else:
            return jsonify({"error": "No camera detected"}), 400
        current_video_path = None
    elif source_type == 'video':
        video_path = data.get('video_path')
        if video_path and os.path.exists(video_path):
            input_type = 'video'
            input_name = os.path.basename(video_path)
            current_video_path = video_path
        else:
            return jsonify({"error": "Invalid video path"}), 400
    
    # Update zone text path for the new input
    zone_text_path = os.path.join(zone_dir, text_name_format.format("zone", input_name, "txt"))
    if not os.path.exists(zone_text_path):
        with open(zone_text_path, "w"):
            pass
    
    # Reload zones for the new input
    load_zones()
    
    return jsonify({"status": "success", "source": input_type})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames("ppe"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/danger_zone_feed')
def danger_zone_feed():
    return Response(generate_frames("danger"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/manage_video_feed')
def manage_video_feed():
    return Response(generate_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/manage')
def manage():
    global img_ow, img_oh
    # Use current_video_path if available, otherwise use args.Input or camera
    if input_type == "video" or input_type == "image":
        video_source = current_video_path if input_type == "video" and current_video_path else args.Input
        cap = cv2.VideoCapture(video_source)
    else:  # camera
        cap = create_camera_source(camera_index)
        
    success, frame = cap.read()
    if success:
        img_oh, img_ow = frame.shape[:2]
        cap.release()
        return render_template('manage.html', img_ow=img_ow, img_oh=img_oh)
    return "Error loading video", 500

@app.route('/save_zone', methods=['POST'])
def save_zone():
    global img_ow, img_oh
    try:
        data = request.json
        coordinates = data.get('coordinates', [])
        if not coordinates:
            return jsonify({"error": "No coordinates provided"}), 400
        
        # Adjust coordinates for original image dimensions
        # Use current_video_path if available
        x_ratio = img_ow / img_rw
        y_ratio = img_oh / img_rh
        adjusted_coords = [[int(pt[0] * x_ratio), int(pt[1] * y_ratio)] for pt in coordinates]
        
        # Save the zone to the file
        with open(zone_text_path, "a") as f:
            f.write(f"{adjusted_coords}\n")
        
        load_zones()  # Reload zones
        socketio.emit('zone_updated', {'message': 'Zones have been updated'})
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_zone', methods=['POST'])
def delete_zone():
    try:
        data = request.json
        zone_index = data.get('index')
        if zone_index is None:
            return jsonify({"error": "No zone index provided"}), 400
        
        # Read existing zones
        zones = []
        if os.path.exists(zone_text_path):
            with open(zone_text_path, "r") as f:
                zones = [eval(line.strip()) for line in f.readlines() if line.strip()]
        
        # Delete the specified zone
        if 0 <= zone_index < len(zones):
            zones.pop(zone_index)
            with open(zone_text_path, "w") as f:
                for zone in zones:
                    f.write(f"{zone}\n")
            
            load_zones()  # Reload zones
            socketio.emit('zone_updated', {'message': 'Zones have been updated'})
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "Invalid zone index"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_zones')
def get_zones():
    global img_ow, img_oh
    try:
        zones = []
        if os.path.exists(zone_text_path):
            with open(zone_text_path, "r") as f:
                zones = [eval(line.strip()) for line in f.readlines() if line.strip()]
        # Adjust coordinates for display (scale down to canvas size)
        # Use current_video_path if available
        x_ratio = img_ow / img_rw
        y_ratio = img_oh / img_rh
        adjusted_zones = [[[int(pt[0] / x_ratio), int(pt[1] / y_ratio)] for pt in zone] for zone in zones]
        
        return jsonify({"zones": adjusted_zones})
    except:
        return jsonify({"zones": []})

@app.route('/get_initial_frame')
def get_initial_frame():
    # Use current_video_path if available, otherwise use args.Input or camera
    if input_type == "video" or input_type == "image":
        video_source = current_video_path if input_type == "video" and current_video_path else args.Input
        cap = cv2.VideoCapture(video_source)
    else:  # camera
        cap = create_camera_source(camera_index)
        
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (img_rw, img_rh))
        ret, buffer = cv2.imencode('.jpg', frame)
        cap.release()
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "Error loading frame", 500

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/get_ppe_stats')
def get_ppe_stats():
    return jsonify(ppe_stats)

@app.route('/screenshots')
def show_screenshots():
    screenshots = [f for f in os.listdir(screenshots_dir) if f.endswith('.jpg')]
    return render_template('screenshots.html', screenshots=screenshots)

@app.route('/api/notifications/clear', methods=['POST'])
def clear_notifications():
    global notification_active, notification_count
    notification_active = False
    notification_count = 0
    socketio.emit('clear_alerts', {'success': True})
    return jsonify({'success': True})

@app.route('/api/notifications/status')
def notification_status():
    return jsonify({
        'active': notification_active,
        'timestamp': notification_timestamp.isoformat() if notification_timestamp else None,
        'count': notification_count
    })

@app.route('/update_model_settings', methods=['POST'])
def update_model_settings():
    data = request.json
    try:
        ModelSettings.conf_threshold = float(data.get('conf', 0.8))
        ModelSettings.iou_threshold = float(data.get('iou', 0.4))
        return jsonify({"status": "success"})
    except ValueError:
        return jsonify({"error": "Invalid threshold values"}), 400

@app.route('/get_current_source')
def get_current_source():
    source_info = {
        'type': input_type,
        'name': input_name
    }
    if input_type == 'video' and current_video_path:
        source_info['path'] = current_video_path
    return jsonify(source_info)

@app.route('/get_available_cameras')
def get_available_cameras_route():
    global cameras
    return jsonify({"cameras": cameras})

@app.route('/set_camera', methods=['POST'])
def set_camera():
    global input_type, input_name, current_video_path, zone_text_path, camera_index
    data = request.json
    camera_idx = data.get('camera_index')
    
    if camera_idx is not None:
        try:
            camera_idx = int(camera_idx)
            # Test if camera can be opened
            cap = cv2.VideoCapture(camera_idx)
            if not cap.isOpened():
                cap.release()
                return jsonify({"error": "Could not open camera"}), 400
            cap.release()
            
            input_type = 'camera'
            camera_index = camera_idx
            input_name = f"Camera {camera_idx}"
            current_video_path = None
            
            # Update zone text path for the new input
            zone_text_path = os.path.join(zone_dir, text_name_format.format("zone", input_name, "txt"))
            if not os.path.exists(zone_text_path):
                with open(zone_text_path, "w"):
                    pass
            
            # Reload zones for the new input
            load_zones()
            
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "No camera index provided"}), 400

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('safety_alert', {
        'active': notification_active,
        'timestamp': notification_timestamp.isoformat() if notification_timestamp else None,
        'count': notification_count
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def save_stats_history():
    global stats_history, stats_last_save_time
    current_time = datetime.now()
    
    # Only save once per minute to avoid excessive writes
    if stats_last_save_time is None or (current_time - stats_last_save_time).total_seconds() > 60:
        stats_last_save_time = current_time
        
        # Add current stats to history with timestamp
        stats_entry = ppe_stats.copy()
        stats_entry['timestamp'] = current_time.isoformat()
        stats_history.append(stats_entry)
        
        # Keep only last 24 hours of data (1440 minutes)
        cutoff_time = current_time - timedelta(hours=24)
        stats_history = [entry for entry in stats_history 
                        if datetime.fromisoformat(entry['timestamp']) > cutoff_time]
        
        # Save to file
        with open(stats_file, 'w') as f:
            json.dump(stats_history, f)

@app.route('/get_ppe_stats_history')
def get_ppe_stats_history():
    return jsonify(stats_history)

if __name__ == '__main__':
    # Print information about running environment
    print(f"Running on {platform.system()} ({platform.machine()})")
    print(f"Python version: {platform.python_version()}")
    print(f"OpenCV version: {cv2.__version__}")
    if torch is None:
        print("PyTorch not loaded: stream-only mode is active.")
    else:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # Start the server
    print(f"Starting server at http://{args.ip}:{args.port}")
    socketio.run(app, host=args.ip, port=args.port, debug=True, allow_unsafe_werkzeug=True)
