import time
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import queue
import json
from flask import Flask, Response, render_template, jsonify

from squat_angle_detector import SquatAngleDetector, SquatRepCounter
from pose_geometry_base import EMAFilter

app = Flask(__name__)

# Global variables for the shared state
current_metric = None
is_correct = True
cue_text = "Initializing..."
state_text = "None"
rep_count = 0
latency_ms = 0.0
live_fps = 0.0

# Load config
with open('exercises_config.json', 'r') as f:
    config_data = json.load(f)

# Hardcoded for FULL SQUAT for now in web app, or we can use config
squat_config = config_data["full_squat"]
EXERCISE_NAME = squat_config["name"]
USE_RATIO_MODE = squat_config.get("use_ratio_mode", False)

SQUAT_MAX_VAL = squat_config.get("ratio_max", squat_config.get("angle_max", 100))
SQUAT_MIN_VAL = squat_config.get("ratio_min", squat_config.get("angle_min", 60))

STAND_THRESHOLD = squat_config["stand_threshold"]
SQUAT_THRESHOLD = squat_config["squat_threshold"]
MAX_ANGLE_DIFF = squat_config.get("max_angle_diff")
MAX_KNEE_DIST_MM = squat_config.get("max_knee_dist_mm")
MAX_TORSO_LEAN = squat_config.get("max_torso_lean")

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Queues for video streaming
jpeg_queue = queue.Queue(maxsize=1)

def camera_tracking_loop():
    global current_metric, is_correct, cue_text, state_text, rep_count, latency_ms, live_fps

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

    detector = SquatAngleDetector(depth_intrinsics=intrinsics)
    rep_counter = SquatRepCounter(
        stand_threshold=STAND_THRESHOLD, 
        squat_threshold=SQUAT_THRESHOLD,
        angle_min=SQUAT_MIN_VAL,
        max_angle_diff=MAX_ANGLE_DIFF,
        max_knee_dist_mm=MAX_KNEE_DIST_MM,
        max_torso_lean=MAX_TORSO_LEAN
    )
    
    angle_smoother = EMAFilter(alpha=0.3 if not USE_RATIO_MODE else 0.1)

    # For drawing the skeleton
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    def draw_skeleton(image, landmarks_proto, is_correct_form):
        color = (0, 200, 0) if is_correct_form else (0, 0, 230)
        landmark_style = mp_drawing.DrawingSpec(color=color, thickness=4, circle_radius=4)
        connection_style = mp_drawing.DrawingSpec(color=color, thickness=3)
        mp_drawing.draw_landmarks(
            image,
            landmarks_proto,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style,
        )

    latency_window = []
    
    print("RealSense camera tracking started.")

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            start_time = time.perf_counter()
            result = detector.process_frame(color_image, depth_frame, depth_image, side="auto")
            end_time = time.perf_counter()

            lat = (end_time - start_time) * 1000
            latency_window.append(lat)
            if len(latency_window) > 30:
                latency_window.pop(0)
            avg_lat = sum(latency_window) / len(latency_window)
            fps_val = 1000 / avg_lat if avg_lat > 0 else 0
            
            latency_ms = avg_lat
            live_fps = fps_val

            display_image = color_image.copy()

            raw_metric = (result["vertical_ratio"] if USE_RATIO_MODE else result["knee_angle_deg"]) if result else None
            angle_diff = result["angle_diff"] if result else None
            knee_dist = result["knee_dist_mm"] if result else None
            torso_lean = result["torso_lean_deg"] if result else None
            
            current_metric = angle_smoother.update(raw_metric)
            is_correct = True
            
            if current_metric is not None:
                state_text, rep_count, just_completed = rep_counter.update(current_metric, angle_diff, knee_dist, torso_lean)
                
                if state_text == "standing":
                    cue_text = "READY TO SQUAT"
                elif state_text == "squatting":
                    if current_metric > SQUAT_MAX_VAL:
                        cue_text = "GO LOWER"
                    elif current_metric < SQUAT_MIN_VAL:
                        cue_text = "TOO DEEP - EASE UP"
                        is_correct = False
                    else:
                        cue_text = "GOOD DEPTH"
                elif state_text == "invalid_depth":
                    cue_text = "WENT TOO DEEP - STAND UP TO RESET"
                    is_correct = False
                elif state_text == "invalid_asymmetric":
                    cue_text = "UNEVEN SQUAT - STAND UP TO RESET"
                    is_correct = False
                elif state_text == "invalid_wide_knees":
                    cue_text = "KNEES TOO WIDE - STAND UP TO RESET"
                    is_correct = False
                elif state_text == "invalid_back_posture":
                    cue_text = "BACK BENT TOO FAR - STAND UP"
                    is_correct = False

                if result and "landmarks" in result:
                    draw_skeleton(display_image, result["landmarks"], is_correct)
            else:
                cue_text = "Step into frame"
                state_text = "None"

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', display_image)
            if ret:
                if jpeg_queue.full():
                    try:
                        jpeg_queue.get_nowait()
                    except queue.Empty:
                        pass
                jpeg_queue.put(buffer.tobytes())

    except Exception as e:
        print(f"Tracking thread crashed: {e}")
    finally:
        pipeline.stop()
        detector.close()


def generate_mjpeg():
    """Generator for MJPEG streaming."""
    while True:
        try:
            frame_bytes = jpeg_queue.get(timeout=2.0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            continue

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html', exercise_name=EXERCISE_NAME)

@app.route('/video_feed')
def video_feed():
    """Route for the MJPEG stream."""
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    """Returns the current tracker state as JSON."""
    return jsonify({
        "metric": round(current_metric, 2) if current_metric is not None else None,
        "is_correct": is_correct,
        "cue_text": cue_text,
        "state_text": state_text.upper(),
        "rep_count": rep_count,
        "latency_ms": round(latency_ms, 1),
        "fps": round(live_fps, 1),
        "use_ratio_mode": USE_RATIO_MODE
    })


if __name__ == '__main__':
    # Start tracking in background
    tracker_thread = threading.Thread(target=camera_tracking_loop)
    tracker_thread.daemon = True
    tracker_thread.start()
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)
