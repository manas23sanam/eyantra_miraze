import time
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import queue
from flask import Flask, Response, render_template, jsonify
from core.exercise_utils import load_config, select_exercise, create_exercise, SessionTracker
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
app = Flask(__name__)
current_metric = None
metric_name = 'Value'
is_correct = True
cue_text = 'Initializing...'
state_text = 'None'
rep_count = 0
last_rep_count = 0
latency_ms = 0.0
live_fps = 0.0
tracker = SessionTracker()
EXERCISE_NAME = ''
SELECTED_KEY = ''
EXERCISE_CONFIG = {}
USE_RATIO_MODE = False
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
jpeg_queue = queue.Queue(maxsize=1)

def draw_skeleton(image, landmarks_proto, is_correct_form):
    color = (0, 200, 0) if is_correct_form else (0, 0, 230)
    landmark_style = mp_drawing.DrawingSpec(color=color, thickness=4, circle_radius=4)
    connection_style = mp_drawing.DrawingSpec(color=color, thickness=3)
    mp_drawing.draw_landmarks(image, landmarks_proto, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=landmark_style, connection_drawing_spec=connection_style)

def camera_tracking_loop():
    global current_metric, metric_name, is_correct, cue_text, state_text, rep_count, latency_ms, live_fps, last_rep_count
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    
    exercise = create_exercise(SELECTED_KEY, intrinsics, EXERCISE_CONFIG)
    
    latency_window = []
    print(f'RealSense camera tracking started for {EXERCISE_NAME}.')
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
            
            result = exercise.process_frame(color_image, depth_frame, depth_image)
            
            end_time = time.perf_counter()
            lat = (end_time - start_time) * 1000
            latency_window.append(lat)
            if len(latency_window) > 30:
                latency_window.pop(0)
            avg_lat = sum(latency_window) / len(latency_window)
            latency_ms = avg_lat
            live_fps = 1000 / avg_lat if avg_lat > 0 else 0
            display_image = color_image.copy()
            
            current_metric = result['metric_value']
            metric_name = result['metric_name']
            is_correct = result['is_correct']
            cue_text = result['cue_text']
            state_text = result['state_text']
            rep_count = result['rep_count']
            just_completed = result['just_completed']
            landmarks = result['landmarks']
            
            last_rep_count = rep_count
            tracker.update(state_text, cue_text, just_completed)
            if landmarks and (current_metric is not None):
                draw_skeleton(display_image, landmarks, is_correct)
            ret, buffer = cv2.imencode('.jpg', display_image)
            if ret:
                if jpeg_queue.full():
                    try:
                        jpeg_queue.get_nowait()
                    except queue.Empty:
                        pass
                jpeg_queue.put(buffer.tobytes())
    except Exception as e:
        print(f'Tracking thread crashed: {e}')
    finally:
        pipeline.stop()
        if hasattr(exercise.detector, 'close'):
            exercise.detector.close()

def generate_mjpeg():
    while True:
        try:
            frame_bytes = jpeg_queue.get(timeout=2.0)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            continue

@app.route('/')
def index():
    return render_template('index.html', exercise_name=EXERCISE_NAME)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    return jsonify({'metric': round(current_metric, 2) if current_metric is not None else None, 'is_correct': is_correct, 'cue_text': cue_text, 'state_text': state_text.upper() if state_text else 'NONE', 'rep_count': rep_count, 'latency_ms': round(latency_ms, 1), 'fps': round(live_fps, 1), 'use_ratio_mode': USE_RATIO_MODE})

@app.route('/api/summary')
def api_summary():
    total_attempts = tracker.correct_reps + tracker.incorrect_reps
    accuracy = tracker.correct_reps / total_attempts * 100 if total_attempts > 0 else 0.0
    return jsonify({'correct_reps': tracker.correct_reps, 'incorrect_reps': tracker.incorrect_reps, 'accuracy': round(accuracy, 1), 'mistake_counts': tracker.mistake_counts})

if __name__ == '__main__':
    config_data = load_config()
    SELECTED_KEY, EXERCISE_CONFIG = select_exercise(config_data)
    EXERCISE_NAME = EXERCISE_CONFIG['name']
    USE_RATIO_MODE = EXERCISE_CONFIG.get('use_ratio_mode', False)
    print('\nStarting Web Dashboard...')
    print('Open http://127.0.0.1:5000 in your browser.')
    print('-' * 30)
    tracker_thread = threading.Thread(target=camera_tracking_loop)
    tracker_thread.daemon = True
    tracker_thread.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
