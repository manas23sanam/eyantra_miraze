import time
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import threading
import queue
import json

from squat_angle_detector import SquatAngleDetector, SquatRepCounter
from pose_geometry_base import EMAFilter

# Load configurations
with open('exercises_config.json', 'r') as f:
    config_data = json.load(f)

def select_exercise():
    print("\n" + "="*30)
    print("  SMART MIRROR EXERCISE TRACKER")
    print("="*30)
    keys = list(config_data.keys())
    for i, key in enumerate(keys):
        print(f"[{i + 1}] {config_data[key]['name']}")
    
    while True:
        try:
            choice = int(input(f"\nSelect an exercise [1-{len(keys)}]: "))
            if 1 <= choice <= len(keys):
                selected_key = keys[choice - 1]
                return config_data[selected_key]
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a valid number.")

squat_config = select_exercise()
EXERCISE_NAME = squat_config["name"]
USE_RATIO_MODE = squat_config.get("use_ratio_mode", False)

if USE_RATIO_MODE:
    SQUAT_MAX_VAL = squat_config["ratio_max"]
    SQUAT_MIN_VAL = squat_config["ratio_min"]
else:
    SQUAT_MAX_VAL = squat_config["angle_max"]
    SQUAT_MIN_VAL = squat_config["angle_min"]

STAND_THRESHOLD = squat_config["stand_threshold"]
SQUAT_THRESHOLD = squat_config["squat_threshold"]
MAX_ANGLE_DIFF = squat_config.get("max_angle_diff")
MAX_KNEE_DIST_MM = squat_config.get("max_knee_dist_mm")
MAX_TORSO_LEAN = squat_config.get("max_torso_lean")

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def color_for_state(is_correct):
    return (0, 200, 0) if is_correct else (0, 0, 230)

def draw_skeleton_with_feedback(image, landmarks_proto, is_correct):
    color = color_for_state(is_correct)
    landmark_style = mp_drawing.DrawingSpec(color=color, thickness=4, circle_radius=4)
    connection_style = mp_drawing.DrawingSpec(color=color, thickness=3)

    mp_drawing.draw_landmarks(
        image,
        landmarks_proto,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=landmark_style,
        connection_drawing_spec=connection_style,
    )

def draw_hud(image, exercise_name, current_val, is_correct, latency_ms, fps, cue_text, rep_count, state_text):
    h, w = image.shape[:2]

    # Top banner: exercise name
    cv2.rectangle(image, (0, 0), (w, 50), (40, 40, 40), -1)
    cv2.putText(image, exercise_name, (15, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Correctness cue, top right
    status_color = color_for_state(is_correct)
    cv2.putText(image, cue_text, (w - 380, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2, cv2.LINE_AA)

    # Bottom-left: angle/ratio reading and rep count
    if current_val is not None:
        val_text = f"Ratio: {current_val:.2f}" if USE_RATIO_MODE else f"Angle: {current_val:.1f} deg"
        cv2.putText(image, f"{val_text} | State: {state_text.upper()}", (15, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"REPS: {rep_count}", (15, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "No pose detected", (15, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    # Bottom-left: latency / fps
    cv2.putText(image, f"Latency: {latency_ms:.1f} ms   FPS: {fps:.1f}", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def camera_capture_thread(pipeline, align, frame_queue, stop_event):
    """Thread to continuously capture frames from RealSense and put in queue."""
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            # If queue is full, drop the oldest frame to keep latency low
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            frame_queue.put((color_frame, depth_frame))
        except Exception as e:
            if not stop_event.is_set():
                print(f"Camera thread error: {e}")


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

    detector = SquatAngleDetector(depth_intrinsics=intrinsics)
    rep_counter = SquatRepCounter(stand_threshold=STAND_THRESHOLD, 
                                  squat_threshold=SQUAT_THRESHOLD,
                                  angle_min=SQUAT_MIN_VAL,
                                  max_angle_diff=MAX_ANGLE_DIFF,
                                  max_knee_dist_mm=MAX_KNEE_DIST_MM,
                                  max_torso_lean=MAX_TORSO_LEAN)
    
    # We use a smaller alpha for ratios since they are a smaller scale
    angle_smoother = EMAFilter(alpha=0.3 if not USE_RATIO_MODE else 0.1)

    # Setup multithreading for camera capture
    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    cam_thread = threading.Thread(target=camera_capture_thread, args=(pipeline, align, frame_queue, stop_event))
    cam_thread.daemon = True
    cam_thread.start()

    latency_window = []
    WINDOW_SIZE = 30

    print(f"Starting live demo: {EXERCISE_NAME}")
    print("Press 'q' to quit.")

    try:
        while True:
            try:
                # Get latest frame from queue
                color_frame, depth_frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            start_time = time.perf_counter()
            result = detector.process_frame(color_image, depth_frame, depth_image, side="auto")
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latency_window.append(latency_ms)
            if len(latency_window) > WINDOW_SIZE:
                latency_window.pop(0)
            avg_latency = sum(latency_window) / len(latency_window)
            live_fps = 1000 / avg_latency if avg_latency > 0 else 0

            display_image = color_image.copy()

            raw_metric = (result["vertical_ratio"] if USE_RATIO_MODE else result["knee_angle_deg"]) if result else None
            angle_diff = result["angle_diff"] if result else None
            knee_dist = result["knee_dist_mm"] if result else None
            torso_lean = result["torso_lean_deg"] if result else None
            
            # 1. Smooth the metric
            smoothed_metric = angle_smoother.update(raw_metric)
            
            is_correct = False
            cue_text = "Step into frame"
            state_text = "None"
            rep_count = rep_counter.rep_count

            if smoothed_metric is not None:
                # 2. Update state machine with new metrics
                state_text, rep_count, just_completed = rep_counter.update(smoothed_metric, angle_diff, knee_dist, torso_lean)
                
                # 3. Give per-frame correctness feedback based on state and metric
                is_correct = True  # Default to green overlay
                
                if state_text == "standing":
                    cue_text = "READY TO SQUAT"
                elif state_text == "squatting":
                    if smoothed_metric > SQUAT_MAX_VAL:
                        cue_text = "GO LOWER"
                    elif smoothed_metric < SQUAT_MIN_VAL:
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
                    draw_skeleton_with_feedback(display_image, result["landmarks"], is_correct)

            draw_hud(display_image, EXERCISE_NAME, smoothed_metric, is_correct,
                     avg_latency, live_fps, cue_text, rep_count, state_text)

            cv2.imshow("Smart Mirror - Live Demo", display_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        cam_thread.join(timeout=1.0)
        pipeline.stop()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()