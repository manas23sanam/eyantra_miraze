import time
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import threading
import queue

from pose_geometry_base import EMAFilter
from exercise_utils import load_config, select_exercise, get_exercise_components, process_frame_logic, SessionTracker

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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

def draw_hud(image, exercise_name, current_val, metric_name, is_correct, latency_ms, fps, cue_text, rep_count, state_text):
    h, w = image.shape[:2]

    # Top banner: exercise name
    cv2.rectangle(image, (0, 0), (w, 50), (40, 40, 40), -1)
    cv2.putText(image, exercise_name, (15, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Correctness cue, top right
    status_color = color_for_state(is_correct)
    cv2.putText(image, cue_text, (w - 380, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2, cv2.LINE_AA)

    # Bottom-left: metric reading and rep count
    if current_val is not None:
        val_text = f"{metric_name}: {current_val:.1f}"
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
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            frame_queue.put((color_frame, depth_frame))
        except Exception as e:
            if not stop_event.is_set():
                print(f"Camera thread error: {e}")


def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    return pipeline, align, intrinsics


def main():
    config_data = load_config()
    selected_key, exercise_config = select_exercise(config_data)
    exercise_name = exercise_config["name"]
    
    pipeline, align, intrinsics = init_camera()

    detector, rep_counter = get_exercise_components(selected_key, intrinsics, exercise_config)
    angle_smoother = EMAFilter(alpha=0.3)

    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    cam_thread = threading.Thread(target=camera_capture_thread, args=(pipeline, align, frame_queue, stop_event))
    cam_thread.daemon = True
    cam_thread.start()

    latency_window = []
    WINDOW_SIZE = 30

    print(f"Starting live demo: {exercise_name}")
    print("Press 'q' to quit.")

    last_rep_count = 0
    tracker = SessionTracker()

    try:
        while True:
            try:
                color_frame, depth_frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            start_time = time.perf_counter()
            result = detector.process_frame(color_image, depth_frame, depth_image)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latency_window.append(latency_ms)
            if len(latency_window) > WINDOW_SIZE:
                latency_window.pop(0)
            avg_latency = sum(latency_window) / len(latency_window)
            live_fps = 1000 / avg_latency if avg_latency > 0 else 0

            display_image = color_image.copy()

            smoothed_metric, metric_name, is_correct, cue_text, state_text, rep_count = process_frame_logic(
                result, rep_counter, angle_smoother, selected_key, exercise_config
            )
            
            just_completed = (rep_count > last_rep_count)
            last_rep_count = rep_count
            tracker.update(state_text, cue_text, just_completed)

            if result and "landmarks" in result and smoothed_metric is not None:
                draw_skeleton_with_feedback(display_image, result["landmarks"], is_correct)

            draw_hud(display_image, exercise_name, smoothed_metric, metric_name, is_correct,
                     avg_latency, live_fps, cue_text, rep_count, state_text)

            cv2.imshow("Smart Mirror - Live Demo", display_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        cam_thread.join(timeout=1.0)
        pipeline.stop()
        if hasattr(detector, 'close'):
            detector.close()
        cv2.destroyAllWindows()
        
        tracker.print_summary()

if __name__ == "__main__":
    main()