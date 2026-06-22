import time
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

from squat_angle_detector import SquatAngleDetector


EXERCISE_NAME = "MINI SQUAT"
SQUAT_ANGLE_MAX = 170.0   
SQUAT_ANGLE_MIN = 130.0  

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30


LANDMARK_VISIBILITY_FLOOR = 0.15

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def color_for_state(is_correct):
    # BGR format for OpenCV
    return (0, 200, 0) if is_correct else (0, 0, 230)  # green vs red


def draw_skeleton_with_feedback(image, landmarks_proto, is_correct):
    """Draws the full MediaPipe skeleton, colored green/red based on correctness."""
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


def draw_hud(image, exercise_name, knee_angle, is_correct, latency_ms, fps, cue_text):
    h, w = image.shape[:2]

    # Top banner: exercise name
    cv2.rectangle(image, (0, 0), (w, 50), (40, 40, 40), -1)
    cv2.putText(image, exercise_name, (15, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Correctness cue, top right
    status_color = color_for_state(is_correct)
    cv2.putText(image, cue_text, (w - 260, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2, cv2.LINE_AA)

    # Bottom-left: angle reading
    if knee_angle is not None:
        cv2.putText(image, f"Knee angle: {knee_angle:.1f} deg", (15, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "No pose detected", (15, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    # Bottom-left: latency / fps (your presentation numbers, live)
    cv2.putText(image, f"Latency: {latency_ms:.1f} ms   FPS: {fps:.1f}", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


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

    # Rolling average for a smoother on-screen FPS readout (raw per-frame
    # latency jitters a lot frame to frame; a short rolling window reads
    # more like what your audience will perceive)
    latency_window = []
    WINDOW_SIZE = 30

    print(f"Starting live demo: {EXERCISE_NAME}")
    print("Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # ===================== LATENCY TIMER =====================
            start_time = time.perf_counter()
            result = detector.process_frame(color_image, depth_frame, depth_image, side="auto")
            end_time = time.perf_counter()
            # ===========================================================

            latency_ms = (end_time - start_time) * 1000
            latency_window.append(latency_ms)
            if len(latency_window) > WINDOW_SIZE:
                latency_window.pop(0)
            avg_latency = sum(latency_window) / len(latency_window)
            live_fps = 1000 / avg_latency if avg_latency > 0 else 0

            display_image = color_image.copy()

            knee_angle = None
            is_correct = False
            cue_text = "Step into frame"

            if result is not None:
                knee_angle = result["knee_angle_deg"]
                is_correct = SQUAT_ANGLE_MIN <= knee_angle <= SQUAT_ANGLE_MAX

                if knee_angle > SQUAT_ANGLE_MAX:
                    cue_text = "GO LOWER"
                elif knee_angle < SQUAT_ANGLE_MIN:
                    cue_text = "TOO DEEP - EASE UP"
                else:
                    cue_text = "GOOD DEPTH"

                draw_skeleton_with_feedback(display_image, result["landmarks"], is_correct)

            draw_hud(display_image, EXERCISE_NAME, knee_angle, is_correct,
                     avg_latency, live_fps, cue_text)

            cv2.imshow("Smart Mirror - Live Demo", display_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()