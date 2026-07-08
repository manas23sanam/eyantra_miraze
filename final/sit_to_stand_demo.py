"""
final_sit_to_stand.py

TWO-PHASE SESSION:

  PHASE 1 - TEACH (from a recording):
    Plays your side-view .db3 recording for 12 seconds (or press SPACE
    to skip early) so the person can SEE what correct form looks like.
    No scoring happens here - it's purely a demonstration.

  PHASE 2 - PRACTICE (live camera):
    Automatically switches to the live D435i feed. The person now does
    the exercise themselves, tracked in real time with the same
    legs/back/arms checklist, color-coded skeleton, live rep counter,
    and an end-of-session accuracy summary when they press 'q'.

REP COUNTING RULE: a rep (one full standing -> sitting -> standing
cycle) counts as CORRECT if at least 40% of its frames passed the
checklist - this is a deliberately lenient buffer so minor wobbles or
brief tracking noise don't unfairly disqualify an otherwise reasonable
attempt, while a rep that's mostly wrong still won't count.

SETUP: this script needs BOTH a working .db3 file path (for the teach
clip) AND a live D435i camera plugged in (for the practice phase) when
you actually run it.
"""

import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import ctypes
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))

from sit_to_stand_detector_v2 import SitToStandDetectorV2, SitToStandChecklist

# ============================================================
# CONFIG
# ============================================================
EXERCISE_NAME = "SIT TO STAND"

# Path to the TEACHING recording (phase 1 only). Confirm this exact
# filename matches what's actually inside your dataset folder - open
# the folder in File Explorer and copy the name directly if unsure.
DATASET_FOLDER = r"C:\Users\BIT\OneDrive\Desktop\software mirror\dataset"
TEACH_CLIP_FILE = "sit to stand_side_manas_1.db3"
TEACH_CLIP_DURATION_SEC = 12

# Calibrate these using validate_sit_to_stand.py against your sample videos.
LEG_STAND_THRESHOLD = 160.0
LEG_SIT_THRESHOLD = 110.0
BACK_MIN_ANGLE = 150.0
ARM_RAISE_MIN = 70.0
ARM_RAISE_MAX = 110.0

# Lenient buffer: a rep counts as correct if 40%+ of its frames passed.
REP_CORRECT_FRACTION = 0.40

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
LIVE_FPS = 30

WINDOW_NAME = "Mirage - Sit to Stand"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 230)
COLOR_AMBER = (0, 165, 255)


def get_screen_resolution():
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1366, 768


def color_for_state(is_correct):
    return COLOR_GREEN if is_correct else COLOR_RED


def frame_passed_checklist(checklist_result):
    total = checklist_result.get("total_measured", 0)
    passed = checklist_result.get("passed", 0)
    if total == 0:
        return None
    return passed >= (total / 2.0)


def draw_skeleton_with_feedback(image, landmarks_proto, is_correct):
    color = color_for_state(is_correct)
    landmark_style = mp_drawing.DrawingSpec(color=color, thickness=4, circle_radius=4)
    connection_style = mp_drawing.DrawingSpec(color=color, thickness=3)
    mp_drawing.draw_landmarks(
        image, landmarks_proto, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=landmark_style,
        connection_drawing_spec=connection_style,
    )


def draw_teach_banner(image, seconds_remaining):
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 70), (40, 40, 40), -1)
    cv2.putText(image, "WATCH: Correct Sit to Stand Form", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Live demo starts in {int(seconds_remaining)}s  (press SPACE to skip)",
                (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)


def check_label(name, status):
    if status is None:
        return f"{name}: --", (160, 160, 160)
    if status:
        return f"{name}: OK", COLOR_GREEN
    return f"{name}: FIX", COLOR_RED


def draw_hud(image, checklist_result, latency_ms, fps, rep_count, correct_rep_count):
    h, w = image.shape[:2]

    cv2.rectangle(image, (0, 0), (w, 60), (40, 40, 40), -1)
    cv2.putText(image, EXERCISE_NAME, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, f"Reps: {correct_rep_count}/{rep_count}", (w - 480, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    score_color = (COLOR_GREEN if checklist_result.get("passed", 0) == checklist_result.get("total_measured", 0)
                   and checklist_result.get("total_measured", 0) > 0 else COLOR_AMBER)
    cv2.putText(image, f"Score: {checklist_result['score_text']}", (w - 260, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2, cv2.LINE_AA)

    y = h - 110
    for name, key in [("Legs", "legs"), ("Back", "back"), ("Arms", "arms")]:
        label, color = check_label(name, checklist_result.get(key))
        cv2.putText(image, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 30

    cv2.putText(image, f"Latency: {latency_ms:.1f} ms   FPS: {fps:.1f}", (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_summary_screen(image, total_reps, correct_reps, fail_counts):
    h, w = image.shape[:2]
    image[:] = (25, 25, 25)

    accuracy = (correct_reps / total_reps * 100) if total_reps > 0 else 0.0

    cv2.putText(image, "SESSION COMPLETE", (int(w * 0.5) - 220, int(h * 0.18)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, f"Total reps: {total_reps}", (int(w * 0.5) - 180, int(h * 0.32)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Correct reps: {correct_reps}", (int(w * 0.5) - 180, int(h * 0.40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2, cv2.LINE_AA)

    acc_color = COLOR_GREEN if accuracy >= 70 else (COLOR_AMBER if accuracy >= 40 else COLOR_RED)
    cv2.putText(image, f"Accuracy: {accuracy:.1f}%", (int(w * 0.5) - 180, int(h * 0.50)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, acc_color, 3, cv2.LINE_AA)

    cv2.putText(image, "Most common issue(s):", (int(w * 0.5) - 180, int(h * 0.62)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if sum(fail_counts.values()) == 0:
        cv2.putText(image, "  None - great form throughout!", (int(w * 0.5) - 180, int(h * 0.69)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2, cv2.LINE_AA)
    else:
        sorted_fails = sorted(fail_counts.items(), key=lambda x: -x[1])
        y = int(h * 0.69)
        for name, count in sorted_fails:
            if count > 0:
                cv2.putText(image, f"  {name}: flagged in {count} frame(s)", (int(w * 0.5) - 180, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2, cv2.LINE_AA)
                y += 32

    cv2.putText(image, "Press 'q' to close", (int(w * 0.5) - 130, int(h * 0.92)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2, cv2.LINE_AA)


# ============================================================
# PHASE 1: TEACH - plays the recorded clip, no scoring
# ============================================================
def run_teach_phase(screen_w, screen_h):
    file_path = os.path.join(DATASET_FOLDER, TEACH_CLIP_FILE)
    if not os.path.exists(file_path):
        print(f"[ERROR] Teaching clip not found: {file_path}")
        print("Check DATASET_FOLDER and TEACH_CLIP_FILE. Skipping teach phase.")
        return

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_path, repeat_playback=True)

    started = False
    try:
        profile = pipeline.start(config)
        started = True
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)  # plays at normal human-watchable speed

        start_time = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - start_time
            remaining = TEACH_CLIP_DURATION_SEC - elapsed
            if remaining <= 0:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break

            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue  # looped clip, just keep going until time is up

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            display_image = cv2.resize(color_image, (screen_w, screen_h))
            draw_teach_banner(display_image, remaining)
            cv2.imshow(WINDOW_NAME, display_image)

    finally:
        if started:
            pipeline.stop()


# ============================================================
# PHASE 2: PRACTICE - live camera, real scoring
# ============================================================
def run_practice_phase(screen_w, screen_h):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, LIVE_FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, LIVE_FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

    detector = SitToStandDetectorV2(depth_intrinsics=intrinsics)
    checklist = SitToStandChecklist(
        leg_stand_threshold=LEG_STAND_THRESHOLD, leg_sit_threshold=LEG_SIT_THRESHOLD,
        back_min_angle=BACK_MIN_ANGLE, arm_raise_min=ARM_RAISE_MIN, arm_raise_max=ARM_RAISE_MAX,
    )

    leg_state = "standing"
    rep_count = 0
    correct_rep_count = 0
    current_rep_frame_results = []
    session_fail_counts = {"Legs": 0, "Back": 0, "Arms": 0}

    latency_window = []
    WINDOW_SIZE = 30

    print("Practice phase started. Press 'q' to end the session.")

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            display_image = cv2.resize(color_image, (screen_w, screen_h))

            # ===================== LATENCY TIMER =====================
            start_time = time.perf_counter()
            result = detector.process_frame(color_image, depth_frame, depth_image, side="auto")
            checklist_result = checklist.evaluate(result)
            end_time = time.perf_counter()
            # ===========================================================

            latency_ms = (end_time - start_time) * 1000
            latency_window.append(latency_ms)
            if len(latency_window) > WINDOW_SIZE:
                latency_window.pop(0)
            avg_latency = sum(latency_window) / len(latency_window)
            live_fps = 1000 / avg_latency if avg_latency > 0 else 0

            if result is not None and result["leg_angle_deg"] is not None:
                leg_angle = result["leg_angle_deg"]

                frame_ok = frame_passed_checklist(checklist_result)
                if frame_ok is not None:
                    current_rep_frame_results.append(frame_ok)

                if checklist_result.get("legs") is False:
                    session_fail_counts["Legs"] += 1
                if checklist_result.get("back") is False:
                    session_fail_counts["Back"] += 1
                if checklist_result.get("arms") is False:
                    session_fail_counts["Arms"] += 1

                if leg_state == "standing" and leg_angle <= LEG_SIT_THRESHOLD:
                    leg_state = "sitting"
                elif leg_state == "sitting" and leg_angle >= LEG_STAND_THRESHOLD:
                    leg_state = "standing"
                    rep_count += 1

                    if current_rep_frame_results:
                        pass_fraction = sum(current_rep_frame_results) / len(current_rep_frame_results)
                        if pass_fraction >= REP_CORRECT_FRACTION:
                            correct_rep_count += 1
                    current_rep_frame_results = []

                overall_ok = frame_ok if frame_ok is not None else True
                draw_skeleton_with_feedback(display_image, result["landmarks"], overall_ok)

            draw_hud(display_image, checklist_result, avg_latency, live_fps, rep_count, correct_rep_count)
            cv2.imshow(WINDOW_NAME, display_image)

    finally:
        pipeline.stop()
        detector.close()

    return rep_count, correct_rep_count, session_fail_counts


def main():
    screen_w, screen_h = get_screen_resolution()

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"Starting {EXERCISE_NAME} session")
    print(f"Phase 1: teaching clip ({TEACH_CLIP_DURATION_SEC}s, press SPACE to skip)")
    run_teach_phase(screen_w, screen_h)

    print("Phase 2: switching to live camera for practice")
    rep_count, correct_rep_count, session_fail_counts = run_practice_phase(screen_w, screen_h)

    accuracy = (correct_rep_count / rep_count * 100) if rep_count > 0 else 0.0
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"Total reps: {rep_count}")
    print(f"Correct reps: {correct_rep_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Issue tally: {session_fail_counts}")

    # Show the summary screen until the person closes it.
    summary_image = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    draw_summary_screen(summary_image, rep_count, correct_rep_count, session_fail_counts)
    cv2.imshow(WINDOW_NAME, summary_image)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()